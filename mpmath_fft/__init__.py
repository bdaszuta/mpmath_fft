"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Arbitrary-precision FFT for mpmath.

Public API: fft, ifft, rfft, irfft, hfft, ihfft, build_plan,
           clear_plan_cache, fftshift, ifftshift, fftfreq, rfftfreq.
Follows pynalgo.fft conventions exactly: same function signatures,
same plan structure, same stack-based executor.  Uses mpmath types
stored in numpy object arrays.
"""
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("mpmath_fft")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
  'build_plan',
  'fft',
  'ifft',
  'rfft',
  'irfft',
  'hfft',
  'ihfft',
  'clear_plan_cache',
  'fftshift',
  'ifftshift',
  'fftfreq',
  'rfftfreq',
]

import mpmath as _mp
import numpy as _np
import threading as _threading

from mpmath_fft._kernels import _build_plan, _fft_exec
from mpmath_fft._real_transforms import (
    rfft, irfft, hfft, ihfft, rfftfreq,
)

# ---------------------------------------------------------------------------
# Precision-aware plan cache
# ---------------------------------------------------------------------------

_plan_cache: dict[tuple[int, int], tuple] = {}
_plan_cache_lock = _threading.Lock()
_MAX_PLAN_CACHE_SIZE = 128


def build_plan(N):
  """
  Build a FFT plan for transform of size N.

  Plans are cached by (N, mp.mp.dps) so twiddle factors are recomputed
  when mpmath precision changes.

  The cache holds at most _MAX_PLAN_CACHE_SIZE entries (default 128).
  When the cache is full, the oldest entry is evicted (FIFO).

  Thread-safe: uses double-checked locking to protect the plan cache.

  Parameters
  ----------
  N : int
    Transform size.

  Returns
  -------
  plan : tuple of ndarray
    Plan arrays (type, N, p1, p2, c1, c2, tw_data, tw_start).
  """
  dps = _mp.mp.dps
  if N <= 0:
    raise ValueError(
        f"build_plan requires positive N, got {N}")
  key = (N, dps)
  # Fast path: no lock for cache hits
  plan = _plan_cache.get(key)
  if plan is not None:
    return plan
  # Build outside lock so concurrent plan builds don't serialize
  plan = _build_plan(N)
  with _plan_cache_lock:
    # Check again inside lock (another thread may have inserted)
    if key not in _plan_cache:
      _plan_cache[key] = plan
      # FIFO eviction when cache exceeds max size
      if len(_plan_cache) > _MAX_PLAN_CACHE_SIZE:
        _plan_cache.pop(next(iter(_plan_cache)))
    return _plan_cache[key]


def clear_plan_cache():
  """
  Clear the plan cache.

  All cached plans are discarded.  Subsequent calls to build_plan()
  will reconstruct plans at the current mpmath precision.

  Thread-safe: acquires the plan cache lock.
  """
  with _plan_cache_lock:
    _plan_cache.clear()


def fftshift(x, axes=None):
  """
  Shift zero-frequency component to center of spectrum.

  Swaps half-spaces for each specified axis.  Operates on object
  arrays (copies the input).

  Parameters
  ----------
  x : ndarray[dtype=object]
    Input array.
  axes : int or tuple of int, optional
    Axes to shift.  Default: all axes.

  Returns
  -------
  y : ndarray[dtype=object]
    Shifted array.

  Examples
  --------
  >>> import mpmath as mp
  >>> import numpy as np
  >>> x = np.array([mp.mpc(0), mp.mpc(1), mp.mpc(2), mp.mpc(3)], dtype=object)
  >>> y = fftshift(x)
  >>> [complex(v) for v in y.flat]
  [(2+0j), (3+0j), 0j, (1+0j)]
  """
  if not isinstance(x, _np.ndarray):
    raise TypeError(
        f"fftshift requires numpy ndarray, got {type(x).__name__}")
  if x.dtype != _np.dtype('object'):
    raise TypeError(
        f"fftshift requires dtype=object array, got {x.dtype}")
  y = x.copy()
  if axes is None:
    axes = tuple(range(x.ndim))
  elif isinstance(axes, int):
    axes = (axes,)
  for ax in axes:
    n = x.shape[ax]
    p2 = n // 2
    y = _np.roll(y, shift=p2, axis=ax)
  return y


def ifftshift(x, axes=None):
  """
  Inverse of fftshift.

  Undoes the shift so that zero frequency returns to index 0.

  Parameters
  ----------
  x : ndarray[dtype=object]
    Input array.
  axes : int or tuple of int, optional
    Axes to unshift.  Default: all axes.

  Returns
  -------
  y : ndarray[dtype=object]
    Unshifted array.
  """
  if not isinstance(x, _np.ndarray):
    raise TypeError(
        f"ifftshift requires numpy ndarray, got {type(x).__name__}")
  if x.dtype != _np.dtype('object'):
    raise TypeError(
        f"ifftshift requires dtype=object array, got {x.dtype}")
  y = x.copy()
  if axes is None:
    axes = tuple(range(x.ndim))
  elif isinstance(axes, int):
    axes = (axes,)
  for ax in axes:
    n = x.shape[ax]
    p2 = -(n // 2)
    y = _np.roll(y, shift=p2, axis=ax)
  return y


def fftfreq(n, d=1.0):
  """
  Frequency bins for an n-point FFT.

  Returns frequencies f_k = k / (n*d) for k = 0..n-1, wrapped
  so that f_k = (k-n) / (n*d) for k > n/2 (negative frequencies).

  Parameters
  ----------
  n : int
    Number of sample points.
  d : float or mpmath scalar, optional
    Sample spacing (default 1.0).  Inverse of sampling rate.

  Returns
  -------
  f : ndarray[dtype=object]
    Array of mp.mpf values of length n.

  Examples
  --------
  >>> import mpmath as mp
  >>> mp.mp.dps = 15
  >>> f = fftfreq(4, d=1.0)
  >>> [float(v) for v in f.flat]
  [0.0, 0.25, -0.5, -0.25]
  """
  if n <= 0:
    raise ValueError(
        f"fftfreq requires positive n, got {n}")
  val = _mp.mpf(1) / (n * d)
  f = _np.empty(n, dtype=object)
  N_mid = (n + 1) // 2
  for i in range(N_mid):
    f[i] = _mp.mpf(i) * val
  for i in range(N_mid, n):
    f[i] = _mp.mpf(i - n) * val
  return f


# ---------------------------------------------------------------------------
# fft / ifft
# ---------------------------------------------------------------------------

def fft(x, axis=-1):
  """
  Discrete Fourier Transform via Cooley-Tukey / radix-2 / Bluestein
  decomposition.  Operates on numpy object arrays of mpmath mpc values.

  Parameters
  ----------
  x : ndarray[dtype=object]
    Input array of mpc values.
  axis : int
    Axis along which to compute the FFT (default -1).

  Returns
  -------
  y : ndarray[dtype=object]
    Transformed array.

  Examples
  --------
  >>> import mpmath as mp
  >>> import numpy as np
  >>> mp.mp.dps = 15
  >>> x = np.array([mp.mpc(1, 0), mp.mpc(2, 0)], dtype=object)
  >>> y = fft(x)
  >>> [complex(v) for v in y.flat]
  [(3+0j), (-1+0j)]
  """
  if not isinstance(x, _np.ndarray):
    raise TypeError(
        f"fft requires numpy ndarray, got {type(x).__name__}")
  nd = x.ndim
  if nd == 0:
    raise ValueError("fft requires at least 1 dimension")

  if x.dtype != _np.dtype('object'):
    raise TypeError(
        f"fft requires dtype=object array, got {x.dtype}")

  if x.size > 0 and not isinstance(x.flat[0], _mp.mpc):
    raise TypeError(
        f"fft requires array elements of type mp.mpc, "
        f"got {type(x.flat[0]).__name__} at index 0")

  if not isinstance(axis, int):
    raise TypeError(
        f"fft axis must be int, got {type(axis).__name__}")
  ax = axis if axis >= 0 else nd + axis
  if ax < 0 or ax >= nd:
    raise ValueError(
        f"axis {axis} out of bounds for array with {nd} dimensions")

  n_fft = x.shape[ax]
  if n_fft == 0:
    raise ValueError(
        f"fft requires at least 1 element along transform axis "
        f"(axis {axis}, shape {x.shape})")

  if nd == 1:
    # Wrap in (1, N), exec, unwrap
    x_2d = _np.empty((1, n_fft), dtype=object)
    for i in range(n_fft):
      x_2d[0, i] = x[i]
    plan = build_plan(n_fft)
    _fft_exec(x_2d, plan)
    out = _np.empty(n_fft, dtype=object)
    for i in range(n_fft):
      out[i] = x_2d[0, i]
    return out

  # nd >= 2: generic stride-based reshaping
  # Compute source strides
  src_stride = _np.empty(nd, dtype=_np.int64)
  acc = _np.int64(1)
  for d in range(nd - 1, -1, -1):
    src_stride[d] = acc
    acc *= x.shape[d]

  # Batch size = product of all dims except ax
  batch = _np.int64(1)
  for d in range(nd):
    if d != ax:
      batch *= x.shape[d]

  n_total = acc

  # Flatten to (batch, n_fft)
  x_2d = _np.empty((batch, n_fft), dtype=object)
  for flat in range(n_total):
    rem = flat
    src_multi = _np.empty(nd, dtype=_np.int64)
    for d in range(nd):
      src_multi[d] = rem // src_stride[d]
      rem = rem % src_stride[d]

    dst_batch = _np.int64(0)
    dst_stride = _np.int64(1)
    for d in range(nd - 1, -1, -1):
      if d != ax:
        dst_batch += src_multi[d] * dst_stride
        dst_stride *= x.shape[d]
    dst_col = src_multi[ax]
    x_2d[dst_batch, dst_col] = x.flat[flat]

  # Execute
  plan = build_plan(n_fft)
  _fft_exec(x_2d, plan)

  # Copy back to output
  out = _np.empty(x.shape, dtype=object)
  for flat in range(n_total):
    rem = flat
    src_multi = _np.empty(nd, dtype=_np.int64)
    for d in range(nd):
      src_multi[d] = rem // src_stride[d]
      rem = rem % src_stride[d]

    dst_batch = _np.int64(0)
    dst_stride = _np.int64(1)
    for d in range(nd - 1, -1, -1):
      if d != ax:
        dst_batch += src_multi[d] * dst_stride
        dst_stride *= x.shape[d]
    dst_col = src_multi[ax]
    out.flat[flat] = x_2d[dst_batch, dst_col]

  return out


def ifft(x, axis=-1):
  """
  Inverse FFT: ifft(x) = conj(fft(conj(x))) / N.

  Parameters
  ----------
  x : ndarray[dtype=object]
    Input array of mpc values.
  axis : int
    Axis along which to compute the iFFT (default -1).

  Returns
  -------
  y : ndarray[dtype=object]
    Inverse transformed array.

  Examples
  --------
  >>> import mpmath as mp
  >>> import numpy as np
  >>> mp.mp.dps = 15
  >>> x = np.array([mp.mpc(1, 0), mp.mpc(2, 0)], dtype=object)
  >>> y = fft(x)
  >>> z = ifft(y)
  >>> [complex(v) for v in z.flat]
  [(1+0j), (2+0j)]
  """
  if not isinstance(x, _np.ndarray):
    raise TypeError(
        f"ifft requires numpy ndarray, got {type(x).__name__}")
  nd = x.ndim
  if nd == 0:
    raise ValueError("ifft requires at least 1 dimension")

  if x.dtype != _np.dtype('object'):
    raise TypeError(
        f"ifft requires dtype=object array, got {x.dtype}")

  if x.size > 0 and not isinstance(x.flat[0], _mp.mpc):
    raise TypeError(
        f"ifft requires array elements of type mp.mpc, "
        f"got {type(x.flat[0]).__name__} at index 0")

  if not isinstance(axis, int):
    raise TypeError(
        f"ifft axis must be int, got {type(axis).__name__}")
  ax = axis if axis >= 0 else nd + axis
  if ax < 0 or ax >= nd:
    raise ValueError(
        f"axis {axis} out of bounds for array with {nd} dimensions")

  n = x.shape[ax]
  if n == 0:
    raise ValueError(
        f"ifft requires at least 1 element along transform axis "
        f"(axis {axis}, shape {x.shape})")

  # conj(x)
  xc = _np.empty(x.shape, dtype=object)
  for i in range(xc.size):
    xc.flat[i] = x.flat[i].conjugate()

  # fft(conj(x))
  yc = fft(xc, axis)

  # conj(fft) / N
  for i in range(yc.size):
    yc.flat[i] = yc.flat[i].conjugate() / n

  return yc


#
# :D
#
