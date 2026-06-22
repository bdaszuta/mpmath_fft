"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Real-valued FFT transforms for mpmath.

rfft, irfft, hfft, ihfft, rfftfreq -- follow numpy.fft conventions.
Input/output are ndarray[dtype=object] of mp.mpf / mp.mpc values.
"""

import mpmath as mp
import numpy as np


def _fft(x, axis):
  """Lazy import wrapper to avoid circular import with __init__.py."""
  from mpmath_fft import fft as _f
  return _f(x, axis=axis)


def _ifft(x, axis):
  """Lazy import wrapper to avoid circular import with __init__.py."""
  from mpmath_fft import ifft as _f
  return _f(x, axis=axis)


def _complexify(x, axis, n_new):
  """
  Convert real-valued mp.mpf object array to complex mp.mpc,
  zero-padding or truncating to length n_new along axis.

  Parameters
  ----------
  x : ndarray[dtype=object] of mp.mpf
  axis : int
    Transform axis (non-negative).
  n_new : int
    Desired length along axis.

  Returns
  -------
  xc : ndarray[dtype=object] of mp.mpc
    Complexified array with xc.shape[axis] == n_new.
  """
  nd = x.ndim
  n_orig = x.shape[axis]

  new_shape = list(x.shape)
  new_shape[axis] = n_new
  xc = np.empty(new_shape, dtype=object)

  if nd == 1:
    n_copy = min(n_orig, n_new)
    for i in range(n_copy):
      val = x[i]
      xc[i] = mp.mpc(val, 0)
    for i in range(n_copy, n_new):
      xc[i] = mp.mpc(0)
    return xc

  # ND case: iterate all multi-indices
  for idx in np.ndindex(*new_shape):
    k = idx[axis]
    if k < n_orig:
      val = x[idx]
      xc[idx] = mp.mpc(val, 0)
    else:
      xc[idx] = mp.mpc(0)
  return xc


def _hermitian_complete(x_half_1d, n_out):
  """
  Reconstruct a full-length Hermitian-symmetric spectrum from
  its non-redundant half.

  Parameters
  ----------
  x_half_1d : ndarray[dtype=object] of mp.mpc, shape (M,)
    Half-spectrum: X[0], X[1], ..., X[M-1] where M = n_out//2 + 1.
  n_out : int
    Full spectrum length.

  Returns
  -------
  y : ndarray[dtype=object] of mp.mpc, shape (n_out,)
    Full spectrum with Hermitian symmetry: y[n_out-k] = conj(y[k]).
  """
  M = len(x_half_1d)
  n2 = n_out // 2

  y = np.empty(n_out, dtype=object)

  # Positive frequencies (including DC; Nyquist when n_out even)
  y[0] = x_half_1d[0]
  for k in range(1, n2 + 1):
    if k < M:
      y[k] = x_half_1d[k]
    else:
      y[k] = mp.mpc(0)

  # Negative frequencies via Hermitian symmetry
  for k in range(n2 + 1, n_out):
    y[k] = y[n_out - k].conjugate()

  return y


def _conjugate_array(x):
  """
  Element-wise conjugate of an object array of mp.mpc.

  Returns a new array.
  """
  xc = np.empty(x.shape, dtype=object)
  for i in range(xc.size):
    xc.flat[i] = x.flat[i].conjugate()
  return xc


def rfft(x, n=None, axis=-1):
  """
  Real-to-complex FFT.

  Computes the forward DFT of real-valued input and returns only
  the non-redundant half of the spectrum (indices 0..n//2).

  Parameters
  ----------
  x : ndarray[dtype=object] of mp.mpf
    Real-valued input array.
  n : int, optional
    Transform length.  If n > x.shape[axis], zero-pad.
    If n < x.shape[axis], truncate.  Default: x.shape[axis].
  axis : int
    Transform axis (default -1).

  Returns
  -------
  y : ndarray[dtype=object] of mp.mpc
    Half-spectrum, shape = (..., n//2 + 1, ...).
  """
  if not isinstance(x, np.ndarray):
    raise TypeError(
        f"rfft requires numpy ndarray, got {type(x).__name__}")
  nd = x.ndim
  if nd == 0:
    raise ValueError("rfft requires at least 1 dimension")

  if x.dtype != np.dtype('object'):
    raise TypeError(
        f"rfft requires dtype=object array, got {x.dtype}")

  if not isinstance(axis, int):
    raise TypeError(
        f"rfft axis must be int, got {type(axis).__name__}")
  ax = axis if axis >= 0 else nd + axis
  if ax < 0 or ax >= nd:
    raise ValueError(
        f"axis {axis} out of bounds for array with {nd} dimensions")

  n_orig = x.shape[ax]
  n_fft = n if n is not None else n_orig
  if n_fft <= 0:
    raise ValueError("rfft requires positive transform length")

  # Validate element type
  if x.size > 0:
    first = x.flat[0]
    if not isinstance(first, mp.mpf):
      raise TypeError(
          f"rfft requires array elements of type mp.mpf, "
          f"got {type(first).__name__} at index 0")

  # Complexify: real -> complex, pad/truncate
  xc = _complexify(x, ax, n_fft)

  # Forward FFT
  y_full = _fft(xc, axis=ax)

  # Slice to half-spectrum
  n_half = n_fft // 2 + 1
  slices = [slice(None)] * nd
  slices[ax] = slice(0, n_half)
  return y_full[tuple(slices)]


def irfft(x, n=None, axis=-1):
  """
  Complex-to-real inverse FFT.

  Reconstructs the full Hermitian-symmetric spectrum from the
  non-redundant half, computes the inverse DFT, and returns the
  real part.

  Parameters
  ----------
  x : ndarray[dtype=object] of mp.mpc
    Half-spectrum, shape = (..., M, ...) where M = n_orig//2 + 1.
  n : int, optional
    Output length along transform axis.
    Default: 2 * (x.shape[axis] - 1).
  axis : int
    Transform axis (default -1).

  Returns
  -------
  y : ndarray[dtype=object] of mp.mpf
    Real-valued inverse transform.
  """
  if not isinstance(x, np.ndarray):
    raise TypeError(
        f"irfft requires numpy ndarray, got {type(x).__name__}")
  nd = x.ndim
  if nd == 0:
    raise ValueError("irfft requires at least 1 dimension")

  if x.dtype != np.dtype('object'):
    raise TypeError(
        f"irfft requires dtype=object array, got {x.dtype}")

  if not isinstance(axis, int):
    raise TypeError(
        f"irfft axis must be int, got {type(axis).__name__}")
  ax = axis if axis >= 0 else nd + axis
  if ax < 0 or ax >= nd:
    raise ValueError(
        f"axis {axis} out of bounds for array with {nd} dimensions")

  M = x.shape[ax]
  if M == 0:
    raise ValueError(
        "irfft requires at least 1 element along transform axis")

  n_out = n if n is not None else 2 * (M - 1)
  if n_out <= 0:
    raise ValueError("irfft requires positive output length")

  # Validate element type
  if x.size > 0:
    first = x.flat[0]
    if not isinstance(first, mp.mpc):
      raise TypeError(
          f"irfft requires array elements of type mp.mpc, "
          f"got {type(first).__name__} at index 0")

  n_full = n_out

  # Hermitian completion for each slice along non-axis dims
  full_shape = list(x.shape)
  full_shape[ax] = n_full
  y_full = np.empty(full_shape, dtype=object)

  # Strategy: for each position in non-axis dims, do 1D hermitian completion
  # Build an iterator over non-axis positions
  non_ax_shape = list(x.shape)
  non_ax_shape[ax] = 1

  for base_idx in np.ndindex(*non_ax_shape):
    # Extract 1D half-spectrum
    x_row = np.empty(M, dtype=object)
    full_idx = list(base_idx)
    for k in range(M):
      full_idx[ax] = k
      x_row[k] = x[tuple(full_idx)]

    # Hermitian completion
    y_row = _hermitian_complete(x_row, n_full)

    # Place back
    for k in range(n_full):
      full_idx[ax] = k
      y_full[tuple(full_idx)] = y_row[k]

  # Inverse FFT
  z = _ifft(y_full, axis=ax)

  # Extract real part as mp.mpf
  out_shape = list(x.shape)
  out_shape[ax] = n_out
  out = np.empty(out_shape, dtype=object)
  for idx in np.ndindex(*out_shape):
    out[idx] = z[idx].real

  return out


def hfft(x, n=None, axis=-1):
  """
  Hermitian FFT.

  Computes the inverse FFT of a Hermitian-symmetric half-spectrum.
  Equivalent to n_used * irfft(conj(x), n=n, axis=axis).

  Parameters
  ----------
  x : ndarray[dtype=object] of mp.mpc
    Half-spectrum input.
  n : int, optional
    Output length.  Default: 2 * (x.shape[axis] - 1).
  axis : int
    Transform axis (default -1).

  Returns
  -------
  y : ndarray[dtype=object] of mp.mpf
    Real-valued output.
  """
  # --- Input validation ---
  if not isinstance(x, np.ndarray):
    raise TypeError(
        f"hfft requires numpy ndarray, got {type(x).__name__}")
  nd = x.ndim
  if nd == 0:
    raise ValueError("hfft requires at least 1 dimension")

  if x.dtype != np.dtype('object'):
    raise TypeError(
        f"hfft requires dtype=object array, got {x.dtype}")

  if x.size > 0 and not isinstance(x.flat[0], mp.mpc):
    raise TypeError(
        f"hfft requires array elements of type mp.mpc, "
        f"got {type(x.flat[0]).__name__} at index 0")

  if not isinstance(axis, int):
    raise TypeError(
        f"hfft axis must be int, got {type(axis).__name__}")
  ax = axis if axis >= 0 else nd + axis
  if ax < 0 or ax >= nd:
    raise ValueError(
        f"axis {axis} out of bounds for array with {nd} dimensions")

  M = x.shape[ax]
  if M == 0:
    raise ValueError(
        "hfft requires at least 1 element along transform axis")

  n_out = n if n is not None else 2 * (M - 1)

  # hfft(X) = n_out * irfft(conj(X), n=n_out)
  xc = _conjugate_array(x)
  y = irfft(xc, n=n_out, axis=axis)
  for i in range(y.size):
    y.flat[i] = mp.mpf(n_out) * y.flat[i]
  return y


def ihfft(x, n=None, axis=-1):
  """
  Inverse Hermitian FFT.

  Computes the forward FFT of a real signal and returns the
  non-redundant half-spectrum, with 1/n scaling.
  Equivalent to conj(rfft(x, n=n, axis=axis)) / n_used.

  Parameters
  ----------
  x : ndarray[dtype=object] of mp.mpf
    Real-valued input.
  n : int, optional
    Transform length.  Default: x.shape[axis].
  axis : int
    Transform axis (default -1).

  Returns
  -------
  y : ndarray[dtype=object] of mp.mpc
    Half-spectrum output.
  """
  # --- Input validation ---
  if not isinstance(x, np.ndarray):
    raise TypeError(
        f"ihfft requires numpy ndarray, got {type(x).__name__}")
  nd = x.ndim
  if nd == 0:
    raise ValueError("ihfft requires at least 1 dimension")

  if x.dtype != np.dtype('object'):
    raise TypeError(
        f"ihfft requires dtype=object array, got {x.dtype}")

  if x.size > 0 and not isinstance(x.flat[0], mp.mpf):
    raise TypeError(
        f"ihfft requires array elements of type mp.mpf, "
        f"got {type(x.flat[0]).__name__} at index 0")

  if not isinstance(axis, int):
    raise TypeError(
        f"ihfft axis must be int, got {type(axis).__name__}")
  ax = axis if axis >= 0 else nd + axis
  if ax < 0 or ax >= nd:
    raise ValueError(
        f"axis {axis} out of bounds for array with {nd} dimensions")

  n_fft = n if n is not None else x.shape[ax]
  if n_fft <= 0:
    raise ValueError("ihfft requires positive transform length")

  # ihfft(x) = conj(rfft(x, n)) / n_used
  y = rfft(x, n=n_fft, axis=axis)
  for i in range(y.size):
    y.flat[i] = y.flat[i].conjugate() / n_fft
  return y


def rfftfreq(n, d=1.0):
  """
  Frequency bins for an n-point real FFT.

  Returns the non-negative frequencies f_k = k / (n*d) for
  k = 0..n//2.

  Parameters
  ----------
  n : int
    Number of sample points.
  d : float or mpmath scalar, optional
    Sample spacing (default 1.0).

  Returns
  -------
  f : ndarray[dtype=object] of mp.mpf
    Frequencies, shape (n//2 + 1,).
  """
  if n <= 0:
    raise ValueError(
        f"rfftfreq requires positive n, got {n}")
  val = mp.mpf(1) / (n * d)
  n_out = n // 2 + 1
  f = np.empty(n_out, dtype=object)
  for i in range(n_out):
    f[i] = mp.mpf(i) * val
  return f


#
# :D
#
