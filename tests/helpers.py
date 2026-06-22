"""
Shared test helpers for mpmath_fft tests.

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
"""
import mpmath as mp
import numpy as np


def _to_mpc(x_np):
  """Convert numpy complex array to mpmath object array."""
  if x_np.ndim == 1:
    return np.array(
      [mp.mpc(float(v.real), float(v.imag)) for v in x_np],
      dtype=object)
  out = np.empty(x_np.shape, dtype=object)
  for idx in range(x_np.size):
    v = x_np.flat[idx]
    out.flat[idx] = mp.mpc(float(v.real), float(v.imag))
  return out


def _to_np(x_obj):
  """Convert mpmath object array back to numpy complex."""
  return np.array([complex(v) for v in x_obj.flat],
                  dtype=complex).reshape(x_obj.shape)


def _make_test_signal(N):
  """Non-trivial signal for testing: a + bi with varying values."""
  x_obj = np.empty(N, dtype=object)
  for i in range(N):
    x_obj[i] = mp.mpc(2 * i + 1, 3 * N - 2 * i)
  return x_obj


def _err_inf(a_obj, b_obj):
  """L-infinity error between two object arrays."""
  return float(max(abs(a_obj.flat[i] - b_obj.flat[i])
                   for i in range(a_obj.size)))


def _make_real_signal(N):
  """Non-trivial real signal for testing."""
  x = np.empty(N, dtype=object)
  for i in range(N):
    x[i] = mp.mpf(2 * i + 1)
  return x


def _conjugate_array(x):
  """Element-wise conj for object array."""
  xc = np.empty(len(x), dtype=object)
  for i in range(len(x)):
    xc[i] = x[i].conjugate()
  return xc
