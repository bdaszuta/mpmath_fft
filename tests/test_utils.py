"""
Test mpmath_fft spectral utilities and input validation.

Tests cover:
  - fftshift / ifftshift vs numpy.fft
  - fftfreq vs numpy.fft
  - Input validation for fft, ifft, fftshift, ifftshift, fftfreq

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
"""
import mpmath as mp
import numpy as np
from numpy.testing import assert_allclose

import pytest

from mpmath_fft import fft, ifft
from mpmath_fft import fftshift, ifftshift, fftfreq

from tests.helpers import _to_mpc, _to_np, _err_inf


# ---------------------------------------------------------------------------
# fftshift / ifftshift
# ---------------------------------------------------------------------------

def test_fftshift_1d_even():
  mp.mp.dps = 15
  x_np = np.arange(8, dtype=complex)
  x = _to_mpc(x_np)
  y = fftshift(x)
  assert_allclose(_to_np(y), np.fft.fftshift(x_np))


def test_fftshift_1d_odd():
  mp.mp.dps = 15
  x_np = np.arange(7, dtype=complex)
  x = _to_mpc(x_np)
  y = fftshift(x)
  assert_allclose(_to_np(y), np.fft.fftshift(x_np))


def test_fftshift_2d():
  mp.mp.dps = 15
  x_np = np.arange(12, dtype=complex).reshape(3, 4)
  x_obj = _to_mpc(x_np)
  y_obj = fftshift(x_obj)
  assert_allclose(_to_np(y_obj), np.fft.fftshift(x_np))


def test_ifftshift_undoes_fftshift():
  mp.mp.dps = 15
  for N in [7, 8, 16, 17]:
    x = _to_mpc(np.arange(N, dtype=complex))
    assert _err_inf(ifftshift(fftshift(x)), x) < 1e-15


# ---------------------------------------------------------------------------
# fftfreq
# ---------------------------------------------------------------------------

def test_fftfreq_even():
  mp.mp.dps = 15
  f = fftfreq(8, d=0.5)
  f_np = np.fft.fftfreq(8, d=0.5)
  assert_allclose(_to_np(f), f_np, atol=1e-15)


def test_fftfreq_odd():
  mp.mp.dps = 15
  f = fftfreq(7, d=1.0)
  f_np = np.fft.fftfreq(7, d=1.0)
  assert_allclose(_to_np(f), f_np, atol=1e-15)


def test_fftfreq_high_precision():
  mp.mp.dps = 50
  f = fftfreq(16, d=1.0)
  assert abs(f[0]) < 1e-50
  assert abs(f[8] + mp.mpf('0.5')) < 1e-50


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_fft_rejects_non_object_dtype():
  mp.mp.dps = 15
  x = np.arange(8, dtype=complex)
  with pytest.raises(TypeError, match='dtype=object'):
    fft(x)


def test_fft_rejects_bad_axis():
  mp.mp.dps = 15
  x = _to_mpc(np.arange(8, dtype=complex))
  with pytest.raises(ValueError, match='out of bounds'):
    fft(x, axis=5)


def test_fft_rejects_empty_axis():
  mp.mp.dps = 15
  x = np.empty((3, 0), dtype=object)
  with pytest.raises(ValueError, match='at least 1 element'):
    fft(x, axis=1)


def test_fft_rejects_non_mpc_elements():
  mp.mp.dps = 15
  x = np.array([1, 2, 3], dtype=object)
  with pytest.raises(TypeError, match='mp\\.mpc'):
    fft(x)


def test_ifft_rejects_non_object_dtype():
  mp.mp.dps = 15
  x = np.arange(8, dtype=complex)
  with pytest.raises(TypeError, match='dtype=object'):
    ifft(x)


def test_ifft_rejects_non_mpc_elements():
  mp.mp.dps = 15
  x = np.array(['a', 'b'], dtype=object)
  with pytest.raises(TypeError, match='mp\\.mpc'):
    ifft(x)


def test_ifft_rejects_bad_axis():
  mp.mp.dps = 15
  x = _to_mpc(np.arange(8, dtype=complex))
  with pytest.raises(ValueError, match='out of bounds'):
    ifft(x, axis=5)


def test_ifft_rejects_empty_axis():
  mp.mp.dps = 15
  x = np.empty((3, 0), dtype=object)
  with pytest.raises(ValueError, match='at least 1 element'):
    ifft(x, axis=1)


def test_fftshift_rejects_non_object_dtype():
    mp.mp.dps = 15
    x = np.arange(8, dtype=complex)
    with pytest.raises(TypeError, match='dtype=object'):
        fftshift(x)


def test_fftshift_rejects_non_ndarray():
    mp.mp.dps = 15
    with pytest.raises(TypeError, match='numpy ndarray'):
        fftshift([1, 2, 3])


def test_ifftshift_rejects_non_object_dtype():
    mp.mp.dps = 15
    x = np.arange(8, dtype=complex)
    with pytest.raises(TypeError, match='dtype=object'):
        ifftshift(x)


def test_ifftshift_rejects_non_ndarray():
    mp.mp.dps = 15
    with pytest.raises(TypeError, match='numpy ndarray'):
        ifftshift([1, 2, 3])


def test_fftfreq_rejects_zero():
    mp.mp.dps = 15
    with pytest.raises(ValueError, match='positive'):
        fftfreq(0)
