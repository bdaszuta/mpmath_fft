"""
Test mpmath_fft real-valued transforms: rfft, irfft, hfft, ihfft.

Tests cover:
  - rfft/irfft vs numpy.fft at dps=15
  - rfft vs fft half-spectrum equivalence
  - irfft roundtrip at dps=15, 50, 100
  - hfft/ihfft roundtrip and algebraic relationships
  - rfftfreq vs numpy.fft.rfftfreq
  - n parameter (padding/truncation)
  - ND axis dispatch
  - Input validation

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
"""
import mpmath as mp
import numpy as np
from numpy.testing import assert_allclose

import pytest

from mpmath_fft import fft, rfft, irfft, hfft, ihfft, rfftfreq

from tests.helpers import (
    _to_np, _err_inf, _make_real_signal, _conjugate_array,
)


# ---------------------------------------------------------------------------
# rfft vs numpy / fft half-spectrum
# ---------------------------------------------------------------------------

def test_rfft_vs_numpy():
  mp.mp.dps = 15
  for N in [4, 5, 8, 16, 32]:
    x_np = np.arange(N, dtype=float)
    x_obj = np.array([mp.mpf(v) for v in x_np], dtype=object)
    yr = rfft(x_obj)
    yr_np = np.fft.rfft(x_np)
    assert_allclose(_to_np(yr), yr_np, atol=1e-12)


def test_rfft_vs_fft_half():
  """rfft(x) == fft(x_complex)[:N//2+1]."""
  mp.mp.dps = 50
  for N in [4, 5, 8, 16]:
    x_obj = _make_real_signal(N)
    xc = np.array([mp.mpc(v, 0) for v in x_obj], dtype=object)
    yr = rfft(x_obj)
    yf = fft(xc)
    n_half = N // 2 + 1
    assert _err_inf(yr, yf[:n_half]) < 1e-48


# ---------------------------------------------------------------------------
# irfft roundtrip
# ---------------------------------------------------------------------------

def test_irfft_roundtrip_dps15():
  mp.mp.dps = 15
  for N in [4, 5, 8, 16, 17, 32]:
    x = _make_real_signal(N)
    yr = rfft(x)
    x_back = irfft(yr, n=N)
    err = _err_inf(x_back, x)
    assert err < 1e-11, f'N={N} roundtrip err={err:.1e}'


def test_irfft_roundtrip_dps50():
  mp.mp.dps = 50
  for N in [4, 5, 8, 16, 17]:
    x = _make_real_signal(N)
    yr = rfft(x)
    x_back = irfft(yr, n=N)
    err = _err_inf(x_back, x)
    assert err < 2e-47, f'N={N} roundtrip err={err:.1e}'


def test_irfft_roundtrip_dps100():
  """High-precision rfft/irfft roundtrip at dps=100."""
  mp.mp.dps = 100
  for N in [4, 8, 16]:
    x = _make_real_signal(N)
    yr = rfft(x)
    x_back = irfft(yr, n=N)
    err = _err_inf(x_back, x)
    assert err < 5e-95, f'N={N} roundtrip err={err:.1e}'


# ---------------------------------------------------------------------------
# hfft / ihfft
# ---------------------------------------------------------------------------

def test_hfft_ihfft_roundtrip():
  mp.mp.dps = 15
  for N in [4, 5, 8, 16, 17]:
    x = _make_real_signal(N)
    Xh = ihfft(x)
    x_back = hfft(Xh, n=N)
    err = _err_inf(x_back, x)
    assert err < 1e-11, f'N={N} hfft/ihfft err={err:.1e}'


def test_hfft_ihfft_roundtrip_dps50():
  """hfft/ihfft roundtrip at dps=50."""
  mp.mp.dps = 50
  for N in [4, 5, 8, 16, 17]:
    x = _make_real_signal(N)
    Xh = ihfft(x)
    x_back = hfft(Xh, n=N)
    err = _err_inf(x_back, x)
    assert err < 2e-47, f'N={N} hfft/ihfft err={err:.1e}'


def test_hfft_relationships():
  """Verify hfft(X) = N * irfft(conj(X)), ihfft(x) = conj(rfft(x)) / N."""
  mp.mp.dps = 50
  N = 8
  x = _make_real_signal(N)

  # ihfft(x) = conj(rfft(x)) / N
  Xh_direct = ihfft(x)
  Xr = rfft(x)
  for i in range(len(Xh_direct)):
    expected = Xr[i].conjugate() / N
    assert abs(Xh_direct[i] - expected) < 1e-48

  # hfft(X) = N * irfft(conj(X))
  x_h_direct = hfft(Xh_direct, n=N)
  x_h_via = irfft(_conjugate_array(Xh_direct), n=N)
  for i in range(N):
    assert abs(x_h_direct[i] - mp.mpf(N) * x_h_via[i]) < 1e-48


# ---------------------------------------------------------------------------
# rfftfreq
# ---------------------------------------------------------------------------

def test_rfftfreq_vs_numpy():
  mp.mp.dps = 15
  for N in [4, 5, 8, 16, 17]:
    for d in [1.0, 0.5]:
      f = rfftfreq(N, d=d)
      f_np = np.fft.rfftfreq(N, d=d)
      assert_allclose(_to_np(f), f_np, atol=1e-15)


# ---------------------------------------------------------------------------
# n parameter (padding / truncation)
# ---------------------------------------------------------------------------

def test_rfft_n_parameter():
  mp.mp.dps = 15
  x = _make_real_signal(8)

  # Pad: n=16
  yr = rfft(x, n=16)
  assert len(yr) == 9  # 16//2 + 1

  # Truncate: n=6
  yr = rfft(x, n=6)
  assert len(yr) == 4  # 6//2 + 1


def test_irfft_n_parameter():
  mp.mp.dps = 15
  x = _make_real_signal(8)
  yr = rfft(x)

  # Default: n = 2*(5-1) = 8
  x_back = irfft(yr)
  assert len(x_back) == 8

  # Explicit: n=10
  x_back = irfft(yr, n=10)
  assert len(x_back) == 10


def test_rfft_odd_n():
  mp.mp.dps = 15
  N = 9
  x = _make_real_signal(N)
  yr = rfft(x)
  n_half = N // 2 + 1  # 5
  assert len(yr) == n_half
  x_back = irfft(yr, n=N)
  assert _err_inf(x_back, x) < 1e-11


def test_rfft_padded_roundtrip():
  """rfft with padding (n > N): truncation after irfft recovers original."""
  mp.mp.dps = 50
  N = 8
  x = _make_real_signal(N)
  yr = rfft(x, n=2 * N)
  assert len(yr) == 2 * N // 2 + 1
  x_full = irfft(yr, n=2 * N)
  # First N elements must match original
  for i in range(N):
    assert abs(x_full[i] - x[i]) < 1e-48, f'i={i}'


def test_irfft_padded_output():
  """irfft with n > default against numpy."""
  mp.mp.dps = 15
  N = 8
  x_np = np.arange(N, dtype=float)
  x_obj = np.array([mp.mpf(v) for v in x_np], dtype=object)
  yr = rfft(x_obj)
  x_back = irfft(yr, n=14)
  assert len(x_back) == 14
  x_np_back = np.fft.irfft(np.fft.rfft(x_np), n=14)
  assert_allclose(_to_np(x_back), x_np_back, atol=1e-12)


# ---------------------------------------------------------------------------
# ND axis dispatch
# ---------------------------------------------------------------------------

def test_rfft_nd_axis():
  mp.mp.dps = 15
  x_np = np.arange(12, dtype=float).reshape(3, 4)
  x_obj = np.empty((3, 4), dtype=object)
  for i in range(3):
    for j in range(4):
      x_obj[i, j] = mp.mpf(x_np[i, j])

  # Axis 0: result shape (3//2+1=2, 4) for even N
  yr0 = rfft(x_obj, axis=0)
  assert yr0.shape == (2, 4)

  # Axis 1: result shape (3, 4//2+1=3)
  yr1 = rfft(x_obj, axis=1)
  assert yr1.shape == (3, 3)


def test_rfft_nd_vs_numpy():
  """2D and 3D rfft against numpy.fft.rfft."""
  mp.mp.dps = 15
  for shape, axes in [
    ((3, 4), [0, 1, -1]),
    ((2, 3, 4), [0, 1, 2]),
  ]:
    x_np = np.arange(np.prod(shape), dtype=float).reshape(shape)
    x_obj = np.empty(shape, dtype=object)
    for idx in np.ndindex(*shape):
      x_obj[idx] = mp.mpf(x_np[idx])
    for ax in axes:
      yr = rfft(x_obj, axis=ax)
      yr_np = np.fft.rfft(x_np, axis=ax)
      assert_allclose(_to_np(yr), yr_np, atol=1e-12,
                      err_msg=f'shape={shape} axis={ax}')


def test_irfft_nd_roundtrip():
  """2D and 3D rfft -> irfft roundtrip at dps=50."""
  mp.mp.dps = 50
  for shape, axes in [
    ((4, 6), [0, 1]),
    ((3, 5, 4), [0, 1, 2]),
  ]:
    x_np = np.arange(np.prod(shape), dtype=float).reshape(shape)
    x_obj = np.empty(shape, dtype=object)
    for idx in np.ndindex(*shape):
      x_obj[idx] = mp.mpf(x_np[idx])
    for ax in axes:
      yr = rfft(x_obj, axis=ax)
      x_back = irfft(yr, n=x_obj.shape[ax], axis=ax)
      err = _err_inf(x_back, x_obj)
      assert err < 1e-48, f'shape={shape} axis={ax} err={err:.1e}'


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_rfft_rejects_non_mpf():
  mp.mp.dps = 15
  x = np.array([mp.mpc(1, 0), mp.mpc(2, 0)], dtype=object)
  with pytest.raises(TypeError, match='mp\\.mpf'):
    rfft(x)


def test_irfft_rejects_non_mpc():
  mp.mp.dps = 15
  x = np.array([mp.mpf(1), mp.mpf(2)], dtype=object)
  with pytest.raises(TypeError, match='mp\\.mpc'):
    irfft(x)


def test_rfftfreq_rejects_zero():
    mp.mp.dps = 15
    from mpmath_fft import rfftfreq
    with pytest.raises(ValueError, match='positive'):
        rfftfreq(0)


def test_hfft_rejects_non_ndarray():
  mp.mp.dps = 15
  with pytest.raises(TypeError, match='numpy ndarray'):
    hfft([1, 2, 3])


def test_hfft_rejects_non_mpc():
  mp.mp.dps = 15
  x = np.array([mp.mpf(1), mp.mpf(2)], dtype=object)
  with pytest.raises(TypeError, match='mp\\.mpc'):
    hfft(x)


def test_ihfft_rejects_non_ndarray():
  mp.mp.dps = 15
  with pytest.raises(TypeError, match='numpy ndarray'):
    ihfft([1, 2, 3])


def test_ihfft_rejects_non_mpf():
  mp.mp.dps = 15
  x = np.array([mp.mpc(1, 0), mp.mpc(2, 0)], dtype=object)
  with pytest.raises(TypeError, match='mp\\.mpf'):
    ihfft(x)
