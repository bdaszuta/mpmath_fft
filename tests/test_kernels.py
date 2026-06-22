"""
Test mpmath_fft kernels: core FFT correctness and plan cache.

Tests cover:
  - Correctness vs numpy.fft at dps=15 (atol=1e-12)
  - Round-trip identity at high precision (dps=50, 100)
  - Naive DFT comparison at high precision
  - Precision convergence: error should decrease with increasing dps
  - ND axis dispatch (2D, 3D, 4D)
  - Plan cache (reuse, precision invalidation, eviction)
  - Thread safety

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
"""
import mpmath as mp
import numpy as np
from numpy.testing import assert_allclose

import pytest
import threading

from mpmath_fft import fft, ifft, build_plan
from mpmath_fft._kernels import _fft_naive

from tests.helpers import (
    _to_mpc, _to_np, _make_test_signal, _err_inf,
)


# ---------------------------------------------------------------------------
# Basic correctness vs numpy.fft (dps=15)
# ---------------------------------------------------------------------------

def test_vs_numpy_pow2():
  mp.mp.dps = 15
  for N in [2, 4, 8, 16, 32, 64, 128, 256]:
    x_np = np.arange(N, dtype=complex)
    x_obj = _to_mpc(x_np)
    assert_allclose(_to_np(fft(x_obj)), np.fft.fft(x_np), atol=1e-12)


def test_vs_numpy_unrolled():
  mp.mp.dps = 15
  for N in [1, 3, 4]:
    x_np = np.arange(N, dtype=complex)
    x_obj = _to_mpc(x_np)
    assert_allclose(_to_np(fft(x_obj)), np.fft.fft(x_np), atol=1e-12)


def test_vs_numpy_composite():
  mp.mp.dps = 15
  for N in [6, 10, 12, 20, 30, 60, 84, 100]:
    x_np = np.arange(N, dtype=complex)
    x_obj = _to_mpc(x_np)
    assert_allclose(_to_np(fft(x_obj)), np.fft.fft(x_np), atol=1e-12)


def test_vs_numpy_prime():
  mp.mp.dps = 15
  for N in [7, 11, 13, 17, 19, 23, 29, 37, 41, 43]:
    x_np = np.arange(N, dtype=complex)
    x_obj = _to_mpc(x_np)
    assert_allclose(_to_np(fft(x_obj)), np.fft.fft(x_np), atol=1e-12)


# ---------------------------------------------------------------------------
# Round-trip identity at multiple precisions
# ---------------------------------------------------------------------------

def test_roundtrip_dps15():
  mp.mp.dps = 15
  for N in [2, 3, 4, 8, 12, 17, 32, 100]:
    x = _make_test_signal(N)
    err = _err_inf(ifft(fft(x)), x)
    assert err < 1e-11, 'N=%d roundtrip err=%.1e' % (N, err)


def test_roundtrip_dps50():
  mp.mp.dps = 50
  for N in [2, 3, 4, 8, 12, 17, 32, 100]:
    x = _make_test_signal(N)
    err = _err_inf(ifft(fft(x)), x)
    assert err < 2e-47, 'N=%d roundtrip err=%.1e' % (N, err)


def test_roundtrip_dps100():
  mp.mp.dps = 100
  for N in [2, 3, 4, 8, 17, 32]:
    x = _make_test_signal(N)
    err = _err_inf(ifft(fft(x)), x)
    assert err < 5e-95, 'N=%d roundtrip err=%.1e' % (N, err)


# ---------------------------------------------------------------------------
# Comparison against naive DFT at high precision
# ---------------------------------------------------------------------------

def test_vs_naive_dps50():
  """Compare FFT output against O(N^2) naive DFT at dps=50."""
  mp.mp.dps = 50
  for N in [4, 5, 7, 8, 12]:
    x = _make_test_signal(N)
    # naive DFT
    x_2d = np.empty((1, N), dtype=object)
    for i in range(N):
      x_2d[0, i] = x[i]
    _fft_naive(x_2d, 0, 0, 1, N)
    ref = np.array([x_2d[0, i] for i in range(N)], dtype=object)

    y = fft(x)
    err = _err_inf(y, ref)
    assert err < 5e-48, 'N=%d vs naive err=%.1e' % (N, err)


def test_vs_naive_dps100():
  """Compare FFT output against O(N^2) naive DFT at dps=100."""
  mp.mp.dps = 100
  for N in [4, 5, 7, 8, 12]:
    x = _make_test_signal(N)
    x_2d = np.empty((1, N), dtype=object)
    for i in range(N):
      x_2d[0, i] = x[i]
    _fft_naive(x_2d, 0, 0, 1, N)
    ref = np.array([x_2d[0, i] for i in range(N)], dtype=object)

    y = fft(x)
    err = _err_inf(y, ref)
    assert err < 5e-95, 'N=%d vs naive err=%.1e' % (N, err)


# ---------------------------------------------------------------------------
# Precision convergence: error should decrease with increasing dps
# ---------------------------------------------------------------------------

def test_precision_convergence():
  """Compare vs naive DFT: higher dps must reduce error monotonically."""
  N = 8
  errors = []
  for dps in [15, 25, 35, 50, 75, 100]:
    mp.mp.dps = dps
    x_dps = np.empty(N, dtype=object)
    for i in range(N):
      x_dps[i] = mp.mpc(2*i+1, 3*N-2*i)
    x_2d = np.empty((1, N), dtype=object)
    for i in range(N):
      x_2d[0, i] = x_dps[i]
    _fft_naive(x_2d, 0, 0, 1, N)
    ref = np.array([x_2d[0, i] for i in range(N)], dtype=object)
    y = fft(x_dps)
    errors.append((dps, _err_inf(y, ref)))

  # Error must decrease monotonically
  for i in range(1, len(errors)):
    dps_i, err_i = errors[i]
    dps_p, err_p = errors[i-1]
    assert err_i < err_p, (
      'dps=%d err=%.1e not < dps=%d err=%.1e' % (dps_i, err_i, dps_p, err_p))


# ---------------------------------------------------------------------------
# Linearity
# ---------------------------------------------------------------------------

def test_linearity():
  mp.mp.dps = 50
  N = 16
  a = mp.mpc(2, 1)
  b = mp.mpc(-1, 3)
  x = _make_test_signal(N)
  y = np.array([mp.mpc(1, 0) for _ in range(N)], dtype=object)
  lhs = fft(a * x + b * y)
  rhs = a * fft(x) + b * fft(y)
  assert _err_inf(lhs, rhs) < 1e-48


# ---------------------------------------------------------------------------
# 2D, 3D, 4D axis dispatch
# ---------------------------------------------------------------------------

def test_nd_axis():
  mp.mp.dps = 50
  for shape, axes in [
    ((3, 4), [0, 1, -1, -2]),
    ((2, 3, 4), [0, 1, 2]),
    ((2, 3, 4, 5), [0, 1, 2, 3, -1, -2]),
  ]:
    x_np = np.arange(int(np.prod(shape)), dtype=complex).reshape(shape)
    x_obj = _to_mpc(x_np)
    for ax in axes:
      y_obj = fft(x_obj, axis=ax)
      y_np = np.fft.fft(x_np, axis=ax)
      err = _err_inf(y_obj, _to_mpc(y_np))
      assert err < 1e-12, (
        'shape=%s axis=%d err=%.1e' % (shape, ax, err))


def test_nd_roundtrip():
  mp.mp.dps = 50
  for shape, axes in [
    ((3, 4), [0, 1]),
    ((2, 3, 4), [0, 1, 2]),
    ((2, 3, 4, 5), [0, 1, 2, 3]),
  ]:
    x_np = np.arange(int(np.prod(shape)), dtype=complex).reshape(shape)
    x_obj = _to_mpc(x_np)
    for ax in axes:
      z_obj = ifft(fft(x_obj, axis=ax), axis=ax)
      err = _err_inf(z_obj, x_obj)
      assert err < 1e-48, (
        'shape=%s axis=%d roundtrip err=%.1e' % (shape, ax, err))


# ---------------------------------------------------------------------------
# Plan cache
# ---------------------------------------------------------------------------

def test_plan_cache_reuse():
  mp.mp.dps = 50
  from mpmath_fft import clear_plan_cache
  clear_plan_cache()
  p1 = build_plan(16)
  p2 = build_plan(16)
  assert p1 is p2


def test_plan_cache_dps_invalidation():
  mp.mp.dps = 50
  from mpmath_fft import clear_plan_cache
  clear_plan_cache()
  p1 = build_plan(8)
  mp.mp.dps = 100
  p2 = build_plan(8)
  assert p1 is not p2


def test_clear_plan_cache():
  mp.mp.dps = 50
  from mpmath_fft import clear_plan_cache, _plan_cache
  clear_plan_cache()
  build_plan(16)
  assert len(_plan_cache) == 1
  clear_plan_cache()
  assert len(_plan_cache) == 0


def test_plan_cache_eviction():
  """Building more than _MAX_PLAN_CACHE_SIZE plans triggers eviction."""
  mp.mp.dps = 50
  from mpmath_fft import clear_plan_cache, _plan_cache, _MAX_PLAN_CACHE_SIZE
  clear_plan_cache()
  for N in range(_MAX_PLAN_CACHE_SIZE + 20):
    build_plan(N + 2)
  assert len(_plan_cache) <= _MAX_PLAN_CACHE_SIZE


def test_plan_cache_eviction_preserves_recent():
  """Recently built plans survive; oldest plans are evicted."""
  mp.mp.dps = 50
  from mpmath_fft import clear_plan_cache, _plan_cache, _MAX_PLAN_CACHE_SIZE
  clear_plan_cache()
  # Build first plan (oldest)
  build_plan(2)
  # Fill cache to capacity
  for N in range(3, 3 + _MAX_PLAN_CACHE_SIZE - 1):
    build_plan(N)
  assert len(_plan_cache) == _MAX_PLAN_CACHE_SIZE
  # First plan should still be in cache (we filled exactly to limit)
  assert (2, 50) in _plan_cache
  # Add one more -- triggers eviction
  build_plan(9999)
  assert len(_plan_cache) == _MAX_PLAN_CACHE_SIZE
  # Oldest entry (N=2) should be evicted
  assert (2, 50) not in _plan_cache
  # Most recent entry should be present
  assert (9999, 50) in _plan_cache


def test_build_plan_rejects_zero():
    mp.mp.dps = 50
    with pytest.raises(ValueError, match='positive'):
        build_plan(0)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

def test_plan_cache_thread_safety():
  """Concurrent build_plan should not corrupt the cache."""
  mp.mp.dps = 50
  from mpmath_fft import clear_plan_cache, _plan_cache
  clear_plan_cache()

  errors = []

  def worker():
    try:
      for N in [16, 32, 64, 128]:
        p = build_plan(N)
        if p is None:
          errors.append('build_plan returned None')
    except Exception as e:
      errors.append(str(e))

  threads = [threading.Thread(target=worker) for _ in range(4)]
  for t in threads:
    t.start()
  for t in threads:
    t.join()

  assert len(errors) == 0, 'thread errors: ' + str(errors)
  assert len(_plan_cache) == 4
