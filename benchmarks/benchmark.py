"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Performance benchmarks for mpmath_fft vs numpy.fft.

Times forward/inverse FFT across algorithm-relevant N, multiple
precision tiers, and compares against numpy.fft baseline.

Usage:
  cd /path/to/mpmath_fft
  python benchmarks/benchmark.py
"""

import sys
import os
sys.path.insert(
  0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import mpmath as mp
import numpy as np
from mpmath_fft import fft, ifft, rfft, irfft, build_plan, clear_plan_cache

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# N sizes covering every algorithm path
BENCH_N = [
  # identity
  1,
  # unrolled small
  3, 4,
  # small-prime naive (N <= 31 and prime)
  5, 7, 11, 13, 17, 19, 23, 29, 31,
  # power-of-2 radix-2
  8, 16, 32, 64, 128, 256, 512, 1024,
  # Bluestein (prime >= 32)
  97,
  # composite naive (N < 32)
  6, 9, 15, 21, 30,
  # composite Cooley-Tukey (N >= 32)
  100,
]

# Precision tiers
BENCH_DPS = [15, 30, 50]

# Repetitions (more for tiny N to average out noise)
REPEATS_LARGE = 10      # N >= 64
REPEATS_SMALL = 50      # N < 64

# Seconds threshold for meaningful measurement
_MIN_MEASURE = 0.05


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _make_data(N, dtype, dps):
  """Build input array of N random mpc/mpf values at precision dps."""
  mp.mp.dps = dps
  out = np.empty(N, dtype=object)
  for i in range(N):
    re = mp.rand()
    im = mp.rand() if dtype is complex else mp.mpf(0)
    out[i] = mp.mpc(re, im) if dtype is complex else mp.mpf(re)
  return out


def _time_func(func, *args, repeats=10):
  """Time func(*args) with warm-up, return best-of-repeats in seconds."""
  # Warm-up
  for _ in range(3):
    func(*args)
  # Measure
  t0 = time.perf_counter()
  for _ in range(repeats):
    func(*args)
  dt = time.perf_counter() - t0
  return dt / repeats


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def bench_fft_vs_numpy(dps=15):
  """
  Time mpmath_fft vs numpy.fft across N at one precision.

  Returns list of (N, t_mp, t_np) tuples for forward FFT.
  """
  results = []
  header_printed = False

  for N in BENCH_N:
    repeats = REPEATS_LARGE if N >= 64 else REPEATS_SMALL

    # mpmath
    clear_plan_cache()
    mp.mp.dps = dps
    x_mp = _make_data(N, complex, dps)
    _ = build_plan(N)          # force plan build before timing
    t_mp = _time_func(fft, x_mp, repeats=repeats)

    # numpy
    x_np = np.array(
        [complex(float(v.real), float(v.imag)) for v in x_mp],
        dtype=complex)
    t_np = _time_func(np.fft.fft, x_np, repeats=max(repeats, 100))

    results.append((N, t_mp, t_np))

    if not header_printed:
      print()
      print(f"--- Forward FFT, dps={dps} ---")
      print(f"{'N':>5}  {'t_mp':>10}  {'t_np':>10}  {'ratio':>8}  path")
      header_printed = True

    ratio = t_mp / t_np if t_np > 0 else float('inf')
    path = _algorithm_path(N)
    print(f"{N:>5}  {t_mp:>10.3e}  {t_np:>10.3e}  {ratio:>8.1f}x  {path}")

  return results


def bench_ifft(dps=15):
  """Time inverse FFT vs numpy at one precision."""
  header_printed = False

  for N in BENCH_N:
    if N == 1:
      continue  # ifft is same as fft for N=1
    repeats = REPEATS_LARGE if N >= 64 else REPEATS_SMALL

    clear_plan_cache()
    mp.mp.dps = dps
    x_mp = _make_data(N, complex, dps)
    _ = build_plan(N)
    t_mp = _time_func(ifft, x_mp, repeats=repeats)

    x_np = np.array(
        [complex(float(v.real), float(v.imag)) for v in x_mp],
        dtype=complex)
    t_np = _time_func(np.fft.ifft, x_np, repeats=max(repeats, 100))

    if not header_printed:
      print()
      print(f"--- Inverse FFT, dps={dps} ---")
      print(f"{'N':>5}  {'t_mp':>10}  {'t_np':>10}  {'ratio':>8}x  path")
      header_printed = True

    ratio = t_mp / t_np if t_np > 0 else float('inf')
    path = _algorithm_path(N)
    print(f"{N:>5}  {t_mp:>10.3e}  {t_np:>10.3e}  {ratio:>8.1f}x  {path}")


def bench_real_transforms(dps=15):
  """Time rfft/irfft round-trip vs numpy at one precision."""
  header_printed = False

  for N in BENCH_N:
    if N == 1:
      continue
    repeats = REPEATS_LARGE if N >= 64 else REPEATS_SMALL

    # rfft -> irfft round-trip
    clear_plan_cache()
    mp.mp.dps = dps
    x_mp = _make_data(N, float, dps)   # real input

    def rtt(x):
      return irfft(rfft(x), n=N)

    t_mp = _time_func(rtt, x_mp, repeats=repeats)

    # numpy round-trip
    x_np = np.array([float(v) for v in x_mp])
    t_np = _time_func(
        lambda x: np.fft.irfft(np.fft.rfft(x), n=N),
        x_np, repeats=max(repeats, 100))

    if not header_printed:
      print()
      print(f"--- Real FFT round-trip, dps={dps} ---")
      print(f"{'N':>5}  {'t_mp':>10}  {'t_np':>10}  {'ratio':>8}x  path")
      header_printed = True

    ratio = t_mp / t_np if t_np > 0 else float('inf')
    path = _algorithm_path(N)
    print(f"{N:>5}  {t_mp:>10.3e}  {t_np:>10.3e}  {ratio:>8.1f}x  {path}")


def bench_precision_scaling(N=256):
  """Time FFT at one N across precision tiers to measure dps overhead."""
  print()
  print(f"--- Precision scaling, N={N} ---")
  print(f"{'dps':>5}  {'t_fft':>10}  {'t_ifft':>10}")

  for dps in BENCH_DPS:
    clear_plan_cache()
    mp.mp.dps = dps
    x_mp = _make_data(N, complex, dps)
    repeats = REPEATS_LARGE

    t_fwd = _time_func(fft, x_mp, repeats=repeats)
    t_inv = _time_func(ifft, x_mp, repeats=repeats)

    print(f"{dps:>5}  {t_fwd:>10.3e}  {t_inv:>10.3e}")


# ---------------------------------------------------------------------------
# Algorithm path identification
# ---------------------------------------------------------------------------

def _algorithm_path(N):
  """Return string naming the algorithm path for transform size N."""
  if N == 1:
    return "ident"
  if N == 3:
    return "N=3"
  if N == 4:
    return "N=4"
  # Check power of 2
  if (N & (N - 1)) == 0 and N > 0:
    return "radix-2"
  # Check prime
  if _is_prime_small(N):
    if N <= 31:
      return "naive"
    return "Bluestein"
  # Composite, non-power-of-2
  if N < 32:
    return "naive"
  return "Cooley-Tukey"


def _is_prime_small(n):
  """Quick primality check for benchmark labelling only."""
  if n < 2:
    return False
  if n % 2 == 0:
    return n == 2
  d = 3
  while d * d <= n:
    if n % d == 0:
      return False
    d += 2
  return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
  print("=" * 72)
  print("mpmath_fft benchmarks")
  print(f"Python {sys.version.split()[0]}, numpy {np.__version__}, "
        f"mpmath {mp.__version__}")
  print(f"Timer resolution: {time.get_clock_info('perf_counter').resolution} s")
  print("=" * 72)

  for dps in BENCH_DPS:
    bench_fft_vs_numpy(dps)

  bench_ifft(dps=15)

  bench_real_transforms(dps=15)

  bench_precision_scaling(256)

  print()
  print("Done.")


if __name__ == "__main__":
  main()
