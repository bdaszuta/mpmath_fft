"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Demonstration of mpmath_fft at multiple precisions.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mpmath as mp
import numpy as np
from mpmath_fft import fft, ifft

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _to_obj(x_np):
  """numpy complex -> mpmath object array."""
  out = np.empty(x_np.shape, dtype=object)
  for idx in range(x_np.size):
    v = x_np.flat[idx]
    out.flat[idx] = mp.mpc(float(v.real), float(v.imag))
  return out


# ---------------------------------------------------------------------------
# 1. Verify identity at double precision (dps=15)
# ---------------------------------------------------------------------------
print("=" * 60)
print("Arbitrary-precision FFT demo")
print("=" * 60)

N = 16
mp.mp.dps = 15

x_np = np.arange(N, dtype=complex)
x_obj = _to_obj(x_np)
y_obj = fft(x_obj)
z_obj = ifft(y_obj)

err = float(max(abs(z_obj[i] - x_obj[i]) for i in range(N)))
print()
print("--- Identity check (dps=15) ---")
print("N           = %d" % N)
print("fft->ifft error = %.1e" % err)
print()

# ---------------------------------------------------------------------------
# 2. Show precision convergence: error vs np.fft as dps increases
# ---------------------------------------------------------------------------
print("--- Precision convergence ---")
print("  dps      |error|_inf vs numpy.fft")
print("  -----    ------------------------")
y_np = np.fft.fft(x_np)
for dps in [5, 10, 15, 20, 30, 50, 100]:
  mp.mp.dps = dps
  x_obj_dps = _to_obj(x_np)
  y_obj_dps = fft(x_obj_dps)
  err = float(max(abs(complex(y_obj_dps[i]) - y_np[i]) for i in range(N)))
  print("  %4d     %.1e" % (dps, err))
print()

# ---------------------------------------------------------------------------
# 3. 2D transform with negative axis
# ---------------------------------------------------------------------------
print("--- 2D FFT (axis=-1) ---")
mp.mp.dps = 15
x2d_np = np.arange(12, dtype=complex).reshape(3, 4)
x2d_obj = _to_obj(x2d_np)
y2d_obj = fft(x2d_obj, axis=-1)
y2d_np = np.fft.fft(x2d_np, axis=-1)
err_2d = float(max(abs(complex(y2d_obj.flat[i]) - y2d_np.flat[i])
                   for i in range(12)))
print("3x4 array, axis=-1 error = %.1e" % err_2d)
print()

# ---------------------------------------------------------------------------
# 4. Prime-size transform (Bluestein path)
# ---------------------------------------------------------------------------
print("--- Prime-size (Bluestein) ---")
for Np in [7, 11, 13, 17, 19]:
  x_np_p = np.arange(Np, dtype=complex)
  x_obj_p = _to_obj(x_np_p)
  y_obj_p = fft(x_obj_p)
  y_np_p = np.fft.fft(x_np_p)
  err_p = float(max(abs(complex(y_obj_p[i]) - y_np_p[i]) for i in range(Np)))
  print("  N=%d  err=%.1e" % (Np, err_p))
print()

# ---------------------------------------------------------------------------
# 5. High-precision example: 50-digit FFT
# ---------------------------------------------------------------------------
print("--- High-precision example (dps=50) ---")
mp.mp.dps = 50
N = 8
x_hp = _to_obj(np.arange(N, dtype=complex))
y_hp = fft(x_hp)
z_hp = ifft(y_hp)
err_hp = float(max(abs(z_hp[i] - x_hp[i]) for i in range(N)))
print("N=8, dps=50, round-trip error = %.1e" % err_hp)
print()
print("First 3 output coeffs (50 digits):")
for i in range(3):
  print("  y[%d] = %s" % (i, mp.nstr(y_hp[i], 50)))

print()
print("Done.")
