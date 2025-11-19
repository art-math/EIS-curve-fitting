import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

### Using new 3-stage process // reduced model process
### And proportional weights for residuals

# ----------------------------
# Load data and known points
# ----------------------------
df = pd.read_excel('EIS data/EIS26.xlsx', 'Sheet1', skiprows=1)

## Note: make sure the calculus and bode scripts load the same data sheet

freqs = df.iloc[:, 1].values  # frequency in Hz
measured_real = df.iloc[:, 3].values  # real part (ohm)
measured_imag = -df.iloc[:, 4].values  # imag part (ohm)

omega = 2 * np.pi * freqs
Z_exp = measured_real + 1j * measured_imag

# ----------------------------
# Define proportional weights
# ----------------------------
w_real = 1.0 / (measured_real**2 + 1e-12)   # avoid divide by zero
w_imag = 1.0 / (measured_imag**2 + 1e-12)

# ----------------------------
# Curve features via calculus script
# Extrema / Inflection / Warburg
# ----------------------------

from calculus101925 import main as calculus_main

features = calculus_main()

# Known local extrema, format [Re_target, -Im_target]
localmax = features[0]
localmin = features[1]
warburgpoint = features[2]
inflectionslope = features[3]
inflection = features[4]
measured_wslope = features[5]

# Define points to split regions by Re(Z)
x1 = np.min(measured_real)          # smallest Re (largest freq)
x2 = float(localmax[0])             # Re at local max
x3 = float(localmin[0])             # Re at local min
x4 = np.max(measured_real)          # largest Re (smallest freq)

# High-frequency window mask: x1 <= Re(Z) <= x2
mask_hf = (measured_real >= x1) & (measured_real <= x2)

# ----------------------------
# Phi from Warburg slope
# ----------------------------
calculated_phi = (2/np.pi) * np.arctan(measured_wslope)
print("Calculated phi =", calculated_phi)

# ----------------------------
# Model functions
# ----------------------------
def Z_CPE(C, omega):
    """CPE with n=1 i.e. ideal capacitor."""
    return 1.0 / (C * (1j * omega))

def Z_Warburg_open(sigma, B, phi, omega):
    """Finite-length Warburg (open)."""
    w_sqrt = np.sqrt(1j * omega)
    arg = (B * w_sqrt) ** phi
    return (sigma * B / arg) * (1.0 / np.tanh(arg))

def Z_total(Ru, Rct, C, sigma, B, phi, omega):
    """Randles: Ru + parallel(CPE, Rct + Warburg)."""
    Zcpe = Z_CPE(C, omega)
    Zt = Z_Warburg_open(sigma, B, phi, omega)
    Zpar = 1.0 / (1.0 / Zcpe + 1.0 / (Rct + Zt))
    return Ru + Zpar

def Z_reduced_HF(Ru, Rct, C, omega):
    """Reduced model for HF stage: Ru + (C in || with Rct)."""
    Zcpe = Z_CPE(C, omega)
    Zpar = 1.0 / (1.0 / Zcpe + 1.0 / Rct)
    return Ru + Zpar

# ----------------------------
# Residuals
# ----------------------------
def stack_residuals(Z_model, Z_meas):
    res_real = np.real(Z_model) - np.real(Z_meas)
    res_imag = np.imag(Z_model) - np.imag(Z_meas)
    return np.concatenate([res_real, res_imag])

def residuals_full(params, omega, Z_meas):
    Ru, Rct, C, sigma, B, phi = params
    Zm = Z_total(Ru, Rct, C, sigma, B, phi, omega)
    return stack_residuals(Zm, Z_meas)

def residuals_full_fix_RuRctC(params, Ru, Rct, C, omega, Z_meas):
    sigma, B, phi = params
    Zm = Z_total(Ru, Rct, C, sigma, B, phi, omega)
    return stack_residuals(Zm, Z_meas)

def residuals_reduced_HF(params, omega, Z_meas):
    Ru, Rct, C = params
    Zm = Z_reduced_HF(Ru, Rct, C, omega)
    return stack_residuals(Zm, Z_meas)

# ----------------------------
# Weighted residuals
# ----------------------------
def stack_weighted_residuals(Z_model, Z_meas, w_real, w_imag):
    """Return concatenated weighted residuals."""
    res_real = np.real(Z_model) - np.real(Z_meas)
    res_imag = np.imag(Z_model) - np.imag(Z_meas)
    return np.concatenate([np.sqrt(w_real) * res_real,
                           np.sqrt(w_imag) * res_imag])

def residuals_full_weighted(params, omega, Z_meas, w_real, w_imag):
    Ru, Rct, C, sigma, B, phi = params
    Zm = Z_total(Ru, Rct, C, sigma, B, phi, omega)
    return stack_weighted_residuals(Zm, Z_meas, w_real, w_imag)

# ----------------------------
# Initial guesses & bounds
# ----------------------------
from bode101925 import main as bode_fit_main

print("\nRunning Bode-phase fitting to get initial parameter guesses...")
bode_best_params = bode_fit_main()
Ru0, Rct0, C0, sigma0, B0, phi0 = bode_best_params
print("Bode best-fit parameters used as initial guess:")
print(f"Ru={Ru0:.6g}, Rct={Rct0:.6g}, C={C0:.6g}, sigma={sigma0:.6g}, B={B0:.6g}, phi={phi0:.6g}")

## Override the initial Bode Ru and phi value
## Can adjust these to expand window on phi value
Ru0 = 141
phi0   = 0.8*float(calculated_phi)
phi_epsilon = 0.009  # confine phi near phi0

## note: phi_epsilon below .0078 seems to yield a poor fit

x0_full = np.array([Ru0, Rct0, C0, sigma0, B0, phi0], dtype=float)

# Bounds for high freq region (hf) (Ru, Rct, C)
lb_hf = np.array([Ru0 - 60,     0.0,     1e-12], dtype=float)
ub_hf = np.array([Ru0 + 5,  1e4,       1e-1 ], dtype=float)

# Bounds for (sigma, B, phi) with Ru,C,Rct fixed
phi_lo = max(1e-6, phi0 - phi_epsilon)
phi_hi = min(1.0,  phi0 + phi_epsilon)
lb_w = np.array([0.0,  1e-12, phi_lo], dtype=float)
ub_w = np.array([1e7,  1e2,   phi_hi], dtype=float)

# Bounds for full 6-param fit
lb_full = np.array([0.0,   0.0,   1e-12,  0.0,   1e-12, phi_lo], dtype=float)
ub_full = np.array([1e4,   1e5,   1e-1,   1e8,   1e2,   phi_hi], dtype=float)

# ----------------------------
# Evaluate initial guess (for reporting)
# ----------------------------
def rmse(vec):
    return float(np.sqrt(np.mean(vec**2)))

def r2_concat(Z_true, Z_pred):
    y_true = np.concatenate([np.real(Z_true), np.imag(Z_true)])
    y_pred = np.concatenate([np.real(Z_pred), np.imag(Z_pred)])
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res/ss_tot

Z_init = Z_total(*x0_full, omega)
res0 = stack_residuals(Z_init, Z_exp)
print(f"Initial guess RMSE = {rmse(res0):.6g}")

# ----------------------------
# STAGE 1: High freq fit (Ru, Rct, C)
# ----------------------------
print("\n=== Stage 1: High-frequency reduced model fit (Ru, Rct, C) ===")
omega_hf = omega[mask_hf]
Z_exp_hf = Z_exp[mask_hf]

x0_hf = np.array([Ru0, Rct0, C0], dtype=float)

res_hf = least_squares(
    residuals_reduced_HF, x0_hf,
    args=(omega_hf, Z_exp_hf),
    method="trf",
    bounds=(lb_hf, ub_hf),
    xtol=1e-9, ftol=1e-9, gtol=1e-9, max_nfev=4000
)

Ru1, Rct1, C1 = res_hf.x
Z_hf_fit = Z_reduced_HF(Ru1, Rct1, C1, omega_hf)
print(f"Ru1  = {Ru1:.6g} Ω")
print(f"Rct1 = {Rct1:.6g} Ω")
print(f"C1   = {C1:.6g} F")
print(f"Stage 1 RMSE = {rmse(res_hf.fun):.6g}, R²(HF window) = {r2_concat(Z_exp_hf, Z_hf_fit):.6f}")

# ----------------------------
# STAGE 2: Full model on all data, fix Ru1,Rct1,C1; fit (sigma, B, phi)
# ----------------------------
print("\n=== Stage 2: Full model with Ru1,Rct1,C1 fixed; fit (sigma, B, phi) ===")
x0_w = np.array([sigma0, B0, phi0], dtype=float)

res_w = least_squares(
    residuals_full_fix_RuRctC, x0_w,
    args=(Ru1, Rct1, C1, omega, Z_exp),
    method="trf",
    bounds=(lb_w, ub_w),
    xtol=1e-9, ftol=1e-9, gtol=1e-9, max_nfev=6000
)

sigma1, B1, phi1 = res_w.x
Z_stage2 = Z_total(Ru1, Rct1, C1, sigma1, B1, phi1, omega)
print(f"sigma1 = {sigma1:.6g}")
print(f"B1     = {B1:.6g} s^0.5")
print(f"phi1   = {phi1:.6g}")
print(f"Stage 2 RMSE = {rmse(res_w.fun):.6g}, R²(full) = {r2_concat(Z_exp, Z_stage2):.6f}")

# ----------------------------
# STAGE 3: Full model on all data, all params free
# ----------------------------
print("\n=== Stage 3: Full model with all parameters free (weighted) ===")
x0_full_from_stages = np.array([Ru1, Rct1, C1, sigma1, B1, phi1], dtype=float)

res_full = least_squares(
    residuals_full_weighted, x0_full_from_stages,
    args=(omega, Z_exp, w_real, w_imag),
    method="trf",
    bounds=(lb_full, ub_full),
    xtol=1e-9, ftol=1e-9, gtol=1e-9, max_nfev=8000
)

Ru_fit, Rct_fit, C_fit, sigma_fit, B_fit, phi_fit = res_full.x
Z_fit = Z_total(Ru_fit, Rct_fit, C_fit, sigma_fit, B_fit, phi_fit, omega)

print("\nFitted parameters (Stage 3):")
print(f"Ru    = {Ru_fit:.6g} Ω")
print(f"Rct   = {Rct_fit:.6g} Ω")
print(f"C     = {C_fit:.6g} F")
print(f"sigma = {sigma_fit:.6g}")
print(f"B     = {B_fit:.6g} s^0.5")
print(f"phi   = {phi_fit:.6g}")

rmse3 = rmse(stack_residuals(Z_fit, Z_exp))  # unweighted residuals
r2_3 = r2_concat(Z_exp, Z_fit)
print(f"Stage 3 (weighted) RMSE = {rmse3:.6g}, R²(full) = {r2_3:.6f}")

# ----------------------------
# Compare initial vs final; keep best
# ----------------------------
res_init = stack_residuals(Z_init, Z_exp)
rmse_init = rmse(res_init)
if rmse_init < rmse3:
    print("\nKeeping initial guess (better than Stage 3).")
    best_params = x0_full
    Z_best = Z_init
    best_rmse = rmse_init
    best_r2 = r2_concat(Z_exp, Z_init)
else:
    print("\nKeeping Stage 3 result.")
    best_params = np.array([Ru_fit, Rct_fit, C_fit, sigma_fit, B_fit, phi_fit])
    Z_best = Z_fit
    best_rmse = rmse3
    best_r2 = r2_3

print("\nBest parameters:")
labels = ["Ru", "Rct", "C", "sigma", "B", "phi"]
for k, v in zip(labels, best_params):
    unit = "Ω" if k in ("Ru","Rct") else ("F" if k=="C" else ("s^0.5" if k=="B" else ""))
    print(f"{k:6s} = {v:.6g} {unit}")
print(f"Best RMSE = {best_rmse:.6g}, Best R² = {best_r2:.6f}")

# ----------------------------
# Plot Nyquist: measured vs initial vs best
# ----------------------------
plt.figure(figsize=(6,6))
plt.plot(measured_real, -measured_imag, 'o', label="Measured", alpha=0.6)
plt.plot(np.real(Z_init), -np.imag(Z_init), '-', label="Initial guess", lw=1.5)
plt.plot(np.real(Z_best), -np.imag(Z_best), '-', label=f"Best fit (R² = {best_r2:.4f})", lw=2.5)

plt.xlabel("Z' (Ω)")
plt.ylabel("-Z'' (Ω)")
plt.title("Nyquist Plot: Measured vs Fit")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()

"""
# ==================================
# Additional Quality-of-Fit Metrics: Chi-Squared Tests
# ==================================
print("\n=== Additional quality-of-fit metrics (Chi-squared) ===")

NOTE- sigma_real and sigma_imag are global std deviations.
for chi^2 you want to use pointwise std deviations sigma_i. for each data point,
across multiple collections of the same dataset with the same setup. 
so in order to get pointwise standard deviations sigma_i and chi^2 we would need
to collect the same data multiple times (say 5-6 times)

# Extract measured and fitted components
Z_real_meas = np.real(Z_exp)
Z_imag_meas = np.imag(Z_exp)
Z_real_fit  = np.real(Z_fit)
Z_imag_fit  = np.imag(Z_fit)

# Standard deviations of measured data (can replace by known σ if available)
sigma_real = np.std(Z_real_meas)
sigma_imag = np.std(Z_imag_meas)

# Chi-squared (sum over all frequencies)
chi_sq = np.sum(
    ((Z_real_meas - Z_real_fit) / sigma_real) ** 2 +
    ((Z_imag_meas - Z_imag_fit) / sigma_imag) ** 2
)

N = len(freqs)          # number of frequencies
m = 6                   # number of adjustable parameters in the model
chi_sq_norm = chi_sq / (2 * N - m)

print(f"Chi-squared = {chi_sq:.6g}")
print(f"Normalized Chi-squared = {chi_sq_norm:.6g}")
print(f"Degrees of freedom = {2*N - m}")
"""

# ===========================
# Plot of Relative Residuals (real & imaginary)
# ===========================

# Extract measured and fitted components
Z_real_meas = np.real(Z_exp)
Z_imag_meas = np.imag(Z_exp)
Z_real_fit  = np.real(Z_fit)
Z_imag_fit  = np.imag(Z_fit)

print("\nPlotting relative residuals vs log10(frequency)...")

# Relative residuals (dimensionless, expected to scatter around 0)
rel_res_real = (Z_real_meas - Z_real_fit) / Z_real_meas
rel_res_imag = (Z_imag_meas - Z_imag_fit) / Z_imag_meas

plt.figure(figsize=(7,5))
plt.axhline(0, color='k', lw=1, linestyle='--', alpha=0.7)
plt.scatter(np.log10(freqs), rel_res_real, color='blue', label="Real residuals", s=20, alpha=0.7)
plt.scatter(np.log10(freqs), rel_res_imag, color='red', label="Imag residuals", s=20, alpha=0.7)
plt.xlabel("log10(Frequency [Hz])")
plt.ylabel("Relative residuals")
plt.title("Residual distribution (real & imaginary)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

