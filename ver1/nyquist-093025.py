import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

### Load the data

df = pd.read_excel('Fitting31.xlsx', 'Sheet2', skiprows=1)
freqs = df.iloc[:, 5].values  # frequency in Hz
measured_real = df.iloc[:, 0].values  # real part (ohm)
measured_imag = df.iloc[:, 1].values  # imag part (ohm)

omega = 2 * np.pi * freqs ## angular frequency

### Fixed parameters

Ru = 151.4     ## x-coordinate of the left endpoint
               ## easy to read this one off directly from the graph

### Model Functions for Impedance with Finite-Length (Open) Warburg

def Z_CPE(C, omega):
    """CPE with n=1 i.e. ideal capacitor."""
    return 1.0 / (C * (1j * omega))

def Z_Warburg_open(sigma, B, phi, omega):
    """Finite-length Warburg (open)."""
    w_sqrt = np.sqrt(1j * omega)
    arg = (B * w_sqrt) ** phi
    return (sigma * B / arg) * (1 / np.tanh(arg))

def Z_total(Rct, C, sigma, B, phi, omega):
    """Randles circuit: Ru + parallel( CPE , Rct + Warburg )."""
    Zcpe = Z_CPE(C, omega)
    Zt = Z_Warburg_open(sigma, B, phi, omega)
    Zpar = 1.0 / (1.0 / Zcpe + 1.0 / (Rct + Zt))
    return Ru + Zpar

# --------- Residuals ----------
def residuals(params, omega, Z_exp):
    Rct, C, sigma, B, phi = params
    Z_model = Z_total(Rct, C, sigma, B, phi, omega)

    # Stack real and imaginary parts to fit both

    res_real = np.real(Z_model) - np.real(Z_exp)
    res_imag = np.imag(Z_model) - np.imag(Z_exp)
    return np.concatenate([res_real, res_imag])

### Initial guess for circuit parameters
Rct0  = 102.6
C0    = 1e-6
sigma0 = 5.20e+4
B0     = 0.005
phi0   = 0.103

x0 = [Rct0, C0, sigma0, B0, phi0]

Z_init = Z_total(Rct0, C0, sigma0, B0, phi0,omega)

### Bounds for circuit parameters, used in least_squares trf method
lower_bounds = [0,    1e-12, 0,     1e-6, 0]
upper_bounds = [1e5,  1e-2,  1e6,   10.0, 1.0]

### Run least_squares trf method
# Later: can add weights to residuals to adapt the algorithm and improve the fit
# For now just using standard residuals

Z_exp = measured_real + 1j * measured_imag

result = least_squares(
    residuals, x0,
    args=(omega, Z_exp),
    method="trf",
    bounds=(lower_bounds, upper_bounds)
)

Rct_fit, C_fit, sigma_fit, B_fit, phi_fit = result.x

### Print fitted parameters

print("\nFitted parameters:")
# print(f"Ru    = {Ru_fit:.6g} Ω")
print(f"Rct   = {Rct_fit:.6g} Ω")
print(f"C     = {C_fit:.6g} F")
print(f"sigma = {sigma_fit:.6g}")
print(f"B     = {B_fit:.6g} s^0.5")
print(f"phi   = {phi_fit:.6g}")

### Quantify Goodness of Fit // RMSE

rmse = np.sqrt(np.mean(result.fun**2))
print(f"\nRMSE = {rmse:.6g} Ω")

### Plot measured vs. fitted Nyquist curve

Z_fit = Z_total(Rct_fit, C_fit, sigma_fit, B_fit, phi_fit, omega)

plt.figure(figsize=(6,6))
plt.plot(measured_real, measured_imag, 'o', label="Measured", alpha=0.6)
plt.plot(np.real(Z_fit), -np.imag(Z_fit), '-', label="Fitted", lw=2)
plt.plot(np.real(Z_init),-np.imag(Z_init),'-', label="Initial guess")
plt.xlabel("Z' (Ω)")
plt.ylabel("-Z'' (Ω)")
plt.title("Nyquist Plot: Measured vs Fitted")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
