import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

### Randles Circuit with Finite-Length (Open) Warburg
### Bode Plot (phase angle vs. frequency)

### Load the data

df = pd.read_excel('Fitting31.xlsx', 'Sheet2', skiprows=1)
measured_phase = df.iloc[:,12].values   # measured phase angles in degrees
freqs = df.iloc[:,5].values             # frequency in Hz
log_freqs = df.iloc[:,4].values         # frequency in log10(Hz)

### Model Functions For Impedance For Finite-length (open) Warburg

def ZT(omega, sigma, B, phi):

    """
    Finite-length (open) Warburg impedance:
      s = (B * sqrt(j*omega))**phi
      Z_T = sigma * B / s * coth(s)
    where coth(s) = 1/tanh(s).
    omega is scalar ndarray; function returns complex ndarray.
    """

    sqrt_jw = np.sqrt(1j * omega) # j = sqrt(-1)
    s = (B * sqrt_jw) ** phi
    coth_s = 1.0 / np.tanh(s)
    Zt = sigma * B / s * coth_s
    return Zt

def model_phase(params, freqs):

    """
    params: [Ru, Rct, C, sigma, B, phi]
    freqs: array of frequencies (Hz)
    returns: predicted phase in degrees (positive magnitude, matches measured convention)
    """

    Ru, Rct, C, sigma, B, phi = params
    omegas = 2.0 * np.pi * freqs # angular frequency

    # compute Zc, Zt, parallel, total
    Zc = 1.0 / (1j * omegas * C)
    Zt = ZT(omegas, sigma, B, phi)
    Zrw = Rct + Zt
    Zp = 1.0 / (1.0 / Zrw + 1.0 / Zc)   # in parallel
    Z_total = Ru + Zp

    # convert angle to degrees then take absolute value
    phi_deg = np.degrees(np.angle(Z_total))   # np.angle returns radians
    return np.abs(phi_deg)

### Residual function for least_squares
# LATER: can add weights to improve the fit
# For now just using standard residuals

def residuals(params, freqs, measured_phase):
    pred = model_phase(params, freqs)
    return pred - measured_phase

### Initial guess for circuit parameters

Ru0   =  150
Rct0  = 100
C0    =  1.000e-6
sigma0 =  52000        # Warburg coeff
B0     =  .004         # B diffusion coeff (s^{1/2})
phi0   =  .103         # Experimental parameter (unitless, between 0 and 1)

x0 = [Ru0, Rct0, C0, sigma0, B0, phi0]

### Bounds for circuit parameters, used in least_squares trf method
lower_bounds = [0.0,   0.0,   1e-12,  0.0,    1e-8,  0.0]
upper_bounds = [1e5,   1e6,   1e-2,   1e6,    1e6,  1.0]

### Run least squares trf method
result = least_squares(
    residuals,
    x0,
    args=(freqs, measured_phase),
    bounds=(lower_bounds, upper_bounds),
    method='trf',
    verbose=2,       # prints progress; set to 0 to silence
    ftol=1e-8,
    xtol=1e-8,
    gtol=1e-8,
    max_nfev=2000
)

### Print fitted parameters
Ru_fit, Rct_fit, C_fit, sigma_fit, B_fit, phi_fit = result.x

print("\nFitted parameters:")
print(f"Ru    = {Ru_fit:.6g} ohm")
print(f"Rct   = {Rct_fit:.6g} ohm")
print(f"C     = {C_fit:.6g} F")
print(f"sigma = {sigma_fit:.6g}")   # T1
print(f"B     = {B_fit:.6g} s^0.5") # T2
print(f"phi  = {phi_fit:.6g}")

### Quantifying goodness-of-fit
predicted_phase = model_phase(result.x, freqs)
resid = predicted_phase - measured_phase
rmse = np.sqrt(np.mean(resid**2))
ss_tot = np.sum((measured_phase - np.mean(measured_phase))**2)
ss_res = np.sum(resid**2)
r2 = 1 - ss_res/ss_tot
print(f"RMSE = {rmse:.6g} deg, R^2 = {r2:.6g}")

### Plot measured vs fitted
plt.figure(figsize=(7,4))
plt.plot(log_freqs, measured_phase, label="Measured phase", color="blue")
plt.plot(log_freqs, predicted_phase, label="Fitted phase", color="red", linestyle='--')
plt.xlabel("log10(Hz)")
plt.ylabel("Phase angle (deg)")
plt.title("Bode Plot: Measured vs Fitted Phase")
plt.legend()
plt.grid(True, ls=':')
plt.tight_layout()
plt.show()
