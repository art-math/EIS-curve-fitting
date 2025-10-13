import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

### NYQUIST PLOT WARBURG OPEN
### LOCAL MAX AND LOCAL MIN PENALTY
### INFLECTION POINT PENALTY
### WARBURG LINE SLOPE PENALTY
### WARBURG POINT VALUE PENALTY

# --------- Load data ----------
df = pd.read_excel('Fitting31.xlsx', 'Sheet2', skiprows=1)
freqs = df.iloc[:, 5].values  # frequency in Hz
measured_real = df.iloc[:, 0].values  # real part (ohm)
measured_imag = -df.iloc[:, 1].values  # imag part (ohm)

omega = 2 * np.pi * freqs # angular frequency

### Fixed parameters

Ru = 151.4     ## x-coordinate of the left endpoint
               ## easy to read this one off directly from the graph

## Define the known local max and local min for the curve,
# which the algorithm should try to match (within a specified window that we can adjust)
# Format: [Re_target, -Im_target]
# localmin = [392.169, 52.2728] ## Fitting32
# localmax = [264.885, 100.407] ## Fitting32

localmin = [781.843, 96.178416] ## Fitting31
localmax = [410.87, 226.944] ## Fitting31

extrema_window = 8  # window in units of Re(Z) (x-axis) around target to apply penalty

# Penalty strengths (tune these: larger -> stricter enforcement)
lambda_deriv = 1e4   # weight for making dY/dX ~ 0 near extrema
lambda_value = 1e4   # weight for making the -Im value near supplied target (optional)

### Define the known inflection point and slope at the inflection point
# Format: [Re_target, -Im_target] because Nyquist plot often uses -Im(Z) on vertical axis.
# inflection = [341.871, 72.814576]  ## Fitting32
inflection = [653.569000, 139.446000] ## Fitting31
inflection_window = 8  # window in units of Re(Z) (x-axis) around target to apply penalty

#inflectionslope = -0.679989  ## Fitting32
inflectionslope = -0.527356  ## Fitting31
inflectionslope_window = 0.01  ## tolerance window around target slope (no penalty if within +- this)

# Penalty strength for inflection slope failing to meet target tolerance
lambda_inflection_slope = 1e3  # tune: larger => stricter enforcement

### Define the known point on the Warburg line and the slope at this line
#warburgpoint = [507.093000, 85.248704] # Fitting32
warburgpoint = [1007.295000, 124.191000] # Fitting31
warburgpoint_window = 8

#warburgslope = 0.467918 ## Fitting 32 // target slope at Warburg point
warburgslope = 0.152665 ## Fitting 31 // target slope at Warburg point
warburgslope_window = 0.01

# penalty strength for Warburg slope failing to meet target tolerance
lambda_warburg_slope = 1e3  # tune: larger => stricter enforcement

# penalty strength for Warburg point
lambda_warburg_value = 1e3  # tune: larger => stricter enforcement


### Model Functions for Impedance with Finite-Length (Open) Warburg

def Z_CPE(C, omega):
    """CPE with n=1 --> ideal capacitor."""
    return 1.0 / (C * (1j * omega))

def Z_Warburg_open(sigma, B, phi, omega):
    """Finite-length Warburg (open)."""
    w_sqrt = np.sqrt(1j * omega)
    arg = (B * w_sqrt) ** phi
    return (sigma * B / arg) * (1.0 / np.tanh(arg))

def Z_total(Rct, C, sigma, B, phi, omega):
    """Randles circuit: Ru + parallel( C , Rct + Warburg )."""
    Zcpe = Z_CPE(C, omega)
    Zt = Z_Warburg_open(sigma, B, phi, omega)
    Zpar = 1.0 / (1.0 / Zcpe + 1.0 / (Rct + Zt))
    return Ru + Zpar

#### Residuals with extrema + inflection slope penalties
def residuals(params, omega, Z_exp,
              extrema_list=[localmin, localmax],
              extrema_window=extrema_window,
              lambda_deriv=lambda_deriv,
              lambda_value=lambda_value):
    """
    params: [Rct, C, sigma, B, phi]
    Z_exp: complex experimental Z values
    """
    Rct, C, sigma, B, phi = params
    Z_model = Z_total(Rct, C, sigma, B, phi, omega)

    # Basic residuals (real and imag)
    res_real = np.real(Z_model) - np.real(Z_exp)
    res_imag = np.imag(Z_model) - np.imag(Z_exp)

    # Extrema & inflection calculations
    Re_model = np.real(Z_model)
    Im_model = np.imag(Z_model)
    minus_Im_model = -Im_model  # Nyquist vertical axis

    # numerical derivatives d(Re)/dω and d(-Im)/dω
    # Note: np.gradient handles non-uniform omega if we pass omega
    dRe_domega = np.gradient(Re_model, omega, edge_order=2)
    dY_domega = np.gradient(minus_Im_model, omega, edge_order=2)

    # safe division to compute dY/dX
    eps = 1e-12
    dRe_domega_safe = dRe_domega.copy()
    small_mask = np.abs(dRe_domega_safe) < eps
    if np.any(small_mask):
        dRe_domega_safe[small_mask] = eps * np.sign(dRe_domega_safe[small_mask] + eps)

    dY_dX = dY_domega / dRe_domega_safe

    # For each supplied extremum [Re_target, -Im_target] build smooth weight (Gaussian in Re)
    penalties = []
    sigma_x = extrema_window / 2.0  # gaussian width; adjust if you prefer hard window
    if sigma_x <= 0:
        sigma_x = 1e-6

    for extremum in extrema_list:
        Re_target, minus_Im_target = extremum

        # Gaussian weight based on closeness in Re(Z) to Re_target
        w = np.exp(-0.5 * ((Re_model - Re_target) / sigma_x) ** 2)

        # 1) derivative penalty: encourage dY/dX ~ 0 weighted by w
        deriv_pen = np.sqrt(lambda_deriv) * (np.sqrt(w) * dY_dX)
        penalties.append(deriv_pen)

        # 2) value penalty: encourage -Im(Z) near given value at that Re (weighted)
        value_pen = np.sqrt(lambda_value) * (np.sqrt(w) * (minus_Im_model - minus_Im_target))
        penalties.append(value_pen)

    # Inflection slope penalty
    # localize around inflection[0] (Re target) with Gaussian weight
    Re_inf_target, minus_Im_inf_target = inflection
    sigma_x_inf = inflection_window / 2.0
    if sigma_x_inf <= 0:
        sigma_x_inf = 1e-6

    w_inf = np.exp(-0.5 * ((Re_model - Re_inf_target) / sigma_x_inf) ** 2)

    # slope difference at each point (model dY/dX minus target slope)
    slope_diff = dY_dX - inflectionslope

    # We want no penalty if |slope_diff| <= inflectionslope_window
    # Implement residual:
    # penalty component = max(|slope_diff| - tol, 0) * sign(slope_diff)
    tol = inflectionslope_window
    hinge_component = np.clip(np.abs(slope_diff) - tol, 0.0, None) * np.sign(slope_diff)

    # scale and weight
    slope_pen = np.sqrt(lambda_inflection_slope) * (np.sqrt(w_inf) * hinge_component)
    penalties.append(slope_pen)

    # we could also encourage the -Im value at the inflection Re to be near the supplied -Im target;

    # Warburg line slope penalty
    Re_w_target, minus_Im_w_target = warburgpoint
    sigma_x_w = warburgpoint_window / 2.0
    if sigma_x_w <= 0:
        sigma_x_w = 1e-6

    # Gaussian weight in Re around Warburg target
    w_w = np.exp(-0.5 * ((Re_model - Re_w_target) / sigma_x_w) ** 2)

    # slope penalty
    slope_diff_w = dY_dX - warburgslope
    tol_w = warburgslope_window
    hinge_component_w = np.clip(np.abs(slope_diff_w) - tol_w, 0.0, None) * np.sign(slope_diff_w)
    slope_pen_w = np.sqrt(lambda_warburg_slope) * (np.sqrt(w_w) * hinge_component_w)
    penalties.append(slope_pen_w)

    # value penalty: encourage curve to pass near (Re_target, -Im_target)
    value_pen_w = np.sqrt(lambda_warburg_value) * (np.sqrt(w_w) * ((minus_Im_model - minus_Im_w_target)))
    penalties.append(value_pen_w)

    # Flatten penalty arrays into single 1D array
    if penalties:
        penalty_vec = np.concatenate([p.flatten() for p in penalties])
    else:
        penalty_vec = np.array([])

    # Combine original residuals and penalty residuals
    return np.concatenate([res_real, res_imag, penalty_vec])

### Initial guess for circuit parameters

### Fitting 32 initial guess
#Rct0  = 200.875
#C0    = 5.17933e-07
#sigma0 = 37656.5
#B0     = 1.81362e-06
#phi0   = 0.380961

### Fitting 31 initial guess
Rct0  = 112
C0    = 9.18204e-07
sigma0 = 43580.6
B0     = 0.00126493
phi0   = 0.166255

x0 = [Rct0, C0, sigma0, B0, phi0]

Z_init = Z_total(Rct0, C0, sigma0, B0, phi0, omega)

### Bounds for circuit parameters, used in least_squares trf method

lower_bounds = [0,    1e-12, 0,     1e-12, 0]
upper_bounds = [1000.0,  100.0,  1e6,   100.0, 1.0]

#### Run least squares
Z_exp = measured_real + 1j * measured_imag

#### Evaluate initial guess before optimization
res0 = residuals(x0, omega, Z_exp)
rmse0 = np.sqrt(np.mean(res0**2))

print(f"Initial guess RMSE = {rmse0:.6g}")

result = least_squares(
    residuals, x0,
    args=(omega, Z_exp),
    method="trf",
    bounds=(lower_bounds, upper_bounds),
    # extra options we can adjust
    xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=15000
)

Rct_fit, C_fit, sigma_fit, B_fit, phi_fit = result.x

#### Print fitted parameters
print("\nFitted parameters:")
print(f"Rct   = {Rct_fit:.6g} Ω")
print(f"C     = {C_fit:.6g} F")
print(f"sigma = {sigma_fit:.6g}")
print(f"B     = {B_fit:.6g} s^0.5")
print(f"phi   = {phi_fit:.6g}")

### Goodness of fit // RMSE
rmse_fit = np.sqrt(np.mean(result.fun**2))
print(f"Optimizer RMSE = {rmse_fit:.6g}")

### Choose best solution
if rmse0 < rmse_fit:
    print("Keeping initial guess (better fit than optimizer).")
    best_params = x0
    best_rmse = rmse0
else:
    print("Keeping optimizer result.")
    best_params = result.x
    best_rmse = rmse_fit

#### Compute R^2 value for fitted vs measured data
Z_best = Z_total(*best_params, omega)
# Flatten real and imaginary parts together to evaluate overall fit quality
y_true = np.concatenate([np.real(Z_exp), np.imag(Z_exp)])
y_pred = np.concatenate([np.real(Z_best), np.imag(Z_best)])
ss_res = np.sum((y_true - y_pred)**2)
ss_tot = np.sum((y_true - np.mean(y_true))**2)
r2 = 1 - ss_res / ss_tot
print(f"R² (fitted vs measured) = {r2:.6f}")

#### Plot measured vs fitted
Rct_fit, C_fit, sigma_fit, B_fit, phi_fit = best_params
Z_fit = Z_total(Rct_fit, C_fit, sigma_fit, B_fit, phi_fit, omega)

plt.figure(figsize=(6,6))
plt.plot(measured_real, -measured_imag, 'o', label="Measured", alpha=0.6)
plt.plot(np.real(Z_fit), -np.imag(Z_fit), '-', label="Fitted", lw=2)
plt.plot(np.real(Z_init), -np.imag(Z_init), '-', label="Initial guess")

# show target extrema
plt.scatter([localmin[0], localmax[0]], [localmin[1], localmax[1]], c=['green','red'], marker='x', s=80, label='Target extrema')
# show window circles for extrema
plt.gca().add_patch(plt.Circle((localmin[0], localmin[1]), extrema_window, color='green', fill=False, alpha=0.3))
plt.gca().add_patch(plt.Circle((localmax[0], localmax[1]), extrema_window, color='red', fill=False, alpha=0.3))

# show inflection marker and its window
plt.scatter([inflection[0]], [inflection[1]], c=['blue'], marker='x', s=100, label='Target inflection')
plt.gca().add_patch(plt.Circle((inflection[0], inflection[1]), inflection_window, color='blue', fill=False, alpha=0.3))

# show warburg point and its window
plt.scatter([warburgpoint[0]], [warburgpoint[1]], c=['yellow'], marker='x', s=100, label='Warburg point')
plt.gca().add_patch(plt.Circle((warburgpoint[0], warburgpoint[1]), warburgpoint_window, color='blue', fill=False, alpha=0.3))

plt.xlabel("Z' (Ω)")
plt.ylabel("-Z'' (Ω)")
plt.title("Nyquist Plot: Measured vs Fitted")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
