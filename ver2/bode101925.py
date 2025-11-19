import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

### ======================================
###  Bode plot (phase angle vs. frequency)
###  Randles Circuit with Finite-Length (Open) Warburg
###  Multi-stage curve-fitting over subintervals
### =========================================

def main():

    # -------------------------------------------------
    # Load data
    # -------------------------------------------------
    df = pd.read_excel('EIS data/EIS26.xlsx', 'Sheet1', skiprows=1)
    #measured_phase = df.iloc[:, 12].values   # measured phase angles in degrees
    #freqs = df.iloc[:, 5].values             # frequency in Hz
    #log_freqs = df.iloc[:, 4].values         # frequency in log10(Hz)

    measured_phase = df.iloc[:,5].values
    freqs = df.iloc[:,1].values
    log_freqs = np.log10(freqs)

    # Sort data by increasing log10 frequency
    sort_idx = np.argsort(log_freqs)
    log_freqs = log_freqs[sort_idx]
    freqs = freqs[sort_idx]
    measured_phase = measured_phase[sort_idx]

    # Identify endpoints
    d1 = (log_freqs[0], measured_phase[0])
    d2 = (log_freqs[-1], measured_phase[-1])

    print(f"Smallest frequency point: {d1}")
    print(f"Largest frequency point: {d2}")

    # -------------------------------------------------
    # Fixed phi from measured slope
    # -------------------------------------------------
    from calculus101925 import main as calculus_main

    features = calculus_main()

    measured_wslope = features[5]
    calculated_phi = (2 / np.pi) * np.arctan(measured_wslope)
    phi_epsilon = 0.03  # window for fitting bounds
    print("Calculated phi =", calculated_phi)

    # -------------------------------------------------
    # Define model functions
    # -------------------------------------------------
    def ZT(omega, sigma, B, phi):
        """Finite-length (open) Warburg impedance."""
        sqrt_jw = np.sqrt(1j * omega)
        s = (B * sqrt_jw) ** phi
        coth_s = 1.0 / np.tanh(s)
        return sigma * B / s * coth_s

    def model_phase(params, freqs):
        """
        params: [Ru, Rct, C, sigma, B, phi]
        freqs:  array of frequencies (Hz)
        returns: predicted phase (deg)
        """
        Ru, Rct, C, sigma, B, phi = params
        omegas = 2.0 * np.pi * freqs
        Zc = 1.0 / (1j * omegas * C)
        Zt = ZT(omegas, sigma, B, phi)
        Zrw = Rct + Zt
        Zp = 1.0 / (1.0 / Zrw + 1.0 / Zc)
        Z_total = Ru + Zp
        phi_deg = np.degrees(np.angle(Z_total))
        return np.abs(phi_deg)

    def residuals(params, freqs, measured_phase):
        pred = model_phase(params, freqs)
        return pred - measured_phase

    # -------------------------------------------------
    # Default parameter bounds
    # -------------------------------------------------
    lower_bounds = [0.0, 0.0, 1e-12, 0.0, 1e-12, calculated_phi - phi_epsilon]
    upper_bounds = [1000, 1e6, 100.0, 1e6, 1e6, calculated_phi + phi_epsilon]


    # -------------------------------------------------
    # Least-squares fit
    # -------------------------------------------------
    def run_fit(x0, freqs, measured_phase):
        """Run least squares given initial guess x0."""
        result = least_squares(
            residuals,
            x0,
            args=(freqs, measured_phase),
            bounds=(lower_bounds, upper_bounds),
            method='trf',
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=2000
        )
        # compute RMSE
        pred = model_phase(result.x, freqs)
        resid = pred - measured_phase
        rmse = np.sqrt(np.mean(resid**2))
        return result.x, rmse

    # -------------------------------------------------
    # Multi-stage fitting procedure
    # -------------------------------------------------
    N = 4  # number of subintervals (can be tuned)
    x_splits = np.linspace(log_freqs.min(), log_freqs.max(), N + 1)

    # Initial guess list
    initial_guesses = []

    # Define a base initial guess manually
    Ru0   = 150
    Rct0  = 100
    C0    = 1.0e-6
    sigma0 = 52000
    B0     = 0.004
    phi0   = calculated_phi
    x0 = [Ru0, Rct0, C0, sigma0, B0, phi0]
    initial_guesses.append(x0)

    print("\n--- Stage 0 ---")
    print("Initial manual guess added:")
    print(x0)

    # Stage-by-stage fitting
    for k in range(1, N + 1):
        # Restrict data to first k subintervals
        mask = (log_freqs >= x_splits[0]) & (log_freqs <= x_splits[k])
        freqs_sub = freqs[mask]
        phase_sub = measured_phase[mask]

        print(f"\n==============================")
        print(f"Stage {k}: fitting over first {k} subinterval(s)")
        print(f"Data points used: {len(freqs_sub)}")

        new_guesses = []
        for i, guess in enumerate(initial_guesses):
            params_fit, rmse = run_fit(guess, freqs_sub, phase_sub)
            new_guesses.append(params_fit)
            print(f"  Guess {i+1} → RMSE = {rmse:.5f}")
            print(f"  Params = {[round(p,6) for p in params_fit]}")
        # Add all new guesses to list for next round
        initial_guesses.extend(new_guesses)
        print(f"Total guesses now stored: {len(initial_guesses)}")

    # -------------------------------------------------
    # Final full-data fit and selection
    # -------------------------------------------------
    print("\n==============================")
    print("Final fitting over entire dataset")
    final_results = []
    for i, guess in enumerate(initial_guesses):
        params_fit, rmse = run_fit(guess, freqs, measured_phase)
        final_results.append((params_fit, rmse))
        print(f"Final run {i+1} → RMSE = {rmse:.5f}")

    # Select best fit (lowest RMSE)
    best_params, best_rmse = min(final_results, key=lambda x: x[1])
    print("\nBest overall fit parameters:")
    labels = ["Ru", "Rct", "C", "sigma", "B", "phi"]
    for name, val in zip(labels, best_params):
        print(f"{name:6s} = {val:.6g}")
    print(f"Best RMSE = {best_rmse:.6f}")

    # Compute R² using best fit
    predicted_phase = model_phase(best_params, freqs)
    resid = predicted_phase - measured_phase
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((measured_phase - np.mean(measured_phase))**2)
    r2 = 1 - ss_res / ss_tot
    print(f"R² = {r2:.6f}")

    # -------------------------------------------------
    # Plot best-fit result
    # -------------------------------------------------
    predicted_phase = model_phase(best_params, freqs)
    plt.figure(figsize=(7,4))
    plt.plot(log_freqs, measured_phase, label="Measured phase", color="blue")
    plt.plot(log_freqs, predicted_phase, label="Best fit", color="red", linestyle='--')
    plt.xlabel("log10(Hz)")
    plt.ylabel("Phase angle (deg)")
    plt.title("Bode Plot: Measured vs Best Fit (Multi-stage)")
    plt.legend()
    plt.grid(True, ls=':')
    plt.tight_layout()
    plt.show()

    return best_params  # this will be the 6 optimal parameters

if __name__ == "__main__":
    params = main()
    print("\nBode script finished. Optimal parameters:")
    print(params)