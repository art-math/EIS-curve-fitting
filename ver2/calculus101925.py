import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### CALCULATE LOCAL EXTREMA, INFLECTION POINT, WARBURG POINT
### CURVE FEATURES

def main():

    # -------- Load the data ----------
    df = pd.read_excel('EIS data/EIS26.xlsx', 'Sheet1', skiprows=1)

    freqs = np.flip(df.iloc[:, 1].values)
    logfreqs = np.flip(df.iloc[:, 4].values)
    measured_real = np.flip(df.iloc[:, 3].values)
    measured_imag = np.flip(df.iloc[:, 4].values)

    omega = 2 * np.pi * freqs

    # -------- Parameters ----------
    amp_threshold = 0.0001
    window = 3  # for oscillation detection
    denoise_window = 0  # we won't need extra points; parabola uses endpoints

    # -------- Step 1: Detect oscillatory points ----------
    signal = measured_imag
    oscillatory_mask = np.zeros_like(signal, dtype=bool)

    for i in range(1, len(signal) - 1):
        d1 = signal[i] - signal[i - 1]
        d2 = signal[i + 1] - signal[i]
        if d1 * d2 < 0:
            if max(abs(d1), abs(d2)) > amp_threshold * np.std(signal):
                oscillatory_mask[i] = True

    # Extend mask to neighbors
    oscillatory_mask = np.convolve(oscillatory_mask.astype(int), np.ones(window, dtype=int), mode='same') > 0

    # -------- Step 2: Denoise using endpoint-constrained parabola --------
    denoised_signal = signal.copy()
    noisy_indices = np.where(oscillatory_mask)[0]

    from itertools import groupby
    from operator import itemgetter

    for k, g in groupby(enumerate(noisy_indices), lambda ix: ix[0] - ix[1]):
        group = list(map(itemgetter(1), g))
        i_start = max(group[0] - 1, 0)  # include left endpoint
        i_end = min(group[-1] + 1, len(signal) - 1)  # include right endpoint

        # x and y values for parabola fit
        x_pts = measured_real[i_start:i_end + 1]
        y_pts = signal[i_start:i_end + 1]

        # Construct parabola through first (left) and last (right) points
        # Parabola: y = a*(x-x0)*(x-x1) + y0*(x1-x)/(x1-x0) + y1*(x-x0)/(x1-x0) ?
        # We'll use a simple approach: quadratic with left & right points fixed
        x0, x1 = x_pts[0], x_pts[-1]
        y0, y1 = y_pts[0], y_pts[-1]
        n = len(x_pts)
        xs = np.array(x_pts)

        # Choose vertex in middle or fit smooth parabola
        # Solve for a in y = a*(x-x0)*(x-x1) + linear interpolation
        linear_interp = y0 + (y1 - y0) * (xs - x0) / (x1 - x0)
        denom = (xs - x0) * (xs - x1)
        denom[denom == 0] = 1  # avoid division by zero at endpoints
        a = np.sum((y_pts - linear_interp) / denom) / n  # average correction
        y_parabola = linear_interp + a * (xs - x0) * (xs - x1)

        # Replace only the noisy points, not the endpoints
        denoised_signal[group] = y_parabola[1:-1]

    # -------- Detect local maxima / minima on denoised data ----------
        y = denoised_signal
        dy = np.diff(y)
        signs = np.sign(dy)

        local_max_idx = []
        local_min_idx = []

        for i in range(1, len(signs)):
            # slope change positive -> negative ⇒ local max
            if signs[i - 1] > 0 and signs[i] < 0:
                local_max_idx.append(i)
            # slope change negative -> positive ⇒ local min
            elif signs[i - 1] < 0 and signs[i] > 0:
                local_min_idx.append(i)

        local_max_idx = np.array(local_max_idx)
        local_min_idx = np.array(local_min_idx)

    ### CALCULATE/DETECT INFLECTION POINT ON THE DENOISED DATA

    x = measured_real
    y = denoised_signal

    # first and second derivatives
    dy_dx = np.gradient(y, x)
    d2y_dx2 = np.gradient(dy_dx, x)

    inflection_idx = None

    if len(local_max_idx) > 0 and len(local_min_idx) > 0:
        # take the first max/min pair (assuming one of each)
        start = min(local_max_idx[0], local_min_idx[0])
        end = max(local_max_idx[0], local_min_idx[0])

        # find sign change in second derivative between them
        for i in range(start + 1, end):
            if d2y_dx2[i - 1] * d2y_dx2[i] < 0:
                inflection_idx = i
                break

    if inflection_idx is not None:
        x_infl = x[inflection_idx]
        y_infl = y[inflection_idx]
        slope_infl = dy_dx[inflection_idx]
    else:
        x_infl, y_infl, slope_infl = None, None, None

    ## Midpoint data point between local min and max-x region

    x = measured_real
    y = denoised_signal

    x_mid = y_mid = slope_mid = None

    if len(local_min_idx) > 0:
        min_idx = int(local_min_idx[0])
        x_min = x[min_idx]

        # 1. take all data points with x >= x_min
        mask = x >= x_min
        x_region = x[mask]
        y_region = y[mask]
        dy_dx_region = dy_dx[mask]

        if len(x_region) > 0:
            # 2. sort by x ascending
            sort_idx = np.argsort(x_region)
            x_region_sorted = x_region[sort_idx]
            y_region_sorted = y_region[sort_idx]
            dy_dx_region_sorted = dy_dx_region[sort_idx]

            # take the midpoint index of this list
            mid_idx = len(x_region_sorted) // 2
            x_mid = x_region_sorted[mid_idx]
            y_mid = y_region_sorted[mid_idx]

            # 3. slope at this midpoint
            slope_mid = dy_dx_region_sorted[mid_idx]
    else:
        print("No local min found; cannot compute region midpoint.")

    # -------------------------
    # Linear regression on Warburg region
    # -------------------------
    feature_warburgslope = None

    if len(local_min_idx) > 0:
        min_idx = int(local_min_idx[0])
        x_min = measured_real[min_idx]
        x_max = np.max(measured_real)

        # Select data points with x between p1 (=x_min) and p2 (=x_max)
        mask_warburg = (measured_real > x_min) & (measured_real < x_max)
        x_warburg = measured_real[mask_warburg]
        y_warburg = denoised_signal[mask_warburg]

        # Sort by x (just in case data are not ordered)
        sort_idx = np.argsort(x_warburg)
        x_warburg = x_warburg[sort_idx]
        y_warburg = y_warburg[sort_idx]

        # Skip the first 3 points beyond the local min
        if len(x_warburg) > 3:
            x_warburg = x_warburg[3:]
            y_warburg = y_warburg[3:]

            # Perform linear regression
            slope, intercept = np.polyfit(x_warburg, y_warburg, 1)
            feature_warburgslope = slope

            print(f"Linear regression on Warburg region:")
            print(f"  Range: Re > {x_min:.6f} to Re < {x_max:.6f}")
            print(f"  Slope (feature_warburgslope): {feature_warburgslope:.6f}")
        else:
            print("Not enough points after local min to compute Warburg slope.")
    else:
        print("No local min found; cannot compute Warburg slope.")


    # -------- Step 3: Plot ----------
    plt.figure(figsize=(6, 6))
    plt.scatter(measured_real[~oscillatory_mask], signal[~oscillatory_mask],
                color='blue', label="Clean Data", alpha=0.7)
    plt.scatter(measured_real[oscillatory_mask], signal[oscillatory_mask],
                color='red', label="Original Noisy Data", alpha=0.7)
    plt.scatter(measured_real[oscillatory_mask], denoised_signal[oscillatory_mask],
                color='green', label="Denoised Data", alpha=0.7)

    plt.xlabel("Z' (ohm)")
    plt.ylabel("-Z'' (ohm)")
    plt.title("Endpoint-Constrained Parabolic Denoising")
    plt.legend()
    plt.grid(True)
    plt.show()

    ## PLOT LOCAL MAX / LOCAL MIN / INFLECTION POINT FROM DENOISED DATA

    plt.figure(figsize=(8,6))

    # plot all points (denoised)
    plt.plot(measured_real, denoised_signal, 'o', color='blue', label="Denoised Data", alpha=0.7)

    # plot local maxima
    plt.plot(measured_real[local_max_idx], denoised_signal[local_max_idx],
             'o', color='red', markersize=10, label="Local Max")

    # plot local minima
    plt.plot(measured_real[local_min_idx], denoised_signal[local_min_idx],
             'o', color='green', markersize=10, label="Local Min")

    # plot inflection point if found
    if inflection_idx is not None:
        plt.plot(x_infl, y_infl, marker = 'o', color='purple', markersize=10, label="Inflection Point")

    # plot midpoint if found
    if x_mid is not None:
        plt.plot(x_mid, y_mid, 's', color='orange', markersize=10, label="Region Midpoint")

    plt.xlabel("Z' (ohm)")
    plt.ylabel("-Z'' (ohm)")
    plt.title("Denoised Nyquist Plot with Local Extrema")
    plt.legend()
    plt.grid(True)
    plt.show()

    print('Local max:', 'Re=', measured_real[local_max_idx], 'Im=', measured_imag[local_max_idx])
    print('Local min:', 'Re=', measured_real[local_min_idx], 'Im=', measured_imag[local_min_idx])

    if inflection_idx is not None:
        print(f"Inflection point: Re={x_infl:.6f}, Im={y_infl:.6f}")
        print(f"Slope at inflection: {slope_infl:.6f}")
    else:
        print("No inflection point detected between extrema.")

    if x_mid is not None:
        print(f"Region midpoint (x>=local min): Re={x_mid:.6f}, Im={y_mid:.6f}")
        print(f"Slope at region midpoint: {slope_mid:.6f}")

    feature_localmax = [measured_real[local_max_idx],measured_imag[local_max_idx]]
    feature_localmin = [measured_real[local_min_idx],measured_imag[local_min_idx]]
    feature_warburgpoint = [x_mid,y_mid]
    feature_inflectionpoint = [x_infl, y_infl]
    feature_inflectionslope = [slope_infl]

    features = [feature_localmax,feature_localmin,
                feature_warburgpoint, feature_inflectionslope,
                feature_inflectionpoint, feature_warburgslope]

    return features

if __name__ == "__main__":
    features = main()
    print("\n Calculus script finished. Curve features:")
    print(features)