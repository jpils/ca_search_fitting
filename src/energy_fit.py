import numpy as np
from scipy.optimize import minimize_scalar

# 1. Directly define the pressures
pressures = range(0, 160, 10)
results = []

for p in pressures:
    try:
        # Construct path directly and open with numpy
        # Change ".p_" to "p_" if your root folders don't have the leading dot
        data = np.loadtxt(f".p_{p}/search/p_{p}.csv", delimiter=",")

        x = data[:, 0]
        y = data[:, 1]

        # Fit 2nd-degree polynomial (returns a callable function)
        poly_func = np.polynomial.Polynomial.fit(x, y, deg=2)

        # Find the exact minimum within data bounds
        res = minimize_scalar(poly_func, bounds=(x.min(), x.max()), method="bounded")

        if res.success:
            results.append([p, res.x, res.fun])
            print(f"P={p:3d} GPa | c/a={res.x:.6f} | E={res.fun:.6f}")

    except IOError:
        # Silently skip missing folders/files
        continue

# 2. Save results array to file
if results:
    results_arr = np.array(results)
    header = f"{'Pressure_GPa':<14} {'Equilibrium_c_a':<18} {'Fitted_Energy_eV':<18}"

    np.savetxt(
        "equilibrium_results.txt",
        results_arr,
        fmt=["%14d", "%18.6f", "%18.6f"],
        header=header,
        comments="",
    )
