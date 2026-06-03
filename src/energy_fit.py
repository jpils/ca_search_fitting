import numpy as np
from scipy.optimize import minimize_scalar

pressures = range(0, 160, 10)
results = []

for p in pressures:
    data = np.loadtxt(f"./mnt/p_{p}/search/energies.csv", delimiter=",", skiprows=1)

    x = data[:, 0]
    y = data[:, 1]

    # Fit 2nd-degree polynomial (returns a callable function)
    poly_func = np.polynomial.Polynomial.fit(x, y, deg=2)

    # Find the exact minimum within data bounds
    res = minimize_scalar(poly_func, bounds=(x.min(), x.max()), method="bounded")

    if res.success:
        results.append([p, res.x, res.fun])
        print(f"P={p:3d} kbar | c/a={res.x:.6f} | E={res.fun:.6f}")

# 2. Save results array to file
if results:
    results_arr = np.array(results)
    header = f"{'Pressure_kbar':<14} {'Equilibrium_c_a':<18} {'Fitted_Energy_eV':<18}"

    np.savetxt(
        "./mnt/best_ca.csv",
        results_arr,
        fmt=["%14d", "%18.6f", "%18.6f"],
        header=header,
        comments="",
    )
