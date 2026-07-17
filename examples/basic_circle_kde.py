#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import optimize

from symdisc import (
    getExtendedFeatureMatrix,
    discover_symmetry_coeffs,
    generate_euclidean_killing_fields_with_names,
)

def main():
    rng = np.random.default_rng(0)

    # ---- Circle
    t = rng.uniform(0.0, 2*np.pi, size=2000)
    X = np.column_stack([np.cos(t), np.sin(t)])

    # Kernel Density Estimate of density
    Xt = X.transpose()
    kde = stats.gaussian_kde(Xt)

    # plot the estimated KDE
    xs = np.linspace(-3, 3, 400)
    Xs, Ys = np.meshgrid(xs, xs)
    grid = np.vstack([Xs.ravel(), Ys.ravel()])
    gridProbs = kde.pdf(grid)
    plt.scatter(grid[0], grid[1], c=gridProbs)
    plt.show(block=False)

    # estimate Jacobian of KDE
    Jg = np.array([optimize.approx_fprime(x, kde.pdf) for x in X])
    Jg = Jg.reshape((Jg.shape[0], 1, Jg.shape[1]))

    # Euclidean Killing fields in ambient R^3
    kvs, names = generate_euclidean_killing_fields_with_names(d=X.shape[1])

    # Build extended feature matrix A for invariance discovery
    A, info = getExtendedFeatureMatrix(X, Jg, kvs, normalize_rows=True)
    # A: shape (N*m, q). Here m=r (# constraints), q=#vector fields

    # SVD-based symmetry coefficients (columns)
    #C, svals = discover_symmetry_coeffs(A, rtol=1e-8)
    C, svals = discover_symmetry_coeffs(A, max_prop=0.2)
    print("Discovered coefficient vectors shape:", C.shape)
    print("Small singular values:", svals)
    print("Discovered vector fields: ")
    print([names[np.argmax(np.abs(vec))] for vec in C.transpose()])

    plt.show()  # keep plot open when running as script

if __name__ == "__main__":
    main()
