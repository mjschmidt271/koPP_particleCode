#!/usr/bin/env python3

import os.path
import numpy as np
from scipy.stats import norm

fname = "./data/particles.txt"
if os.path.isfile(fname):
    abc = 1
else:
    fname = "./data/particles1.txt"

f_ens = "./data/ens.txt"

with open(f_ens) as f:
    N_ens = int(f.readline())

with open(fname) as f:
    shapeData = f.readline()
    dim = int(f.readline())
    p = f.readline()

omega = np.zeros(2);

params = p.split()
IC_type_space = float(params[0])
IC_type_mass = float(params[1])
omega[0]= float(params[2])
omega[1]= float(params[3])
X0_space = float(params[4])
hat_pct = float(params[5])
X0_mass = float(params[6])
maxT = float(params[7])
dt = float(params[8])
D = float(params[9])
pctRW = float(params[10])
cdist_coeff = float(params[11])
cutdist = float(params[12])

# delta IC
sigma = 0.0

shapeData = shapeData.split()
shapeData = [int(i) for i in shapeData]
Np = shapeData[0]
Nsteps = shapeData[1] + 1

X = np.ndarray((Np, Nsteps))

for e in range(1, N_ens + 1):
    fname_ens = "./data/particles" + str(e) + ".txt"
    with open(fname_ens) as f:
        data = np.loadtxt(f, skiprows=3)
        tmpX = np.reshape(data[:, 0], (Np, Nsteps), 'f')
        if e == 1:
            X = tmpX
        else:
            X = np.concatenate((X, tmpX))
# sort spatially for comparison later
X.sort(axis=0)

L = omega[1] - omega[0]


def analytic1d(X, t, sigma, D, L):
    sol = (1 / np.sqrt(2 * np.pi * (sigma + 2 * D * t)))\
        * np.exp(-((0.5 * L - X[:])**2 / (2 * (sigma + 2 * D * t))))
    return sol


def analytic2d(dim, X, Y, t, sigma, D, L):
    sol =  (1 / np.power(2 * np.pi * (sigma + 2 * D * t), float(dim) / 2.0))\
           * np.exp(-(((0.5 * L - X)**2 + (0.5 * L - Y)**2)/ (2 * (sigma + 2 * D * t))));
    return sol


asoln = analytic1d(X[:, -1], maxT, sigma, D, L)

xplot = np.linspace(min(X[:, -1]), max(X[:, -1]), len(X[:, -1]))
mu, sigma = norm.fit(X[:, -1])
p = norm.pdf(xplot, mu, sigma)

print('max analytic                      = {:.4f}'.format(max(asoln)))
print('max fitted                        = {:.4f}'.format(max(p)))
print('fitted: mu, sigma                 = {:.4f}, {:.4f}'.format(mu, sigma))
print('true: mean, std. [sqrt(2 D maxT)] = {:.4f}, {:.4f}'.format(X0_space, np.sqrt(2.0 * D * maxT)))

mse = np.mean((asoln - p)**2)
error_max_val = abs(max(asoln) - max(p))
error_mean = abs(mu - X0_space)
error_std = abs(sigma - np.sqrt(2.0 * D * maxT))

print('MSE                 = {:.4f}'.format(mse))
print('error in max val    = {:.4f}'.format(error_max_val))
print('error in mean, std. = {:.4f}, {:.4f}'.format(error_mean, error_std))

mse_tol = 5.0e-2
assert mse <= mse_tol, '1D MSE error too high: error = {:.4f}. tol = {:.4f}.'.format(mse, mse_tol)
maxval_tol = 1.0e-2
assert error_max_val <= maxval_tol, '1D Max Value error too high: error = {:.4f}. tol = {:.4f}.'.format(error_max_val, maxval_tol)
mean_tol = 5.0e-2
assert error_mean <= mean_tol, '1D Mean error too high: error = {:.4f}. tol = {:.4f}.'.format(error_mean, mean_tol)
std_tol = 5.0e-2
assert error_std <= std_tol, '1D Std. Dev. error too high: error = {:.4f}. tol = {:.4f}.'.format(error_std, std_tol)
print('SUCCESS: {}-d RW passes with tolerances: mse_tol = {:.4f}, maxval_tol = {:.4f}, \
      mean_tol = {:.4f} std_tol = {:.4f}.'.format(dim, mse_tol, maxval_tol, mean_tol, std_tol))
