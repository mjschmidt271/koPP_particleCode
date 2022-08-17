#!/usr/bin/env python3

# script to verify that MT unit test is passing

import numpy as np
# import matplotlib.pyplot as plt

fname = "./data/particles.txt"

omega = np.zeros(2);

with open(fname) as f:
    shapeData = f.readline()
    dim = int(f.readline())
    p = f.readline()

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
data = np.loadtxt(fname, skiprows=3)

# delta IC
sigma = 0.0

shapeData = shapeData.split()
shapeData = [int(i) for i in shapeData]
Np = shapeData[0]
Nsteps = shapeData[1] + 1

X = np.ndarray([dim, Np, Nsteps])

for i in range(dim):
    X[i, :, :] = np.reshape(data[:, i], (Np, Nsteps), 'f')
mass = np.reshape(data[:, dim], (Np, Nsteps), 'f')

L = omega[1] - omega[0]
def analytic1d(X, t, sigma, D, L):
    sol =  (1 / np.sqrt(2 * np.pi * (sigma + 2 * D * t)))\
        * np.exp(-((0.5 * L - X[:])**2 / (2 * (sigma + 2 * D * t))));
    return sol
def analytic2d(dim, X, Y, t, sigma, D, L):
    sol =  (1 / np.power(2 * np.pi * (sigma + 2 * D * t), float(dim) / 2.0))\
           * np.exp(-(((0.5 * L - X)**2 + (0.5 * L - Y)**2)/ (2 * (sigma + 2 * D * t))));
    return sol

if dim == 1:
    asoln = analytic1d(X[0, :, -1], maxT, sigma, D, L)
    asoln = asoln / sum(asoln)
    error = np.linalg.norm(asoln - mass[:, -1])
    mse = np.square(np.subtract(asoln, mass[:, -1])).mean()
    rmse = np.sqrt(mse)
elif dim == 2:
    asoln = analytic2d(dim, X[0, :, -1], X[1, :, -1], maxT, sigma, D, L)
    asoln = asoln / sum(asoln)
    error = np.linalg.norm(asoln - mass[:, -1])
    mse = np.square(np.subtract(asoln, mass[:, -1])).mean()
    rmse = np.sqrt(mse)
# print('error = ', error, 'rmse = ', rmse)

if dim == 1:
    tol = 1.0e-14
elif dim == 2:
    tol = 1.0e-3

assert error <= tol, '{}-d MT error too high: error = {}'.format(dim, error)

print('SUCCESS: {}-d MT passes with tolerance = {}.'.format(dim, tol))
