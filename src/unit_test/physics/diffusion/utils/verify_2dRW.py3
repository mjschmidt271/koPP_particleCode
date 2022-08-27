#!/usr/bin/env python3

import numpy as np
from numpy import linalg as la
from scipy.stats import norm

fname = "./data/particles"
start_fname = fname + "1.txt"
f_ens = "./data/ens.txt"

with open(f_ens) as f:
    N_ens = int(f.readline())

with open(start_fname) as f:
    shapeData = f.readline()
    dim = int(f.readline())
    p = f.readline()

omega = np.zeros(2)

params = p.split()
IC_type_space = float(params[0])
IC_type_mass = float(params[1])
omega[0] = float(params[2])
omega[1] = float(params[3])
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

X = np.ndarray([dim, Np, Nsteps])
tmpX = np.ndarray([Np * Nsteps])
tmpY = np.ndarray([Np * Nsteps])
tmpXY = np.ndarray([dim, Np, Nsteps])

for e in range(1, N_ens + 1):
    fname_ens = fname + str(e) + ".txt"
    with open(fname_ens) as f:
        data = np.loadtxt(f, skiprows=3)
        tmpX = np.reshape((data[:, 0]), [Np, Nsteps], 'f')
        tmpY = np.reshape((data[:, 1]), [Np, Nsteps], 'f')
        tmpXY = np.stack((tmpX, tmpY), axis=0)
        if e == 1:
            X = tmpXY
        else:
            X = np.append(X, tmpXY, axis=1)

L = omega[1] - omega[0]

def analytic2d(X, Y, t, D, X0):
    sol = (1 / (4 * np.pi * D * t))\
          * np.exp(-(((X0 - X)**2 + (X0 - Y)**2) / (4 * D * t)));
    return sol

asoln = analytic2d(X[0, :, -1], X[1, :, -1], maxT, D, X0_space)

def analytic2d_fit(_X, _Y, _sigma_x, _sigma_y, _mean_x, _mean_y):
    sol = (1 / (2 * np.pi * (_sigma_x * _sigma_y)))\
          * np.exp(-((_mean_x - _X)**2 / (2 * _sigma_x**2) +
                      (_mean_y - _Y)**2 / (2 * _sigma_y**2)))
    return sol


def analytic_fit(_X, sigma_x, sigma_y, mean_x, mean_y):
    XXX, YYY = np.meshgrid(_X[0, :], _X[1, :])
    sol = analytic2d_fit(XXX, YYY, sigma_x, sigma_y, mean_x, mean_y)
    return np.ravel(sol)

from scipy.optimize import curve_fit

# Produce 2D histogram
nBins = 15
H, xedges, yedges = np.histogram2d(X[0, :, -1], X[1, :, -1], nBins, density=True)
H = np.ravel(H)
bin_centers_x = (xedges[:-1] + xedges[1 :]) / 2.0
bin_centers_y = (yedges[:-1] + yedges[1 :]) / 2.0
pt_histmax = H.max()

Xin = np.vstack([bin_centers_x, bin_centers_y])

an_stdDev = np.sqrt(2.0 * D * maxT)
# Curve Fit parameters--Note that coeff = [sigma_x, sigma_y, mean_x, mean_y]
coeff, var_matrix = curve_fit(analytic_fit, Xin, H,
                              p0=[an_stdDev, an_stdDev, 25, 25], method='lm')

def analytic_fit_unravel(_x, _y, sigma_x, sigma_y, mean_x, mean_y):
    sol = analytic2d_fit(_x, _y, sigma_x, sigma_y, mean_x, mean_y)
    return sol

xplot = np.linspace(min(X[0, :, -1]), max(X[0, :, -1]), 100)
yplot = np.linspace(min(X[1, :, -1]), max(X[1, :, -1]), 100)
xx, yy = np.meshgrid(xplot, yplot)

fit_ans_pts = analytic2d_fit(X[0, :, -1], X[1, :, -1], coeff[0], coeff[1], coeff[2], coeff[3])
fit_ans_mgrid = analytic_fit_unravel(xx, yy, coeff[0], coeff[1], coeff[2], coeff[3])
fit_ans_mgrid = analytic_fit_unravel(xx, yy, coeff[0], coeff[1], coeff[2], coeff[3])
asoln_surf = analytic2d(xx, yy, maxT, D, X0_space)
abs_particle_error_mg = np.abs(fit_ans_mgrid - asoln_surf)
abs_particle_error_pts = np.abs(fit_ans_pts - asoln)

particle_mean = [np.mean(X[0, :, -1]), np.mean(X[1, :, -1])]
particle_std = [np.std(X[0, :, -1]), np.std(X[1, :, -1])]

print(f'analytic max = {np.max(np.max(asoln)):.4}')
print(f'sample max = {pt_histmax:.4}')
print(f'analytic mean = [{X0_space:.4}, {X0_space:.4}]')
print(f'sample mean = [{particle_mean[0]:.4}, {particle_mean[1]:.4}]')
print(f'analytic sigma_[x,y] = sqrt(2 D maxT) = {np.sqrt(2.0 * D * maxT):.4}')
print(f'sample std. dev. [sigma_x, sigma_y] =  [{particle_std[0]:.4},\
      {particle_std[1]:.4}]')
print()

print('2-norm error (meshgrid) = {}'.format(la.norm(abs_particle_error_mg)))
print('inf-norm error (meshgrid) = {}'.format(la.norm(abs_particle_error_mg,\
                                                      np.inf)))
print('MSE (meshgrid) = {}'.format(np.mean(abs_particle_error_mg**2)))
print()
print('2-norm error (pt locs) = {}'.format(la.norm(abs_particle_error_pts)))
print('inf-norm error (pt locs) = {}'.format(la.norm(abs_particle_error_pts,\
                                                     np.inf)))
print('MSE (pt locs) = {}'.format(np.mean(abs_particle_error_pts**2)))
print()

print('max analytic (particles, meshgrid) = ' +
      '{}, {}'.format(np.amax(asoln), np.amax(np.amax(asoln_surf))))
print('max fitted (particles, meshgrid) = ' +
      '{}, {}'.format(np.amax(fit_ans_pts), np.amax(fit_ans_mgrid)))
print('maxval error (particles) = ' +
      '{}'.format(np.amax(asoln) - np.amax(np.amax(fit_ans_pts))))
print('maxval error (meshgrid) = ' +
      '{}'.format(np.abs(np.amax(asoln_surf) - np.amax(fit_ans_mgrid))))
print()

print('fitted: [mu_x, mu_y], [sigma_x, sigma_y] = ' +
      '[{:.4f}, {:.4f}], [{:.4f}, {:.4f}]'.format(coeff[3], coeff[2], coeff[1], coeff[0]))
print('particle: [mu_x, mu_y], [sigma_x, sigma_y]' +
      '= [{:.4f}, {:.4f}], [{:.4f}, {:.4f}]'.format(particle_mean[0], particle_mean[1],
      particle_std[0], particle_std[1]))
print('analytic: [mu_x, mu_y], [sigma_x, sigma_y]' +
      '= [{:.4f}, {:.4f}], [{:.4f}, {:.4f}]'.format(X0_space, X0_space,
      np.sqrt(2.0 * D * maxT), np.sqrt(2.0 * D * maxT)))
print()

print('abs difference btw. max of analytical, fitted (pt locs) = ' +
      '{}'.format(np.abs(np.amax(asoln) - np.amax(fit_ans_pts))))
print('abs difference btw. max of analytical, fitted (meshgrid) = ' +
      '{}'.format(np.abs(np.amax(asoln_surf) - np.amax(fit_ans_mgrid))))
print('abs difference in mean_[x,y], std_[x,y] btw. analytical, fitted = ' +
      '[{:.4f}, {:.4f}], [{:.4f}, {:.4f}]'.format(np.abs(coeff[3] - X0_space),
                                  np.abs(coeff[2] - X0_space),
                                  np.abs(coeff[1] - np.sqrt(2.0 * D * maxT)),
                                  np.abs(coeff[0] - np.sqrt(2.0 * D * maxT))))
print()

print('abs difference btw. max of analytical, PT sim. (sample histogram) = ' +
      '{}'.format(np.abs(np.amax(asoln) - pt_histmax)))
print('abs difference in mean_[x,y], std_[x,y] btw. analytical, PT sim,' +
      ' (sample) = [{:.4f}, {:.4f}], [{:.4f}, {:.4f}]'.format(
      np.abs(particle_mean[0] - X0_space),
      np.abs(particle_mean[1] - X0_space),
      np.abs(particle_std[0] - np.sqrt(2.0 * D * maxT)),
      np.abs(particle_std[1] - np.sqrt(2.0 * D * maxT))))

mse = np.mean(abs_particle_error_mg**2)
error_max_val = np.abs(np.amax(asoln_surf) - np.amax(fit_ans_mgrid))
error_mean = [np.abs(particle_mean[0] - X0_space),
              np.abs(particle_mean[1] - X0_space)]
error_std = [np.abs(particle_std[0] - np.sqrt(2.0 * D * maxT)),
             np.abs(particle_std[1] - np.sqrt(2.0 * D * maxT))]

xy_str = ['x', 'y']

mse_tol = 1.0e-6
assert mse <= mse_tol, (f'2D MSE error too high: error = {mse:.4e}. ' +
                        f'tol = {mse_tol:.1e}.')
maxval_tol = 5.0e-3
assert error_max_val <= maxval_tol,\
    (f'2D Max Value error too high: error = {error_max_val:.4e}. ' +
     f'tol = {maxval_tol:.1e}.')
mean_tol = 1.0e-1
for coord, err in zip(xy_str, error_mean):
    assert err <= mean_tol, (f'2D Mean error too high in {coord}-coordinate: ' +
                             f'error = {err:.4e}. ' +
                             f'tol = {mean_tol:.1e}.')
std_tol = 1.0e-1
for coord, err in zip(xy_str, error_std):
    assert err <= std_tol, (f'2D Std. Dev. error too high in {coord}-coordinate: ' +
                             f'error = {err:.4e}. ' +
                             f'tol = {std_tol:.1e}.')
print(f'SUCCESS: {dim}-d RW passes with tolerances: ' +
       f'mse_tol = {mse_tol:.1e}, ' +
       f'maxval_tol = {maxval_tol:.1e}, ' +
       f'mean_tol = {mean_tol:.1e}, ' +
       f'std_tol = {std_tol:.1e}.')
