#! /usr/bin/python

import spinmob as sm
import numpy as np
import sys

data_paths_Cu = ["data/Cu_" + str(percent) + ".UXD" for percent in np.arange(0,125,25)];
data_paths_Pb = ["data/Pb_" + str(percent) + ".UXD" for percent in np.arange(0,125,25)];

data_Cu = sm.data.load_multiple(paths=data_paths_Cu);
data_Pb = sm.data.load_multiple(paths=data_paths_Pb);

def _gaussian(x, sigma):
    norm_factor  = 1 / (sigma * np.sqrt(2 * np.pi));
    distribution = np.exp(-x**2 / (2 * sigma**2));
    return norm_factor * distribution

def _lorentzian(x, gamma):
    norm_factor  = gamma / np.pi;
    distribution = 1 / (x**2 + gamma**2);
    return norm_factor * distribution

def pseudo_voigt(x, x0, sigma, gamma, norm, eta):
    """
    pseudo-voigt function for fitting data:
    y = norm * (eta * L(x-x0;gamma) + (1 - eta) * G(x-x0;sigma))
    """
    G = _gaussian(x - x0, sigma);
    L = _lorentzian(x - x0, gamma);
    return norm * (eta * L + (1 - eta) * G)

def fit_peak(dataset, x0_expected):
    parameters = "x0=%d, sigma=1, gamma=1, norm=1, eta=0.5" % (x0_expected,);
    voigt_fit  = sm.data.fitter(f=pseudo_voigt, p=parameters);

    voigt_fit.set_data(dataset, eydata=1, exdata=0.1);
    voigt_fit.fit();

    return voigt_fit
