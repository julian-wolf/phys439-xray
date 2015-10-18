import spinmob as sm
import numpy   as np

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

def gaussian(x, bg, sigma, norm, x0):
    """
    Gaussian function for fitting data:
    y = norm * G(x-x0;sigma) + bg
    """
    return norm * _gaussian(x-x0, sigma) + bg

def _multiple_gaussian(x, bg, sigma, norm, x0):
    """
    Gaussian with multiple peaks for fitting data:
    y = sum(norm[i] * G(x-x0[i];sigma[i])) + bg
    """
    n_peaks = len(x0);
    assert(len(sigma) == n_peaks and len(norm) == n_peaks);

    distribution = bg;
    for i in range(n_peaks):
        distribution += norm[i] * _gaussian(x-x0[i], sigma[i]);
    return distribution

def gaussian2(x, bg, s0, s1, n0, n1, x0, x1):
    return _multiple_gaussian(x, bg, [s0, s1],
                                     [n0, n1],
                                     [x0, x1])
def gaussian3(x, bg, s0, s1, s2, n0, n1, n2, x0, x1, x2):
    return _multiple_gaussian(x, bg, [s0, s1, s2],
                                     [n0, n1, n2],
                                     [x0, x1, x2])
def gaussian4(x, bg, s0, s1, s2, s3, n0, n1, n2, n3, x0, x1, x2, x3):
    return _multiple_gaussian(x, bg, [s0, s1, s2, s3],
                                     [n0, n1, n2, n3],
                                     [x0, x1, x2, x3])
def gaussian5(x, bg, s0, s1, s2, s3, s4, n0, n1, n2, n3, n4, x0, x1, x2, x3, x4):
    return _multiple_gaussian(x, bg, [s0, s1, s2, s3, s4],
                                     [n0, n1, n2, n3, n4],
                                     [x0, x1, x2, x3, x4])

def lorentzian(x, bg, gamma, norm, x0):
    """
    Lorentzian function for fitting data:
    y = norm * L(x-x0;gamma) + bg
    """
    return norm * _lorentzian(x-x0, gamma) + bg

def _multiple_lorentzian(x, bg, gamma, norm, **kwargs):
    """
    Lorentzian with multiple peaks for fitting data:
    y = sum(norm[i] * L(x-x0[i];gamma[i])) + bg
    """
    n_peaks = len(x0);
    assert(len(gamma) == n_peaks and len(norm) == n_peaks);

    distribution = bg;
    for i in range(n_peaks):
        distribution += norm[i] * _gaussian(x-x0[i], gamma[i]);
    return distribution

def lorentzian2(x, bg, g0, g1, n0, n1, x0, x1):
    return _multiple_lorentzian(x, bg, [g0, g1],
                                       [n0, n1],
                                       [x0, x1])
def lorentzian3(x, bg, g0, g1, g2, n0, n1, n2, x0, x1, x2):
    return _multiple_lorentzian(x, bg, [g0, g1, g2],
                                       [n0, n1, n2],
                                       [x0, x1, x2])
def lorentzian4(x, bg, g0, g1, g2, g3, n0, n1, n2, n3, x0, x1, x2, x3):
    return _multiple_lorentzian(x, bg, [g0, g1, g2, g3],
                                       [n0, n1, n2, n3],
                                       [x0, x1, x2, x3])
def lorentzian5(x, bg, g0, g1, g2, g3, g4, n0, n1, n2, n3, n4, x0, x1, x2, x3, x4):
    return _multiple_lorentzian(x, bg, [g0, g1, g2, g3, g4],
                                       [n0, n1, n2, n3, n4],
                                       [x0, x1, x2, x3, x4])

def pseudo_voigt(x, bg, sigma, gamma, norm, eta, x0):
    """
    pseudo-Voigt function for fitting data:
    y = norm * (eta*L(x-x0;gamma) + (1-eta)*G(x-x0;sigma))
    """
    G = _gaussian(  x-x0, sigma);
    L = _lorentzian(x-x0, gamma);
    return norm * (eta * L + (1 - eta) * G) + bg

def multiple_pseudo_voigt(x, bg, sigma, gamma, norm, eta, **kwargs):
    """
    pseudo-Voigt with multiple peaks for fitting data:
    y = sum(norm[i] * (eta[i]*L(x-x0[i];gamma[i]) + (1-eta[i])*G(x-x0[i];sigma[i]))
    """
    n_peaks = len(x0);
    distribution = bg;
    for _ , x0 in kwargs.iteritems():
        G = _gaussian(  x-x0, sigma[i]);
        L = _lorentzian(x-x0, gamma[i]);
        distribution += norm * (eta * L + (1-eta) * G);
    return distribution # TODO: broken

def get_yerr(dataset):
    """
    Gets the statistical uncertainty in the number of counts per bin.
    Assumes no peaks in the range 10 deg < 2 theta < 20 deg;
    check to be sure this holds!
    """
    bg_data = dataset.c(1)[0:200];
    return np.std(bg_data)

def fit_peak(dataset, func, parameters, xmin=10, xmax=110):
    """
    Fits a function func to a single peak of dataset lying
    between xmin and xmax, with expected location x0_expected.
    """
    peak_fit = sm.data.fitter(f=func, p=parameters);
    yerr     = get_yerr(dataset);

    peak_fit.set_data(xdata=dataset.c(0), ydata=dataset.c(1), eydata=yerr);
    peak_fit.set(xmin=xmin, xmax=xmax);
    peak_fit.fit();

    return peak_fit
