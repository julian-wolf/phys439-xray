import spinmob as sm
import numpy   as np
import matplotlib.pyplot as plt

data_paths_Cu = ["data/Cu_" + str(percent) + ".UXD"
                 for percent in np.arange(0, 125, 25)];
data_paths_Pb = ["data/Pb_" + str(percent) + ".UXD"
                 for percent in np.arange(0, 125, 25)];

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

def lorentzian(x, bg, gamma, norm, x0):
    """
    Lorentzian function for fitting data:
    y = norm * L(x-x0;gamma) + bg
    """
    return norm * _lorentzian(x-x0, gamma) + bg

def pseudo_voigt(x, bg, sigma, gamma, norm, eta, x0):
    """
    pseudo-Voigt function for fitting data:
    y = norm * (eta*L(x-x0;gamma) + (1-eta)*G(x-x0;sigma))
    """
    G =   _gaussian(x-x0, sigma);
    L = _lorentzian(x-x0, gamma);
    return norm * (eta * L + (1 - eta) * G) + bg

def get_yerr(dataset):
    """
    Gets the statistical uncertainty in the number of counts per bin.
    Assumes no peaks in the range 10 deg < 2 theta < 20 deg;
    check to be sure this holds!
    """
    bg_data = dataset.c(1)[0:200];
    return np.std(bg_data)

def fit_peak(dataset, f, p, xmin=10, xmax=110):
    """
    Fits a function func to a single peak of dataset lying
    between xmin and xmax, with expected location x0_expected.
    """
    peak_fit = sm.data.fitter(f=f, p=p);
    yerr     = get_yerr(dataset);

    peak_fit.set_data(xdata=dataset.c(0), ydata=dataset.c(1), eydata=yerr);
    peak_fit.set(xmin=xmin, xmax=xmax);
    peak_fit.fit();

    return peak_fit

def analyze(datasets=data_Cu, fit_range=[40, 48], f="a*x+b", p="a,b"):
    """
    Automates analysis
    """
    x0 = (fit_range[1] + fit_range[0]) / 2;
    parameters = "bg=20,sigma=0.1,gamma=0.1,norm=500,eta=0.5,x0=%d" % x0;

    primary_element_percentages = np.arange(0, 125, 25);
    peak_2theta = np.zeros(len(datasets));
    peak_error  = np.zeros(len(datasets));
    good_fits   = [True] * len(datasets);
    for i in range(len(datasets)):
        first_peak_fit = fit_peak(datasets[i], pseudo_voigt, parameters,
                                  fit_range[0], fit_range[1]);

        if first_peak_fit.results[1] is None:
            good_fits[i] = False;
            continue;

        x0    = first_peak_fit.results[0][5];
        eta   = first_peak_fit.results[0][4];
        sigma = first_peak_fit.results[0][1];
        gamma = first_peak_fit.results[0][2];

        peak_2theta[i] = x0;
        peak_error[i]  = (np.abs(eta * gamma) + np.abs((1-eta) * sigma)) / 30;

    good_fits = np.array(good_fits, dtype=bool);

    primary_element_percentages = primary_element_percentages[good_fits];
    peak_2theta = peak_2theta[good_fits];
    peak_error  = peak_error[ good_fits];

    wavelength = 1.5406; # in angstroms

    peak_A  = wavelength * np.sqrt(3) / \
              (2 * np.sin(np.radians(peak_2theta / 2)));
    error_A = wavelength * np.sqrt(3) / \
               (4 * np.abs(np.cos(np.radians(peak_error / 2)) / \
                           np.sin(np.radians(peak_error / 2))**2));

    peak_offset_fit = sm.data.fitter(f=f, p=p);

    peak_offset_fit.set_data(xdata=primary_element_percentages,
                             ydata=peak_A,
                             eydata=peak_error);
    peak_offset_fit.fit();

    plt.xlabel("$\\mathrm{nominal\ percent\ copper}$");
    plt.ylabel("$\\mathrm{lattice\ constant\ } a \\mathrm{\ (\\AA)}$");

    return peak_offset_fit;
