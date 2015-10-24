import spinmob as sm
import numpy   as np
from scipy.optimize import minimize_scalar

data_paths_Cu = ["data/Cu_" + str(percent) + ".UXD"
                 for percent in np.arange(0, 125, 25)]
data_paths_Pb = ["data/Pb_" + str(percent) + ".UXD"
                 for percent in np.arange(0, 125, 25)]

data_Cu = sm.data.load_multiple(paths=data_paths_Cu)
data_Pb = sm.data.load_multiple(paths=data_paths_Pb)

def _gaussian(x, sigma):
    norm_factor  = 1 / (sigma * np.sqrt(2 * np.pi))
    distribution = np.exp(-x**2 / (2 * sigma**2))
    return norm_factor * distribution

def _lorentzian(x, gamma):
    norm_factor  = gamma / np.pi
    distribution = 1 / (x**2 + gamma**2)
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
    G =   _gaussian(x-x0, sigma)
    L = _lorentzian(x-x0, gamma)
    return norm * (eta * L + (1 - eta) * G) + bg

def get_yerr(dataset):
    """
    Gets the statistical uncertainty in the number of counts per bin.
    Assumes no peaks in the range 10 deg < 2 theta < 20 deg
    check to be sure this holds!
    """
    bg_data = dataset.c(1)[0:200]

    bg_std  = np.std( bg_data)
    bg_mean = np.mean(bg_data)

    return bg_std * np.sqrt(dataset.c(1) / bg_mean)

def fit_peak(dataset, f, p, xmin=10, xmax=110):
    """
    Fits a function func to a single peak of dataset lying
    between xmin and xmax, with expected location x0_expected.
    """
    peak_fit = sm.data.fitter(f=f, p=p, plot_guess=True)
    yerr     = get_yerr(dataset)

    peak_fit.set_data(xdata=dataset.c(0), ydata=dataset.c(1), eydata=yerr)
    peak_fit.set(xmin=xmin, xmax=xmax)
    peak_fit.fit()

    return peak_fit

def analyze(d='Cu', f="a*x+b", p="a,b", fudge_errors=False):
    """
    Automates analysis
    """
    datasets = {'Cu' : data_Cu,      'Pb' : data_Pb}
    x0       = {'Cu' : [44] * 5,     'Pb' : [32.1, 31.5, 31.4]}
    xmins    = {'Cu' : [40] * 5,     'Pb' : [26,   26,   26]}
    xmaxs    = {'Cu' : [48] * 5,     'Pb' : [35,   35,   35]}
    fit_func = {'Cu' : pseudo_voigt, 'Pb' : gaussian}

    datasets = datasets[d]
    x0       = x0[d]
    xmins    = xmins[d]
    xmaxs    = xmaxs[d]
    fit_func = fit_func[d]

    parameters_base = {'Cu' : "bg=20,sigma=0.1,gamma=0.1,norm=1000,eta=0.5,x0=",
                       'Pb' : "bg=3,sigma=0.01,norm=1000,x0="}
    parameters_base = parameters_base[d]

    primary_element_percentages = np.arange(0, 125, 25)
    peak_2theta = np.zeros(len(datasets))
    peak_error  = np.zeros(len(datasets))
    good_fits   = [True] * len(datasets)
    for i in range(len(datasets)):
        parameters = parameters_base + str(x0[i])
        first_peak_fit = fit_peak(datasets[i], fit_func, parameters,
                                  xmins[i], xmaxs[i])

        if first_peak_fit.results[1] is None:
            good_fits[i] = False
            continue

        peak_2theta[i] = first_peak_fit.results[0][5]
        peak_error[i]  = first_peak_fit.results[1][5][5]
        peak_error[i]  = np.sqrt(peak_error[i])

    good_fits = np.array(good_fits, dtype=bool)

    primary_element_percentages = primary_element_percentages[good_fits]
    peak_2theta = peak_2theta[good_fits]
    peak_error  = peak_error[ good_fits]

    wavelength = 1.5406; # in angstroms

    peak_A  = wavelength * np.sqrt(3) / \
              (2 * np.sin(np.radians(peak_2theta / 2)))
    error_A = wavelength * np.sqrt(3) / \
               (4 * np.abs(np.cos(np.radians(peak_error / 2)) / \
                           np.sin(np.radians(peak_error / 2))**2))
    if fudge_errors:
        def minimize_chi2(error_norm_factor):
            peak_offset_fit = sm.data.fitter(f=f, p=p, autoplot=False)
            peak_offset_fit.set_data(xdata=primary_element_percentages,
                                     ydata=peak_A,
                                     eydata=error_norm_factor*error_A)
            peak_offset_fit.fit()
            return (peak_offset_fit.reduced_chi_squareds()[0] - 1)**2

        error_norm_factor = minimize_scalar(minimize_chi2).x
        error_A *= error_norm_factor

        print "Multiplying errors by fudge factor %f" % error_norm_factor

    peak_offset_fit = sm.data.fitter(f=f, p=p, plot_guess=False)
    peak_offset_fit.set_data(xdata=primary_element_percentages,
                             ydata=peak_A,
                             eydata=error_A)
    peak_offset_fit.fit()

    return peak_offset_fit

def print_data_to_columns(sm_fit, fname, residuals=False):
    xmin = sm_fit._settings['xmin']
    xmax = sm_fit._settings['xmin']

    xdata  = sm_fit.get_data()[0][0]
    i_used = (xdata >= xmin) & (xdata <= xmax)
    xdata  = xdata[i_used]

    n_data = len(xdata);
    if not residuals:
        ydata  = sm_fit.get_data()[1][0][i_used]
        eydata = sm_fit.get_data()[2][0][i_used]
    else:
        ydata  = sm_fit.studentized_residuals()[0]
        eydata = sm_fit.get_data()[2][0][i_used] / sm_fit.get_data()[1][0][i_used]

    with open(fname, 'w') as f_out:
        for i in range(n_data):
            print "n_data = %d\ti = %d\txdata[i] = %f\n" % (n_data, i, xdata[i])
            entry = "%f\t%f\t%f\n" % (xdata[i], ydata[i], eydata[i])
            f_out.write(entry)

    return
