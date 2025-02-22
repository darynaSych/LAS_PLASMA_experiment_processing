import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


"""Here all math will be stored"""

def fit_quadratic(x: np.ndarray, y: np.ndarray):
        """
        y = ax^2 + bx + c
        Return: coeffs, quadratic_func
        """
        coeffs = np.polyfit(x, y, 2)  # Fit y = ax^2 + bx + c
        quadratic_func = np.poly1d(coeffs)
        return coeffs, quadratic_func


def apply_square_fit_to_function( x_pxl_ROI: np.ndarray, intensity_ROI: np.ndarray
) -> np.ndarray:
    # Fit the intensity profiles with quadratic functions
    coef_absorption, quadratic_func_absorption = fit_quadratic(
        x_pxl_ROI, intensity_ROI
    )
    # Generate the fitted intensity values
    intensity_square_fit = quadratic_func_absorption(x_pxl_ROI)
    return intensity_square_fit

    
def manual_gradient(y, x):
        dy = np.zeros_like(y)
        dx = np.diff(x)
        dy[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
        dy[0] = (y[1] - y[0]) / (x[1] - x[0])
        dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
        return dy

def integral_summator(r, r0, tau_prime, x_values):
    summ = 0
    dx = 1e-6
    r_space = np.arange(r + dx, r0, dx)
    for x in r_space:
        x_index = np.argmin(np.abs(x_values - x))
        summ += -tau_prime[x_index] * dx / (np.sqrt(x**2 - r**2) * np.pi)
    return summ


def __integrand_abel(x, r, tau_prime, x_values):
    x_index = np.argmin(np.abs(x_values - x))
    if x_index >= len(tau_prime):
        return 0
    return -tau_prime[x_index] / (np.sqrt(x**2 - r**2) * np.pi)

def compute_integral(r, r0, tau_prime, x_values):
    result, error = quad(
        __integrand_abel, r, r0, args=(r, tau_prime, x_values)
    )
    return result, error

def compute_abel_integral_trapezoidal(r, r0, tau_prime, x_values):
    integral_result = 0.0
    for i in range(1, len(x_values)):
        x1 = x_values[i - 1]
        x2 = x_values[i]
        if x1 <= r or x2 <= r:
            continue
        integrand_x2 = __integrand_abel(x2, r, tau_prime, x_values)
        integrand_x1 = __integrand_abel(x1, r, tau_prime, x_values)
        integral_result += 0.5 * (integrand_x1 + integrand_x2) * (x2 - x1)
    return integral_result


def read_txt_to_array(file_path):
    data = np.loadtxt(file_path)
    return data

def save_plot(fig, filename):
    """Save a matplotlib figure to a file."""
    fig.savefig(filename)
    plt.close(fig)
    

def interpolate_function( x_result, x_initial, y_initial):
    return np.interp(x_result, x_initial, y_initial)


