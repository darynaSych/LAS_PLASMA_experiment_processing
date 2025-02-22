# import all neccessary packages
import numpy as np
import matplotlib.pyplot as plt
import rawpy
import cv2
from scipy.integrate import quad, IntegrationWarning
import warnings
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter
from matplotlib.ticker import ScalarFormatter
import os
from units_constants import *
from PIL import Image



class ImagePreprocess:
    def __init__(
        self,
        filepath: str,
        x_min_electrode: int,
        x_max_electrode: int,
        y_min_electrode: int,
        y_max_electrode: int,
        y_crssctn: int = None,
    ):
        self.filepath = filepath
        self.x_min = x_min_electrode
        self.x_max = x_max_electrode
        self.y_min = y_min_electrode
        self.y_max = y_max_electrode
        self.y_crssctn = (
            y_crssctn if y_crssctn is not None else y_min_electrode + (y_max_electrode - y_min_electrode) // 2
        )        
        width_electrode_m = 6e-3
        self.dpxl_m = width_electrode_m/ (x_max_electrode - x_min_electrode) #pixel size


        # with rawpy.imread(filepath) as raw:
        #     self.rgb_image = raw.postprocess()  # Process the raw image to RGB

        # # RGB image to greyscale
        # self.grayscale_image = np.dot(
        #     self.rgb_image[..., :3], [0.2989, 0.5870, 0.1140]
        # ).astype(np.uint8)

        with Image.open(filepath) as img:
            # Convert to RGB (in case the image is in another mode like grayscale or CMYK)
            self.rgb_image = np.array(img.convert("RGB"))
        
        # Convert RGB to Grayscale
        self.grayscale_image = np.dot(
            self.rgb_image[..., :3], [0.2989, 0.5870, 0.1140]
        ).astype(np.uint8)

    # def extract_intensity_row(self, x_min: int, x_max: int, y_crssctn: int = None, image = None) -> np.ndarray:
    #     # If y_crssctn is not provided, use self.y_crssctn as default
    #     if y_crssctn is None:
    #         y_crssctn = self.y_crssctn
    #     if image is None:
    #         image = self.grayscale_image

    #     intensity_values = image[y_crssctn, x_min:x_max]
    #     return intensity_values
    # Треба написати так, аби повертало і рядок координат одразу. Інакше може бути неспівпадіння по елементам масиву
    def extract_intensity_row(self, x_min: int, x_max: int, y_crssctn: int = None, image=None, n: int = 10) -> np.ndarray:
        # If y_crssctn is not provided, use self.y_crssctn as default
        if y_crssctn is None:
            y_crssctn = self.y_crssctn
        if image is None:
            image = self.grayscale_image

        # Ensure the range is within bounds
        y_start = max(0, y_crssctn - n // 2)
        y_end = min(image.shape[0], y_crssctn + n // 2 + (n % 2))
        x_start = max(0, x_min)
        x_end = min(image.shape[1], x_max)

        # Extract intensity for the n x 1 region and calculate mean
        intensity_values = [
            image[y_start:y_end, x].mean() for x in range(x_start, x_end)
        ]
        return np.array(intensity_values)


    def color_rectangle(self, image_not_cropped, y_crssctn) -> np.ndarray:
        bgr_image = cv2.cvtColor(image_not_cropped, cv2.COLOR_GRAY2BGR)

        # Draw a red rectangle on the selected region
        cv2.rectangle(
            bgr_image,
            (self.x_min, self.y_min),
            (self.x_max, self.y_max),
            (255, 0, 0),
            thickness=4,
        )
        # Draw a red horizontal line in the center
        cv2.line(
            bgr_image,
            (self.x_min, y_crssctn),
            (self.x_max, y_crssctn),
            (255, 0, 0),
            thickness=4,
        )
        return bgr_image

    def edge_detection(self) -> np.ndarray:
        # Apply Canny edge detection
        t_lower = 195  # Lower threshold
        t_upper = 230  # Upper threshold
        aperture_size = 5
        L2Gradient = True

        image_edges_detection = cv2.Canny(
            self.grayscale_image,
            t_lower,
            t_upper,
            apertureSize=aperture_size,
            L2gradient=L2Gradient,
        )
        return image_edges_detection


class ImageAnalysis(ImagePreprocess):
    def __init__(self, filepath: str, x_min: int, x_max: int, y_min: int, y_max: int, dpxl_m: float):
        # Inherit initialization from ImagePreprocess
        super().__init__(filepath, x_min, x_max, y_min, y_max)
        
        self.dpxl_m = dpxl_m
        self.x_min = x_min
        self.x_max = x_max
        self.x_pxl = np.linspace(x_min, x_max, x_max - x_min)
        self.x_m = self.x_pxl * dpxl_m

    def fit_quadratic(self, x: np.ndarray, y: np.ndarray):
        coeffs = np.polyfit(x, y, 2)  # Fit y = ax^2 + bx + c
        quadratic_func = np.poly1d(coeffs)
        return coeffs, quadratic_func
    
class ParametricAnalysis():
    def __init__(self, x_min, x_max, dpxl_m):
        self.x_min = x_min
        self.x_max = x_max
        self.dpxl_m = dpxl_m
        self.x_pxl = np.linspace(x_min, x_max, x_max - x_min)
        self.x_m = self.x_pxl * dpxl_m

    def fit_quadratic(self, x: np.ndarray, y: np.ndarray):
        coeffs = np.polyfit(x, y, 2)  # Fit y = ax^2 + bx + c
        quadratic_func = np.poly1d(coeffs)
        return coeffs, quadratic_func

    def tau_r(self, i, i_0):
        return np.log(i_0 / i)

    def integrand_abel(self, x, r, tau_prime, x_values):
        x_index = np.argmin(np.abs(x_values - x))
        if x_index >= len(tau_prime):
            return 0
        return -tau_prime[x_index] / (np.sqrt(x**2 - r**2) * np.pi)

    def compute_integral(self, r, r0, tau_prime, x_values):
        result, error = quad(self.integrand_abel, r, r0, args=(r, tau_prime, x_values))
        return result, error

    def integral_summator(self, r, r0, tau_prime, x_values):
        summ = 0
        dx = 1e-6
        r_space = np.arange(r + dx, r0, dx)
        for x in r_space:
            x_index = np.argmin(np.abs(x_values - x))
            summ += -tau_prime[x_index] * dx / (np.sqrt(x**2 - r**2) * np.pi)
        return summ

    def compute_abel_integral_trapezoidal(self, r, r0, tau_prime, x_values):
        integral_result = 0.0
        for i in range(1, len(x_values)):
            x1 = x_values[i-1]
            x2 = x_values[i]
            if x1 <= r or x2 <= r:
                continue
            integrand_x1 = self.integrand_abel(x1, r, tau_prime, x_values)
            integrand_x2 = self.integrand_abel(x2, r, tau_prime, x_values)
            integral_result += 0.5 * (integrand_x1 + integrand_x2) * (x2 - x1)
        return integral_result

    def manual_gradient(self, y, x):
        dy = np.zeros_like(y)
        dx = np.diff(x)
        dy[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
        dy[0] = (y[1] - y[0]) / (x[1] - x[0])
        dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
        return dy

class ConcentrationCalculator:
    def __init__(self, lambda_m, mu_Cu, f_ik, g_i, E_i, E_k, k_B=1.38e-23, eV=1.602e-19):
        self.lambda_m = lambda_m
        self.mu_Cu = mu_Cu
        self.f_ik = f_ik
        self.g_i = g_i
        self.E_i = E_i * eV  
        self.E_k = E_k * eV
        self.k_B = k_B

    def delta_lambda_doppler(self, t_K):
        return 7.16e-7 * self.lambda_m * np.sqrt(t_K / self.mu_Cu)

    def concentration_n_i(self, delta_lambda_m, kappa):
        # print(f"kappa: {kappa}")
        return kappa * 1e-2 * delta_lambda_m / nm / (8.19e-20 * self.f_ik * (self.lambda_m / nm) ** 2)

    def concentration_n(self, n_i, stat_sum, g_i, T_K):
        return n_i * stat_sum / np.exp(-self.E_i  / self.k_B / T_K) / g_i

    def stat_sum_read(self, file_path, temperature_value):
        with open(file_path, 'r') as file:
            data = file.readlines()
        stat_sum = np.array([float(line.strip()) for line in data])
        return stat_sum[temperature_value]

    def calculate_concentration(self, radius, t_profile_interpolated, kappa_profile, filepath_statsum):
        n = np.empty_like(radius)
        n_i = np.empty_like(radius)
        d_lambda_m = np.empty_like(radius)
        for i, r in enumerate(radius):
            current_T_K = t_profile_interpolated[i]
            stat_sum = self.stat_sum_read(filepath_statsum, int(current_T_K))
            d_lambda_m[i] = self.delta_lambda_doppler(current_T_K)
            n_i[i] = self.concentration_n_i(d_lambda_m[i], kappa_profile[i])
            n[i] = self.concentration_n(n_i[i], stat_sum, self.g_i, current_T_K)
            # print(f'T: {current_T_K}\td_lambda_m: {d_lambda_m[i]}\tn_i: {n_i[i]}\tn: {n[i]}')
        return n, n_i, d_lambda_m

    @staticmethod
    def read_txt_to_array(file_path):
        data = np.loadtxt(file_path)
        return data

    def interpolate_temperature_profile(self, r_kappa, r_t_K, t_profile_K):
        return np.interp(r_kappa, r_t_K, t_profile_K)   
    
    def save_plot(fig, filename):
        """Save a matplotlib figure to a file."""
        fig.savefig(filename)
        plt.close(fig)




    
