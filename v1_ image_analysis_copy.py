# import all neccessary packages
import numpy as np
import matplotlib.pyplot as plt
import rawpy
import cv2
from scipy.integrate import quad, IntegrationWarning
from scipy.signal import savgol_filter
from matplotlib.ticker import ScalarFormatter
import os
from units_constants import *
from PIL import Image
from utilites import *  # fit_quadratic, manual_gradient, integral_summator
import warnings


class ImagePreprocess:
    width_electrode_m = 6e-3  # ВИНЕСТИ ЯК КОНСТАНТУ

    def __init__(
        self,
        filepath: str,
        image_parameters: dict,
        y_crssctn: int,
    ):
        self.filepath = filepath
        self.x_min_electrode = image_parameters.get("x_min_electrode")
        self.x_max_electrode = image_parameters.get("x_max_electrode")
        self.y_min_electrode = image_parameters.get("y_min_electrode")
        self.y_max_electrode = image_parameters.get("y_max_electrode")
        self.region_size = image_parameters.get("region_size")

        self.y_crssctn = (
            y_crssctn
            if y_crssctn is not None
            else y_min_electrode + (y_max_electrode - y_min_electrode) // 2
        )
        self.dpxl_m = self.width_electrode_m / (
            self.x_max_electrode - self.x_min_electrode
        )  # pixel size in meters

        self.rgb_image = None  # Placeholder for RGB image
        self.grayscale_image = None  # Placeholder for grayscale image

    def _convert_to_grayscale(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Convert an RGB image to grayscale using standard weights.
        """
        return np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    def read_raw_image(self) -> None:
        """
        Read a raw image and convert it to grayscale.
        """
        with rawpy.imread(self.filepath) as raw:
            self.rgb_image = raw.postprocess()
        self.grayscale_image = self._convert_to_grayscale(self.rgb_image)

    def read_jpg_image(self) -> None:
        """
        Read a JPG image and convert it to grayscale.
        """
        with Image.open(self.filepath) as img:
            self.rgb_image = np.array(img.convert("RGB"))
        self.grayscale_image = self._convert_to_grayscale(self.rgb_image)

    def read_image_based_on_extension(self):
        """
        Reads an image using the appropriate method based on its file extension.

        Args:
            image_processor (ImagePreprocess): An instance of the ImagePreprocess class.
        """
        # Extract the file extension
        _, file_extension = os.path.splitext(self.filepath)

        # Call the appropriate method based on the extension
        if file_extension.lower() == ".jpg":
            self.read_jpg_image()
        else:
            self.read_raw_image()

    def _draw_rectangle_with_overlay(
        self, bgr_image: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        """
        Draw a rectangle with a semi-transparent overlay on the image.

        Args:
            bgr_image (np.ndarray): Image to modify.
            alpha (float): Transparency factor for the overlay.

        Returns:
            np.ndarray: Image with the overlay.
        """
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
        overlay = bgr_image.copy()

        cv2.rectangle(
            overlay,
            (self.x_min_electrode, self.y_min_electrode),
            (self.x_max_electrode, self.y_max_electrode),
            (255, 0, 0),
            thickness=-1,  # Fill the rectangle
        )
        return cv2.addWeighted(overlay, alpha, bgr_image, 1 - alpha, 0)

    def draw_rectangle_with_overlay(
        self, bgr_image: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

        # Draw a red rectangle on the selected region
        cv2.rectangle(
            bgr_image,
            (self.x_min_electrode, self.y_min_electrode),
            (self.x_max_electrode, self.y_max_electrode),
            (255, 0, 0),
            thickness=4,
        )
        # Draw a red horizontal line in the center
        x_min = max(0, self.x_min_electrode)
        x_max = min(bgr_image.shape[1], self.x_max_electrode)
        y_start = max(0, self.y_crssctn - self.region_size // 2)
        y_end = min(
            bgr_image.shape[0],
            self.y_crssctn + self.region_size // 2 + (self.region_size % 2),
        )
        # Define the rectangular region (top-left and bottom-right corners)
        x_start, y_start = x_min, y_start  # Top-left corner (x, y)
        x_end, y_end = x_max, y_end  # Bottom-right corner (x, y)
        channel_to_modify = 2
        # Modify the red channel (index 2 for OpenCV BGR format) within the region

        # Create a red overlay with the same shape as the image
        overlay = bgr_image.copy()
        overlay[y_start:y_end, x_start:x_end] = [255, 0, 0]  # Red in BGR format

        # Blend the overlay with the original image
        alpha = 0.5  # Transparency factor (0.0 = fully transparent, 1.0 = fully opaque)
        bgr_image[y_start:y_end, x_start:x_end] = cv2.addWeighted(
            bgr_image[y_start:y_end, x_start:x_end],
            1 - alpha,
            overlay[y_start:y_end, x_start:x_end],
            alpha,
            0,
        )
        return bgr_image

    def edge_detection(self) -> np.ndarray:
        """
        Apply Canny edge detection to the grayscale image.

        Returns:
            np.ndarray: Binary image with edges detected.
        """
        return cv2.Canny(
            self.grayscale_image,
            threshold1=195,
            threshold2=230,
            apertureSize=5,
            L2gradient=True,
        )


class IntensityAnalysis:
    """
    This class is aimed at analyzing the intensity of an image.
    """

    def __init__(self, image_preprocessor: ImagePreprocess):
        """
        Initialize with an instance of ImagePreprocess to avoid reloading the image.

        Args:
            image_preprocessor (ImagePreprocess): Preprocessed image object.
        """
        self.image_preprocessor = image_preprocessor
        self.y_crssctn = image_preprocessor.y_crssctn
        self.region_size = image_preprocessor.region_size
        self.x_min = image_preprocessor.x_min_electrode
        self.x_max = image_preprocessor.x_max_electrode
        self.dpxl_m = image_preprocessor.dpxl_m  # Pixel size in meters

        # Ensure image is preprocessed
        if image_preprocessor.grayscale_image is None:
            raise ValueError("Image has not been loaded or processed.")

        self.grayscale_image = image_preprocessor.grayscale_image

    def extract_intensity_from_region(
        self, x_min_ROI=None, x_max_ROI=None, y_crssctn=None, region_size=None
    ):
        """
        Extract intensity values from defined rectangular regions.

        Returns:
            tuple:
                - x_array_pxl (Pixel coordinates)
                - intensity_array (Intensity values)
        """
        image = self.grayscale_image
        y_crssctn = self.y_crssctn if y_crssctn is None else y_crssctn
        region_size = self.region_size if region_size is None else region_size
        # If ROI is not mentioned, extract the full row
        x_min_ROI = 0 if x_min_ROI is None else x_min_ROI
        x_max_ROI = image.shape[1] if x_max_ROI is None else x_max_ROI

        image = self.grayscale_image
        y_start = max(0, y_crssctn - region_size // 2)
        y_end = min(image.shape[0], y_crssctn + region_size // 2 + (region_size % 2))

        x_array_pxl, intensity_array = [], []

        for x in range(x_min_ROI, x_max_ROI, region_size):
            x_start, x_end = x, min(x + region_size, image.shape[1])
            region = image[y_start:y_end, x_start:x_end]
            region_intensity = region.mean() if region.size > 0 else 0

            intensity_array.append(region_intensity)
            x_array_pxl.append((x_start + x_end) // 2)

        return np.array(x_array_pxl), np.array(intensity_array)

    def apply_square_fit(
        self, x_pxl_ROI: np.ndarray, intensity_ROI: np.ndarray
    ) -> np.ndarray:
        """
        Calls the utility function to fit intensity data.
        """
        return apply_square_fit_to_function(x_pxl_ROI, intensity_ROI)

    def x_array_rescale_to_m(
        self,
        x_array_pxl: np.ndarray,
        x_array_pxl_ROI: np.ndarray,
        intensity_fit_ROI: np.ndarray,
    ) -> np.ndarray:
        """
        Transforms cropped x_array in pixels to radius array in meters.
        Transition : x_array -> (-r, r)
        x_array - pixels on x-axis which were selected from the region (from x1 to xn [start and end pixel])
        """

        # Shift and scale x values to center around zero in meters
        min_index = np.argmin(intensity_fit_ROI)
        x_central_point = x_array_pxl_ROI[min_index]
        print(x_central_point)
        print(min_index)
        x_real_m = (x_array_pxl - x_central_point) * self.dpxl_m
        return x_real_m


class OpticalParamAnalysis(IntensityAnalysis):
    """На вхід уже зафічені квадратично інтенсивності для обчислення"""

    def __init__(
        self,
        x_array_m: np.ndarray,
        i_probe: np.ndarray,
        i_absorption: np.ndarray,
    ):
        self.x_array_m = x_array_m
        self.i_probe = i_probe
        self.i_absorption = i_absorption

    def __tau_r(self, i, i_0):
        """
        log(i_0 / i)
        """
        return np.log(i_0 / i)

    def compute_tau(self, intensity_row_absorption=None, intensity_row_gt=None):
        """
        Якщо інтенсивності не Ноне, то можна обчислити незафічені, по дефолту будуть очислені уже фітовані
        Compute the optical depth (tau) for an intensity profile by fitting it to a quadratic function.

        Args:
            intensity_row_absorption (np.ndarray): Measured intensity profile.
            intensity_row_gt (np.ndarray): Ground truth intensity profile.

        Returns:
            tuple: Fitted coefficients and optical depth profile.
                - tau (np.ndarray): Optical depth profile. (1/m)
                - tau_prime (np.ndarray): Gradient of the optical depth. (1/m)
        """
        if intensity_row_absorption is None:
            intensity_row_absorption = self.i_absorption

        if intensity_row_gt is None:
            intensity_row_gt = self.i_probe

        # Compute tau: log(I_gt / I_absorption)
        tau = self.__tau_r(intensity_row_absorption, intensity_row_gt)

        return tau

    def compute_tau_prime(self, radius: np.ndarray, tau: np.ndarray):
        """
        Computes tau prime from tau
        """

        # Compute tau_prime: derivative of tau with respect to x
        tau_prime = manual_gradient(tau, radius)

        return tau_prime

    def analysis_side_picker(
        self,
        tau_array: np.ndarray,
        radius_array_m: np.ndarray,
        right_side=True,
    ):
        """
        Splits tau array on two arrays: >0 and <0.
        Chooses specified and returns it for the further analysis.
        N - sample rate. Precisely - number of points that will be considered in analysis
        Спочатку я конвертувала пікселі в метри і змістила нуль. А тут вибираю яку сторону аналізувати: ліву від 0 чи праву
        Результат цього методу це інтенсивність та радіус
        Analysis of the right side by default

        return:
        radius_for_analysis_m, tau_for_analysis_1_cm
        """
        # Shift and scale x values to center around zero in meters
        zero_index = np.argmin(np.abs(radius_array_m))

        # Split tau in two arrays for positive and negative x
        tau_negative = tau_array[: zero_index + 1]
        tau_positive = tau_array[zero_index:]

        radius_m_negative = radius_array_m[: zero_index + 1]
        radius_m_positive = radius_array_m[zero_index:]

        radius_for_analysis_m = radius_m_positive if right_side else radius_m_negative
        tau_for_analysis_1_cm = tau_positive if right_side else tau_negative
        return radius_for_analysis_m, tau_for_analysis_1_cm

    def integrate_Abel(
        self,
        number_points: int,
        radius_m: np.ndarray,
        tau_prime: np.ndarray,
    ) -> np.ndarray:
        """
        Integrate tau.
        Input: tau, tau_prime and number of points which will define sampling of integration.
        Output: number of points which will define kappa
        """

        # Check if `number_points` exceeds available data points
        if number_points > len(radius_m):
            warnings.warn("Number of points is greater than the available radius data. Using the maximum available points.")
            number_points = len(radius_m)

        # Define the maximum radius and the sampling points
        r0 = np.max(radius_m)
        radius = np.linspace(0, r0, number_points)

        # Exclude the radius value at 0 for integration to avoid singularity
        radius_for_integration = radius[1:]

        # Initialize arrays to store integration results and errors
        integrate_result = np.empty_like(radius_for_integration)
        integrate_error = np.empty_like(radius_for_integration)

        # Loop over radius values and compute the Abel integral
        for i, r in enumerate(radius_for_integration):
            integrate_result[i], integrate_error[i] = compute_integral(
                r=r, r0=r0, tau_prime=tau_prime, x_values=radius_m
            )

        # Filter out invalid or NaN results
        valid_indices = ~np.isnan(integrate_result)
        radius_for_integration = radius_for_integration[valid_indices]
        
        integrate_result = integrate_result[valid_indices]

        integrate_error = integrate_error[
            valid_indices
        ]  # If required for error analysis

        return radius_for_integration, integrate_result, integrate_error

    def reduce_number_of_points_array(
        x_array: np.ndarray, y_array: np.ndarray, new_number_of_points: int
    ) -> np.ndarray:

        if len(x_array) != len(y_array):
            raise ValueError("x_array and y_array must have the same length.")

        if new_number_of_points <= 0:
            raise ValueError("new_number_of_points must be greater than 0.")

        # Reduce x to 15 points evenly spaced
        x_reduced = np.linspace(x_array[0], x_array[-1], new_number_of_points)

        # Interpolate y values at the new x points
        y_reduced = np.interp(x_reduced, x_array, y_array)
        return x_reduced, y_reduced


class ConcentrationCalculator:
    def __init__(self, plasma_parameters, k_B=1.38e-23, eV=1.602e-19):

        self.lambda_m = plasma_parameters.get("lambda_m")
        self.mu_Cu = plasma_parameters.get("mu_Cu")
        self.f_ik = plasma_parameters.get("f_ik")
        self.g_i = plasma_parameters.get("g_i")
        self.E_i = plasma_parameters.get("E_i")
        self.E_k = plasma_parameters.get("E_k")
        self.k_B = k_B


    def delta_lambda_doppler(self,t_K):
        d_lambda_d = 7.16e-7 * self.lambda_m * np.sqrt(t_K / self.mu_Cu)
        return d_lambda_d

    def read_t_K(self, filepath_t_K):
        data_temp = read_txt_to_array(filepath_t_K)
        t_K = data_temp[:, 1]
        d_T_K = data_temp[:,2]
        r_t_K = data_temp[:, 0]*1e-3
        return t_K, d_T_K, r_t_K

    def concentration_n_i(self, delta_lambda_m, kappa):
        # print(f"kappa: {kappa}")
        return (
            kappa
            * 1e-2
            * delta_lambda_m
            / nm
            / (8.19e-20 * self.f_ik * (self.lambda_m / nm) ** 2)
        )


    def concentration_n(self, n_i, stat_sum, g_i, T_K):
        return n_i * stat_sum / np.exp(-self.E_i / self.k_B / T_K) / g_i


    def __stat_sum_read(self, file_path, temperature_value):
        with open(file_path, "r") as file:
            data = file.readlines()
        stat_sum = np.array([float(line.strip()) for line in data])
        return stat_sum[temperature_value]


    def calculate_concentration(
        self, radius, T_K, d_T_K, kappa_profile, filepath_statsum
    ):
        n = np.empty_like(radius)
        n_i = np.empty_like(radius)
        d_lambda_m = np.empty_like(radius)
        for i, r in enumerate(radius):
            current_T_K = T_K[i]
            stat_sum = self.__stat_sum_read(filepath_statsum, int(current_T_K))
            d_lambda_m[i] = self.delta_lambda_doppler(current_T_K)
            n_i[i] = self.concentration_n_i(d_lambda_m[i], kappa_profile[i])
            n[i] = self.concentration_n(n_i[i], stat_sum, self.g_i, current_T_K)
            # print(f'T: {current_T_K}\td_lambda_m: {d_lambda_m[i]}\tn_i: {n_i[i]}\tn: {n[i]}')
            
        dn = n * self.E_i*eV*d_T_K /T_K**2/1.38e-23
        return n, dn,  n_i, d_lambda_m




