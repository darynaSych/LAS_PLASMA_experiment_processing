import numpy as np
from units_constants import *
from utilites import *  
import warnings

class OpticalParamAnalysis():
    """На вхід уже зафічені квадратично інтенсивності для обчислення"""

    def __init__(
        self,
        x_array_m: np.ndarray,
        intensity_probe: np.ndarray,
        intensity_absorption: np.ndarray,
    ):
        self.x_array_m = x_array_m
        self.i_probe = intensity_probe
        self.i_absorption = intensity_absorption

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

    def tau_derivative(self, radius: np.ndarray, tau: np.ndarray):
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

        radius_for_analysis_m = radius_m_positive if right_side else -radius_m_negative
        tau_for_analysis_1_cm = tau_positive if right_side else tau_negative
        if len(radius_for_analysis_m) == 0 or len(tau_for_analysis_1_cm) == 0:
            raise ValueError("Selected side for analysis is empty. Check input data or ROI.")
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
