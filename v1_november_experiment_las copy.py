import os
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter
from scipy.signal import savgol_filter
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
# Import the classes
from image_analysis import (
    ImagePreprocess,
    ParametricAnalysis,
    ConcentrationCalculator,
)


filepath_ = "_DSC2834.NEF"
filepath_0 = "_DSC2821.NEF"
x_min_electrode = 1930
x_max_electrode = 4000
y_min = 1650
y_max = 3600
x_minROI = 1500
x_maxROI  = 4700

# Instantiate the class
image_preprocess = ImagePreprocess(
    filepath_, x_min_electrode, x_max_electrode, y_min, y_max
)
image_preprocess_0 = ImagePreprocess(
    filepath_0, x_min_electrode, x_max_electrode, y_min, y_max
)

# Display the image with the red rectangular selection
fig, [ax1, ax2] = plt.subplots(1, 2)
# Display grayscale image
ax1.imshow(
    image_preprocess.color_rectangle(
        image_not_cropped=image_preprocess.grayscale_image
    ), cmap = 'gray'
)
ax1.set_title("Grayscale image")
ax1.set_xlabel("x [pxl]")
ax1.set_ylabel("y [pxl]")
# Display edge-detected image
ax2.imshow(
    image_preprocess.color_rectangle(
        image_not_cropped=image_preprocess.edge_detection()
    ),
    cmap="gray",
)
ax2.set_title("Edge-detected image")
ax2.set_xlabel("x [pxl]")
ax2.set_ylabel("y [pxl]")

intensity_crssctn = image_preprocess.extract_intensity_row(
    x_min=x_minROI,x_max= x_maxROI, y_crssctn=None
)
smoothed_intensity = savgol_filter(intensity_crssctn, window_length=51, polyorder=2)  # window_length and polyorder can be adjusted

fig, [ax1, ax2] = plt.subplots(1,2)
ax1.plot(np.arange(x_minROI,x_maxROI),intensity_crssctn, label = 'intensity')
ax1.plot(np.arange(x_minROI,x_maxROI),smoothed_intensity, label = 'smoothed intensity')
ax1.set_title(f"Cropped towards ROI {x_minROI}x{x_maxROI}")
ax1.set_xlabel("x [pxl]")
ax1.set_ylabel("Intensity")
ax1.legend()

ax2.plot(image_preprocess.extract_intensity_row(
    x_min=0,x_max= 6000, y_crssctn=None
), label = 'full row')
ax2.set_title("Full width")
ax2.set_xlabel("x [pxl]")
ax2.set_ylabel("Intensity")
ax2.legend()


x_min=2500
x_max=4500

# Create a new dataarray, where data from chosen previous range would be used
# Extract intensity values (replace with actual image processing calls)
intensity = image_preprocess.extract_intensity_row(x_min=x_min, x_max=x_max,  y_crssctn = None)
intensity_0 = image_preprocess_0.extract_intensity_row(x_min=x_min, x_max=x_max,  y_crssctn = None)
analysis = ParametricAnalysis(x_min=x_min, x_max=x_max, dpxl_m=image_preprocess.dpxl_m)

# Fit the intensity profile with a quadratic function
coef, quadratic_func = analysis.fit_quadratic(analysis.x_pxl, intensity)
coef_0, quadratic_func_0 = analysis.fit_quadratic(analysis.x_pxl, intensity_0)

# Generate the fitted y values
y_fitted = quadratic_func(analysis.x_pxl)
y_fitted_0 = quadratic_func_0(analysis.x_pxl)

# Plot the intensity profiles and fitted curves
colorsBlue = plt.cm.Blues(np.linspace(0.55, 0.95, 20))  # Light to dark blue
colorsOrange = plt.cm.Oranges(np.linspace(0.45, 0.85, 20))
plt.figure()
plt.plot(analysis.x_pxl, intensity_0, color=colorsBlue[5], label=r"Reference intensity, $I_{ref}$")
plt.plot(analysis.x_pxl, y_fitted_0, '--', color=colorsOrange[5], label=r"Result of fit $I_{ref\,fit}$")
plt.plot(analysis.x_pxl, intensity, color=colorsBlue[15], label=r"Intensity in region of absorption, $I_{abs}$")
plt.plot(analysis.x_pxl, y_fitted, '--', color=colorsOrange[15], label=r"Result of fit $I_{abs\,fit}$")
plt.xlabel("Column Index, x [pxl]")
plt.ylabel("Intensity")
plt.title("Intensity Profile and Quadratic Fit")
plt.legend(loc='upper right')

# Compute tau
tau = analysis.tau_r(y_fitted, y_fitted_0)

print(f"y_fit{y_fitted}\ty_0{y_fitted_0}")
print(analysis.tau_r(104.13336246,136.0970849))

# Shift and scale x values to center around zero in meters
x_central_point = x_min + np.argmin(y_fitted)
x_real_m = (analysis.x_pxl - x_central_point) * analysis.dpxl_m
zero_index = np.argmin(np.abs(x_real_m))

# Split tau in two arrays for positive and negative x
tau_negative = tau[:zero_index + 1]
tau_positive = tau[zero_index:]
x_real_m_negative = x_real_m[:zero_index + 1]
x_real_m_positive = x_real_m[zero_index:]

# Plot tau
plt.figure()
plt.plot(x_real_m, tau)
plt.plot(x_real_m_negative, tau_negative, 'o')
plt.plot(x_real_m_positive, tau_positive, 'o')
plt.xlabel("x [m]")
plt.ylabel(r"$\tau$")

# Set up plot for tau(r)
fig3, ax3 = plt.subplots()
ax3.plot(x_real_m_positive * 1e3, tau_positive)  # Convert x to mm
ax3.set_xlabel("x [mm]")
ax3.set_ylabel(r"$\tau_{0}$")
ax3.set_xlim(left=0)
ax3.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax3.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
plt.show()
# Compute the inverse Abel transform using integral and summation methods
r0 = max(x_real_m)
N = 25
dr = r0 / N
radius = np.linspace(1e-5, r0, N)
integrate_result = np.empty_like(radius)
tau_prime = analysis.manual_gradient(tau_positive, x_real_m_positive)

# Loop over the radius array and compute Abel integral
for i, r in enumerate(radius):
    integrate_result[i] = analysis.compute_integral(r=r, r0=r0, tau_prime=tau_prime, x_values=x_real_m_positive)
# Plot inverse Abel transform results
fig2, ax2 = plt.subplots()
ax2.scatter(radius * 1e3, integrate_result * 1e-2)  # Convert r to mm and results to 1/cm
ax2.set_title("Inverse Abel Transform")
ax2.set_xlabel("r [mm]")
ax2.set_ylabel(r"$\kappa_{0}\;[1/cm]$")
ax2.set_xlim(0, r0 * 1e3)


# Compute Abel integral using `compute_integral` method for a single r
r_single = radius[0]
integral_result = analysis.compute_integral(r=r_single, r0=r0, tau_prime=tau_prime, x_values=x_real_m_positive)
print(f"Abel integral at r={r_single}: {integral_result}")


# Constants and inputs
lambda_m = 510.5*nm  # Wavelength in meters
mu_Cu = 64  # Atomic mass of copper
f_ik = 0.00328  # Oscillator strength
g_i = 6  # Statistical weight for the lower energy level
E_i = 1.38  # Energy of level in eV
E_k = 3.82  # Energy of upper level in eV
foldername = 'plots_and_results'
filename_statsum = 'Statsum_CuI.txt'
filename_temperature = 'temperature_profile.txt'
filepath_statsum = os.path.join(foldername, filename_statsum)
filepath_temperature = os.path.join(foldername, filename_temperature)

# Initialize class
concentration_calculator = ConcentrationCalculator(lambda_m, mu_Cu, f_ik, g_i, E_i, E_k)

# Load temperature profile
data_temp = concentration_calculator.read_txt_to_array(filepath_temperature)
t_profile_K = data_temp[:, 1]
r_t_K = data_temp[:, 0]*1e-3

# Example data for kappa and radius from previous computation
kappa_profile = integrate_result  
r_kappa = radius  

# Interpolate the temperature profile to match the length of kappa_profile
t_profile_interpolated = concentration_calculator.interpolate_temperature_profile(r_kappa, r_t_K, t_profile_K)

# Calculate concentration
n_values, n_i, d_lambda_m  = concentration_calculator.calculate_concentration(r_kappa, t_profile_interpolated, kappa_profile, filepath_statsum)


# Plot kappa out of radious
plt.figure()
plt.plot(r_kappa, kappa_profile, label='Kappa Profile')
plt.xlabel("Radius [m]")
plt.ylabel(r"Kappa")

#plot temperature profile
fig, ax = plt.subplots()
ax.plot(r_t_K/mm, t_profile_K, "o", alpha=0.5, label='Input Temperature Profile')
ax.plot(r_kappa/mm, t_profile_interpolated, 'o', alpha=0.5, label='Interpolated Temperature Profile')
ax.set_xlabel("Radius [mm]")
ax.set_ylabel("Temperature [K]")
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
plt.legend()

#Plot Doplers broadening
plt.figure()
plt.plot(r_kappa * 1e3, d_lambda_m*1e9, "o", alpha=0.5)
plt.xlabel("Radius [mm]")
plt.ylabel(r"Doplers broadening, $\Delta \lambda_{D}$ [$nm$]")
plt.legend()
ax = plt.gca()
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

# Population number density
plt.figure()
plt.plot(r_kappa * 1e3, n_i*1e6, "o", alpha=0.5, label='Concentration')
plt.xlabel("Radius [mm]")
plt.ylabel(r"Population number density $n_{i}$ [m$^{-3}$]")
plt.yscale("log")
plt.ylim(5e18,1e20)

# Plot number density (m)
plt.figure()
plt.plot(r_kappa * 1e3, n_values*1e6, "o", alpha=0.5, label='Concentration')
plt.xlabel("Radius [mm]")
plt.ylabel(r"Number density $n$ [m$^{-3}$]")
plt.yscale("log")
plt.ylim(1e19,1e21)

print(r_kappa* 1e3)
print(n_values*1e6)
for i in range(len(r_kappa)):
    print(f"{r_kappa[i]* 1e3}\t{n_values[i]*1e6}\n")

# Plot number density (m)
plt.figure()
plt.plot(r_kappa * 1e3, n_values , "o-", alpha=0.5, label='Concentration')
plt.xlabel("Radius [mm]")
plt.ylabel(r"Number density $n$ [cm$^{-3}$]")

plt.legend()

plt.show(block=False)
input("Press any key to close all plots...")
plt.close('all')