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
from v1_image_analysis import (
    ImagePreprocess,
    ParametricAnalysis,
    ConcentrationCalculator,
)

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from units_constants import *
from v1_image_analysis import ImagePreprocess, ParametricAnalysis, ConcentrationCalculator

"""
Ця версія ще буде зберігати в .тхт отримані значення заселеностей та концентрацій 
"""

# SET THE PARAMETERS OF IMAGE

# Folders and profiles
foldername_img = 'Photos_19-12'
filename_img_absorption = "_DSC3426.jpg" # зображення з поглинанням
filename_img_gt = "_DSC3417o.jpg" # ground truth image
filepath_img_absorption = os.path.join(foldername_img, filename_img_absorption)
filepath_img_gt =os.path.join(foldername_img, filename_img_gt)
foldername_savefig = 'Photos_19-12/3426'

# filepath_img_absorption = "_DSC3231.NEF" # зображення з поглинанням
# filepath_img_gt = "_DSC3217.NEF" # ground truth image

x_min_electrode = 2041 # left lim of electrode
x_max_electrode = 4154 # right lim of electrode

y_min_electrode = 1078 # lower limit of an electrode
y_max_electrode = 2820#2993  # upper limit of an electrode

# Create ROI which determines the boundaries of analysis
x_min=2100
x_max=3800
x_minROI = x_min # Define ROI for preanalysis
x_maxROI  = x_max



# SET THE PARAMETERS OF PLASMA AND LASER
# Plasma
mu_Cu = 64  # Atomic mass of copper
f_ik = 0.00328  # Oscillator strength
g_i = 6  # Statistical weight for the lower energy level
E_i = 1.64  # Energy of level in eV
E_k = 3.82  # Energy of upper level in eV
# Laser
lambda_m = 578.2*nm  # Wavelength in meters

# mu_Cu = 64  # Atomic mass of copper
# f_ik = 0.00328  # Oscillator strength
# g_i = 6  # Statistical weight for the lower energy level
# E_i = 1.38  # Energy of level in eV
# E_k = 3.82  # Energy of upper level in eV
# # Laser
# lambda_m = 510.5*nm  # Wavelength in meters


# Folders and profiles
foldername = 'plots_and_results' 
filename_statsum = 'Statsum_CuI.txt' # Statistic sum
filename_temperature = 'temperature_profile_3436.txt' # Temperature profile
filename_OES_results = 'oes_results_3428.txt' # Temperature profile

filepath_statsum = os.path.join(foldername, filename_statsum)
filepath_temperature = os.path.join(foldername, filename_temperature)
filepath_OES_results = os.path.join(foldername, filename_OES_results)

y_crssctn = y_min_electrode + (y_max_electrode - y_min_electrode) // 2 # DEVINE Y CROSSECTION
y_crssctn_gt = y_min_electrode + (y_max_electrode - y_min_electrode) // 2 # DEVINE Y CROSSECTION
y_crssctn_gt = 1869
y_crssctn = 2046

#FLAGS
save_output_to_txt = True
filepath_save_results_txt = os.path.join(foldername, "results_of_plotting.txt")
save_message = ""

# Instantiate the class of image preprocess
preprocess_img_absorption = ImagePreprocess(
    filepath_img_absorption, x_min_electrode, x_max_electrode, y_min_electrode, y_max_electrode,
y_crssctn= y_crssctn)

preprocess_img_gt = ImagePreprocess(
    filepath_img_gt, x_min_electrode, x_max_electrode, y_min_electrode, y_max_electrode,y_crssctn=y_crssctn_gt)

# Find intensity crossection for ROI and y_crssctn
intensity_crssctn = preprocess_img_absorption.extract_intensity_row(
    x_min=x_minROI,x_max= x_maxROI, y_crssctn=None
)
intensity_crssctn_smoothed = savgol_filter(intensity_crssctn, window_length=51, polyorder=2)  # window_length and polyorder can be adjusted


# DISPLAY IMAGE WITH RECTANGULAR FRAME AND EDGE-DETECTED IMAGE
fig, [ax1, ax2] = plt.subplots(1, 2)
ax1.imshow(
    preprocess_img_absorption.color_rectangle(
        image_not_cropped=preprocess_img_absorption.grayscale_image, y_crssctn=y_crssctn
    ), cmap = 'gray'
) # Display grayscale image with rectange
ax1.set_title("Grayscale image")
ax1.set_xlabel("x [pxl]")
ax1.set_ylabel("y [pxl]")

ax2.imshow(
    preprocess_img_absorption.color_rectangle(
        image_not_cropped=preprocess_img_absorption.edge_detection(), y_crssctn=y_crssctn
    ),
    cmap="gray",
) # Display edge-detected image
ax2.set_title("Edge-detected image")
ax2.set_xlabel("x [pxl]")
ax2.set_ylabel("y [pxl]")

# # DISPLAY IMAGE background and absorbtion
# fig, [ax1, ax2] = plt.subplots(1, 2)
# ax1.imshow(
#     preprocess_img_absorption.color_rectangle(
#         image_not_cropped=preprocess_img_absorption.grayscale_image, y_crssctn=y_crssctn
#     ), cmap = 'gray'
# ) # Display grayscale image with rectange
# ax1.set_title("Grayscale image")
# ax1.set_xlabel("x [pxl]")
# ax1.set_ylabel("y [pxl]")

# ax2.imshow(
#     preprocess_img_absorption.color_rectangle(
#         image_not_cropped=preprocess_img_gt.grayscale_image, y_crssctn=y_crssctn_gt
#     ),
#     cmap="gray",
# ) # Display edge-detected image
# ax2.set_title("Background image")
# ax2.set_xlabel("x [pxl]")
# ax2.set_ylabel("y [pxl]")

# DISPLAY GT AND ABSORBTION IMAGE
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(16, 9))
ax1.imshow(
    preprocess_img_gt.color_rectangle(
        image_not_cropped=preprocess_img_gt.grayscale_image, y_crssctn=y_crssctn_gt
    ), cmap = 'gray'
) # Display grayscale image with rectange
ax1.set_title("Edge-detected image ground truth")
ax1.set_xlabel("x [pxl]")
ax1.set_ylabel("y [pxl]")

ax2.imshow(
    preprocess_img_absorption.color_rectangle(
        image_not_cropped=preprocess_img_absorption.grayscale_image, y_crssctn=y_crssctn
    ),
    cmap="gray",
) # Display edge-detected image
ax2.set_title("Grayscale image of absorption")
ax2.set_xlabel("x [pxl]")
ax2.set_ylabel("y [pxl]")
plt.savefig(os.path.join(foldername_savefig, 'img_gt_abs.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)



# DISPLAY CROSSECTION ROI AND FULL of GT image and absorbtion
fig, [ax1, ax2] = plt.subplots(1,2)
ax1.plot(np.arange(x_minROI,x_maxROI),intensity_crssctn, label = 'Initial intensity')
ax1.plot(np.arange(x_minROI,x_maxROI),preprocess_img_gt.extract_intensity_row(
    x_min=x_minROI,x_max= x_maxROI, y_crssctn=None), label = 'Initial intensity')

ax1.plot(np.arange(x_minROI,x_maxROI),intensity_crssctn_smoothed, label = 'Smoothed intensity')
ax1.set_title(f"Cropped towards ROI {x_minROI}x{x_maxROI}")
ax1.set_xlabel("x [pxl]")
ax1.set_ylabel("Intensity")
ax1.legend()

ax2.plot(preprocess_img_absorption.extract_intensity_row(
    x_min=0,x_max= 6000, y_crssctn=None
), label = 'full row absorption')
ax2.plot(preprocess_img_gt.extract_intensity_row(
    x_min=0,x_max= 6000, y_crssctn=None
), label = 'full row background')
ax2.set_title("Full width")
ax2.set_xlabel("x [pxl]")
ax2.set_ylabel("Intensity")
ax2.legend()


# Create a new datarray, where data from chosen previous range would be used
intensity_row_absorbtion = preprocess_img_absorption.extract_intensity_row(x_min=x_min, x_max=x_max,  y_crssctn = y_crssctn)
intensity_row_gt = preprocess_img_gt.extract_intensity_row(x_min=x_min, x_max=x_max,  y_crssctn = y_crssctn_gt)
analysis = ParametricAnalysis(x_min=x_min, x_max=x_max, dpxl_m=preprocess_img_absorption.dpxl_m)
print(f"d_pxl = {analysis.dpxl_m}")

# split_num_left = 2000
# split_num_right = 3100

# # Split data into ranges
# x1 = analysis.x_pxl[ (analysis.x_pxl < split_num_left)]
# y1 = intensity_row_absorbtion[(analysis.x_pxl < split_num_left)]

# x2 = analysis.x_pxl[(analysis.x_pxl >= split_num_left) & (analysis.x_pxl < split_num_right)]
# y2 = intensity_row_absorbtion[(analysis.x_pxl >= split_num_left) & (analysis.x_pxl < split_num_right)]
# x3 = analysis.x_pxl[(analysis.x_pxl >= split_num_right) ]
# y3 = intensity_row_absorbtion[(analysis.x_pxl >= split_num_right) ]

# # Fit quadratic functions for each range
# coef1 = np.polyfit(x1, y1, 2)
# coef2 = np.polyfit(x2, y2, 2)
# coef3 = np.polyfit(x3, y3, 2)

# quadratic1 = np.poly1d(coef1)
# quadratic2 = np.poly1d(coef2)
# quadratic3 = np.poly1d(coef3)

# # Generate fitted values
# y1_fitted = quadratic1(x1)
# y2_fitted = quadratic2(x2)
# y3_fitted = quadratic3(x3)

# x_combined = np.concatenate([x1, x2, x3])

# # Combine y arrays
# y_combined = np.concatenate([y1_fitted, y2_fitted, y3_fitted])

# # Sort the combined arrays by x to maintain order
# sorted_indices = np.argsort(x_combined)
# x_combined = x_combined[sorted_indices]
# y_combined = y_combined[sorted_indices]

# y_fitted = y_combined


# Fit the intensity profile with a quadratic function
coef, quadratic_func = analysis.fit_quadratic(analysis.x_pxl, intensity_row_absorbtion)
coef_0, quadratic_func_0 = analysis.fit_quadratic(analysis.x_pxl, intensity_row_gt)

# # Generate the fitted y values
y_fitted = quadratic_func(analysis.x_pxl)
y_fitted_0 = quadratic_func_0(analysis.x_pxl)


# Plot the intensity profiles and fitted curves
colorsBlue = plt.cm.Blues(np.linspace(0.55, 0.95, 20))  # Light to dark blue
colorsOrange = plt.cm.Oranges(np.linspace(0.45, 0.85, 20))
plt.figure()
plt.plot(analysis.x_pxl, intensity_row_gt, color=colorsBlue[5], label=r"Reference intensity, $I_{ref}$")
plt.plot(analysis.x_pxl, y_fitted_0, '--', color=colorsOrange[5], label=r"Result of fit $I_{ref\,fit}$")
plt.plot(analysis.x_pxl, intensity_row_absorbtion, color=colorsBlue[15], label=r"Intensity in region of absorption, $I_{abs}$")
plt.plot(analysis.x_pxl, y_fitted, '--', color=colorsOrange[15], label=r"Result of fit $I_{abs\,fit}$")
plt.xlabel("Column Index, x [pxl]")
plt.ylabel("Intensity")
plt.title("Intensity Profile and Quadratic Fit")
plt.legend(loc='upper right')
plt.savefig(os.path.join(foldername_savefig, 'intensity_profile.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)


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

# Chose which side to analyze: left or right
x_for_analysis_m = x_real_m_positive
tau_for_analysis_1_cm = tau_positive

# Compute the inverse Abel transform using integral and summation methods
r0 = max(x_real_m)
N = 50
dr = r0 / N
radius = np.linspace(0, r0, N)
radius_for_integration = radius[1:]
integrate_result = np.empty_like(radius_for_integration)
integrate_error = np.empty_like(radius_for_integration)
tau_prime = analysis.manual_gradient(tau_positive, x_real_m_positive)
#tau_prime = -analysis.manual_gradient(tau_negative, x_real_m_negative)

# Loop over the radius array and compute Abel integral
for i, r in enumerate(radius_for_integration):
    integrate_result[i] , integrate_error [i]= analysis.compute_integral(r=r, r0=r0, tau_prime=tau_prime, x_values=x_real_m_positive)

# # Fit the intensity profile with a quadratic function
# coef_integral, quadratic_func_integral = analysis.fit_quadratic(radius, integrate_result)
# # Generate the fitted y values
# integral_fitted = quadratic_func_integral(radius)


valid_indices = ~np.isnan(integrate_result)
radius_for_integration = radius_for_integration[valid_indices]
integrate_result = integrate_result[valid_indices]
integrate_error = integrate_error[valid_indices]  # If needed for error bars

# Assuming radius, integrate_result, integrate_error, and analysis are already defined
# Fit the integrate_result with a quadratic function
coef_integral, quadratic_func_integral = analysis.fit_quadratic(radius_for_integration, integrate_result)

# Generate fitted values using the quadratic function
fitted_values_kappa = quadratic_func_integral(radius)
# # Calculate residuals
# residuals = integrate_result - fitted_values
# # Compute MSE
# mse = np.mean(residuals ** 2)
# # Compute RMSE
# rmse = np.sqrt(mse)
# print(integrate_result)
# print(fitted_values)


# Plot the original integrate_result with error bars
plt.figure()
plt.errorbar(radius_for_integration* 1e3, integrate_result* 1e-2, yerr=integrate_error*1e-2, fmt='o', label='Integrate Result', capsize=5)
# Plot the fitted quadratic function
plt.plot(radius* 1e3, fitted_values_kappa* 1e-2, label='Fitted Quadratic', linestyle='--', color='red')

# Add labels, legend, and grid
plt.xlabel("r [mm]")
plt.ylabel(r"$\kappa_{0}\;[1/cm]$")
plt.xlim(left=0)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(foldername_savefig, 'kappa_fit.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)

# Initialize class
concentration_calculator = ConcentrationCalculator(lambda_m, mu_Cu, f_ik, g_i, E_i, E_k)

# Load temperature profile
data_temp = concentration_calculator.read_txt_to_array(filepath_temperature)
t_profile_K = data_temp[:, 1]
d_t_profile = data_temp[:,2]


r_t_K = data_temp[:, 0]*1e-3

filtered_kappa = []
for radius_T_K in r_t_K:
    idx = np.abs(radius - radius_T_K).argmin()  # Find the index of the closest x_1 value
    filtered_kappa.append(fitted_values_kappa[idx])  # Append the corresponding y_1 value

filtered_y_1 = np.array(filtered_kappa)

# Example data for kappa and radius from previous computation
# kappa_profile = fitted_values_kappa  
# r_kappa = radius  
kappa_profile = filtered_kappa
r_kappa = r_t_K

# kappa_profile = integrate_result  
# r_kappa = radius_for_integration  

# Interpolate the temperature profile to match the length of kappa_profile
t_profile_interpolated = concentration_calculator.interpolate_temperature_profile(r_kappa, r_t_K, t_profile_K)
# d_t_profile = np.ones_like(t_profile_interpolated)*350

# Calculate concentration
n_values, n_i, d_lambda_m  = concentration_calculator.calculate_concentration(r_kappa, t_profile_interpolated, kappa_profile, filepath_statsum)
dn_values = n_values*E_i*eV*d_t_profile/t_profile_interpolated**2/1.38e-23
print(dn_values[3])
# Compute Abel integral using `compute_integral` method for a single r
r_single = radius[0]
integral_result = analysis.compute_integral(r=r_single, r0=r0, tau_prime=tau_prime, x_values=x_real_m_positive)
print(f"Abel integral at r={r_single}: {integral_result}")

# Load OES results profile
data_OES = concentration_calculator.read_txt_to_array(filepath_OES_results)
OES_profile_n = data_OES[:, 1]
r_OES_m = data_OES[:, 0]*1e-3

# Save concentration and population number density to .txt
if save_output_to_txt:
    # Write the data to a .txt file
    with open(filepath_save_results_txt, "a") as file:
        # Write basic information
        file.write(f"image absorption: {filename_img_absorption}\n")
        file.write(f"image background: {filename_img_gt}\n")
        file.write(f"Message: {save_message}\n")
        file.write(f"ROI x: {x_min} - {x_max}\n")
        file.write(f"y_crssctn = {y_crssctn}\n")
        file.write(f"lambda = {lambda_m}\n")

        # Define a helper function for structured data writing
        def write_section(header, data, scale_factors):
            file.write(35*"*" + "\n")
            file.write(header + "\n")
            for values in zip(*data):
                scaled_values = [value * scale for value, scale in zip(values, scale_factors)]
                file.write("\t".join(map(str, scaled_values)) + "\n")

        # Write sections
        write_section("r,mm\ttau", [r_kappa, tau_for_analysis_1_cm], [1e3, 1])
        write_section("r,mm\tkappa, 1/cm", [r_kappa, kappa_profile], [1e3, 1])
        write_section("r,mm\td_lambda, nm", [r_kappa, d_lambda_m], [1e3, 1e9])
        write_section("r,mm\tn_i, SI", [r_kappa, n_i], [1e3, 1e6])
        write_section("r,mm\tn, m^-3\tdn_values, m^-3", [r_kappa, n_values, dn_values], [1e3, 1e6, 1e6])

        file.write("\n\n\n")

    print("Data saved to plot_data.txt")


#ALL FOR PLOTTING
# Plot tau
plt.figure()
plt.plot(x_real_m, tau)
# plt.plot(x_real_m_negative, tau_negative, 'o')
plt.plot(x_real_m_positive, tau_positive, 'o')
plt.xlabel("x [m]")
plt.ylabel(r"$\tau$")

# Set up plot for tau(r)
fig3, ax3 = plt.subplots()
ax3.plot(x_for_analysis_m * 1e3, tau_for_analysis_1_cm)  # Convert x to mm
ax3.set_xlabel("x [mm]")
ax3.set_ylabel(r"$\tau_{0}$")
ax3.set_xlim(left=0)
ax3.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax3.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
plt.savefig(os.path.join(foldername_savefig, 'tau_r.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)


# Plot inverse Abel transform results
fig2, ax2 = plt.subplots()
ax2.scatter(radius_for_integration * 1e3, integrate_result * 1e-2)  # Convert r to mm and results to 1/cm
ax2.set_title("Inverse Abel Transform")
ax2.set_xlabel("r [mm]")
ax2.set_ylabel(r"$\kappa_{0}\;[1/cm]$")
ax2.set_xlim(0, r0 * 1e3)
plt.savefig(os.path.join(foldername_savefig, 'kappa_r.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)


#plot temperature profile
fig, ax = plt.subplots()
ax.errorbar(r_t_K/mm, t_profile_K, yerr=d_t_profile, fmt='o', label='Input Temperature Profile', capsize=3)
ax.plot(r_kappa/mm, t_profile_interpolated, 'o', alpha=0.5, label='Interpolated Temperature Profile')
ax.set_xlabel("Radius [mm]")
ax.set_ylabel("Temperature [K]")
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
plt.legend()
plt.savefig(os.path.join(foldername_savefig, 'temp_profile.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)


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
plt.savefig(os.path.join(foldername_savefig, 'doplers_broadening.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)


# Population number density
plt.figure()
plt.plot(r_kappa * 1e3, n_i*1e6, "o", alpha=0.5, label='Concentration')
plt.xlabel("Radius [mm]")
plt.ylabel(r"Population number density $n_{i}$ [m$^{-3}$]")
plt.yscale("log")
plt.ylim(5e18,1e21)
plt.savefig(os.path.join(foldername_savefig, 'p_number_density.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)


# Plot number density (m)
plt.figure()
plt.plot(r_kappa * 1e3, n_values*1e6, "o", alpha=0.5, label='Concentration')
plt.errorbar(r_kappa * 1e3, n_values*1e6, color = 'black', yerr=dn_values*1e6, fmt='o', label='Integrate Result', capsize=3)
plt.xlabel("Radius [mm]")
plt.ylabel(r"Number density $n$ [m$^{-3}$]")
plt.yscale("log")
plt.ylim(5e19,5e21)
plt.savefig(os.path.join(foldername_savefig, 'number_density.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)


# Plot number density (m)
plt.figure()
plt.plot(r_kappa * 1e3, n_values*1e6 , "o-", alpha=0.5, label='Concentration')
plt.scatter(r_OES_m*1e3,OES_profile_n, label='OES results')
plt.xlabel("Radius [mm]")
plt.ylabel(r"Number density $n$ [cm$^{-3}$]")

plt.legend()

plt.show(block=False)
input("Press any key to close all plots...")
plt.close('all')