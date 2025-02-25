import os
import matplotlib.pyplot as plt
from units_constants import *
from utilites import *
from PLASMA_PARAMETERS import *

# Import the classes
from image_preprocess import ImagePreprocess
from intensity_analysis import IntensityAnalysis
from optical_param_analysis import OpticalParamAnalysis
from concentration_calculator import PlasmaValuesCalculator

from plotting_def import *
from utilites import *

"""
Ця версія ще буде зберігати в .тхт отримані значення заселеностей та концентрацій 
"""

# Load parameters from text file
config_file = "Input_file.txt"  # Path to config file
config = load_config(config_file)

# Assign values to variables
x_minROI = config["x_minROI"]
x_maxROI = config["x_maxROI"]

wavelength_flag_G_510nm = config["wavelength_flag_G_510nm"]
wavelength_flag_Y_578nm = config["wavelength_flag_Y_578nm"]

foldername_img = config["foldername_img"]
filename_img_absorption = config["filename_img_absorption"]
filename_img_gt = config["filename_img_gt"]
foldername_savefig = config["foldername_savefig"]

foldername = config["foldername"]
filename_statsum = config["filename_statsum"]
filename_temperature = config["filename_temperature"]
filename_OES_results = config["filename_OES_results"]

filepath_img_absorption = os.path.join(foldername_img, filename_img_absorption)
filepath_img_gt = os.path.join(foldername_img, filename_img_gt)
filepath_statsum = os.path.join(foldername, filename_statsum)
filepath_temperature = os.path.join(foldername, filename_temperature)
filepath_OES_results = os.path.join(foldername, filename_OES_results)


# MODIFY file FLAGS
save_output_to_txt = config["save_output_to_txt"]
filepath_save_results_txt = config["filepath_save_results_txt"]
save_message = config["save_message"]

y_crssctn_absorbtion = config["y_crssctn_absorbtion"]
y_crssctn_gt = config["y_crssctn_gt"]

# SET THE PARAMETERS OF PLASMA AND LASER
image_parameters = {
    "x_min_electrode": config["x_min_electrode"],  # Left limit of the electrode
    "x_max_electrode": config["x_max_electrode"],  # Right limit of the electrode
    "y_min_electrode": config["y_min_electrode"],  # Lower limit of the electrode
    "y_max_electrode": config["y_max_electrode"],  # Upper limit of the electrode
    "region_size": config["region_size"],  # defines size of square (number of pixels)
}

right_side_pick_flag = config["right_side_pick_flag"]
number_of_points_for_integration = config["number_of_points_for_integration"]


# PLASMA parameters. Will be chosen automatically accordingly to your lambda_laser flag
plasma_parameters = (
    plasma_parameters_G_510nm if wavelength_flag_G_510nm else plasma_parameters_Y_578nm
)


# Instantiate the ImagePreprocess class to read image
image_absorption = ImagePreprocess(
    filepath=filepath_img_absorption,
    image_parameters=image_parameters,
    y_crssctn=y_crssctn_absorbtion,
)
image_absorption.read_image_based_on_extension()
analyse_intensity_abs = IntensityAnalysis(preprocessor_object=image_absorption)
x_pxl_abs, intensity_abs = analyse_intensity_abs.extract_intensity_from_region(
    y_crssctn=y_crssctn_absorbtion
)
x_pxl_abs_ROI, intensity_abs_ROI = analyse_intensity_abs.extract_intensity_from_region(
    x_min_ROI=x_minROI, x_max_ROI=x_maxROI, y_crssctn=y_crssctn_absorbtion
)


# Instantiate the ImagePreprocess class for the ground truth image
image_gt = ImagePreprocess(
    filepath=filepath_img_gt, image_parameters=image_parameters, y_crssctn=y_crssctn_gt
)
image_gt.read_image_based_on_extension()
intensity_analysis_gt = IntensityAnalysis(preprocessor_object=image_gt)

# Extract intensity from full width
x_pxl_gt, intensity_gt = intensity_analysis_gt.extract_intensity_from_region(
    y_crssctn=y_crssctn_gt
)
# Extract intensity from ROI
x_pxl_gt_ROI, intensity_gt_ROI = intensity_analysis_gt.extract_intensity_from_region(
    x_min_ROI=x_minROI, x_max_ROI=x_maxROI, y_crssctn=y_crssctn_gt
)

# Fit extracted ROI intensity with a squared function
intensity_abs_ROI_square_fit = apply_square_fit_to_function(
    x_array=x_pxl_abs_ROI, y_array=intensity_abs_ROI
)
intensity_gt_ROI_square_fit = apply_square_fit_to_function(
    x_array=x_pxl_gt_ROI, y_array=intensity_gt_ROI
)

# Transform x-array from pixels to meters
x_m_abs = analyse_intensity_abs.x_array_rescale_to_m(
    x_array_pxl=x_pxl_abs,
    x_array_pxl_ROI=x_pxl_abs_ROI,
    intensity_fit_ROI=intensity_abs_ROI_square_fit,
)
x_m_abs_ROI = analyse_intensity_abs.x_array_rescale_to_m(
    x_array_pxl_ROI=x_pxl_abs_ROI,
    x_array_pxl=x_pxl_abs_ROI,
    intensity_fit_ROI=intensity_abs_ROI_square_fit,
)

# Compute tau (optical thickness)
optical_analysis_from_inensity_sq_fit = OpticalParamAnalysis(
    x_array_m=x_m_abs_ROI,
    i_probe=intensity_gt_ROI_square_fit,
    i_absorption=intensity_abs_ROI_square_fit,
)
optical_analysis_from_intensity_scatter = OpticalParamAnalysis(
    x_array_m=x_m_abs_ROI, i_probe=intensity_gt_ROI, i_absorption=intensity_abs_ROI
)  # tau from scatter is computed for comparison. For calculation tau fitted is taken into account

# Compute optical thickness (tau)
tau_ROI = optical_analysis_from_inensity_sq_fit.compute_tau()
tau_ROI_point = optical_analysis_from_intensity_scatter.compute_tau()

radius_x_m, tau_radius = optical_analysis_from_inensity_sq_fit.analysis_side_picker(
    tau_array=tau_ROI, radius_array_m=x_m_abs_ROI, right_side=right_side_pick_flag
)  # Chooses which side to analyze. Transforms x into radius

# Computes derivative
tau_radius_prime = optical_analysis_from_inensity_sq_fit.tau_derivative(
    radius=radius_x_m, tau=tau_radius
)

# Abel's transform. The result of integration is kappa [1/cm]. Decreases number of points in radius for integration precision
radius_for_integration, kappa_1_cm, integrate_error = (
    optical_analysis_from_inensity_sq_fit.integrate_Abel(
        number_points=number_of_points_for_integration,
        radius_m=radius_x_m,
        tau_prime=tau_radius_prime,
    )
)
kappa_1_cm_sq_fit = apply_square_fit_to_function(radius_for_integration, kappa_1_cm)

compute_plasma_param = PlasmaValuesCalculator(
    plasma_parameters=plasma_parameters_G_510nm
)

# Load temperature profile
t_K, d_T_K, r_t_K = compute_plasma_param.read_t_K(filepath_t_K=filepath_temperature)
kappa_intepolated = interpolate_function(
    r_t_K, radius_for_integration, kappa_1_cm_sq_fit
)  # Interpolate kappa to corresond to real temperature profile

# Compute plasma parameters
''' 
n - Cooper number density
dn - - Cooper number density error
n_i - population number density
lambda_broadening_Doplers - Doplers' mechanism of broadening
'''
n, dn, n_i, lambda_broadening_Doplers = compute_plasma_param.calculate_plasma_parameters(
    radius=r_t_K,
    T_K=t_K,
    d_T_K=d_T_K,
    kappa_profile=kappa_intepolated,
    filepath_statsum=filepath_statsum,
)

# DISPLAY IMAGE WITH RECTANGULAR FRAME AND EDGE-DETECTED IMAGE
plot_image_and_edge_detection(
    image_left=image_absorption.draw_rectangle_with_overlay(
        bgr_image=image_absorption.grayscale_image
    ),
    image_right_edge_detection=image_absorption.draw_rectangle_with_overlay(
        bgr_image=image_absorption.edge_detection()
    ),
)

#Plotting

# DISPLAY GT AND ABSORBTION IMAGE
plot_absorption_gt_image(
    image_absorption=image_absorption.draw_rectangle_with_overlay(
        bgr_image=image_absorption.grayscale_image
    ),
    image_gt=image_gt.draw_rectangle_with_overlay(bgr_image=image_gt.grayscale_image),
)
# plt.savefig(os.path.join(foldername_savefig, 'img_gt_abs.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)

plot_region_intensity(
    x=x_pxl_abs,
    intensity=intensity_abs,
    x_ROI=x_pxl_abs_ROI,
    intensity_ROI=intensity_abs_ROI,
    region_size=image_parameters.get("region_size"),
)

plot_region_intensity_abs_gt(
    x_abs=x_pxl_abs,
    intensity_abs=intensity_abs,
    x_ROI_abs=x_pxl_abs_ROI,
    intensity_ROI_abs=intensity_abs_ROI,
    x_gt=x_pxl_gt,
    intensity_gt=intensity_gt,
    x_ROI_gt=x_pxl_gt_ROI,
    intensity_ROI_gt=intensity_gt_ROI,
    region_size=image_parameters.get("region_size"),
)

# Plot crossection in the selected ROI and ax2 that will plot full row intensity pattern
plt.figure()
plt.scatter(x_pxl_abs, intensity_abs)
plt.scatter(x_pxl_gt, intensity_gt)
plt.title("Region intensity")
# прописати  так, аби абс було кольорове або чорне, а опорне було сірим

plt.figure()
plt.scatter(x_pxl_abs_ROI, intensity_abs_ROI)
plt.scatter(x_pxl_gt_ROI, intensity_gt_ROI)
plt.title("Region intensity ROI")


plt.figure()
plt.scatter(x_pxl_abs_ROI, intensity_abs_ROI)
plt.scatter(x_pxl_gt_ROI, intensity_gt_ROI)
plt.plot(x_pxl_abs_ROI, intensity_abs_ROI_square_fit)
plt.plot(x_pxl_gt_ROI, intensity_gt_ROI_square_fit)
plt.title("Region intensity ROI squared fit")


plt.figure()
plt.scatter(x_m_abs_ROI, intensity_abs_ROI)
plt.scatter(x_m_abs, intensity_abs)
plt.title("Region intensity ROI and full row with x in meters")

plt.figure()
plt.scatter(x_pxl_abs_ROI, intensity_abs_ROI)
plt.scatter(x_pxl_abs, intensity_abs)
plt.title("Region intensity ROI and full row with x in meters")


plt.figure()
plt.scatter(x_m_abs_ROI, tau_ROI, label="tau square fit")
plt.scatter(x_m_abs_ROI, tau_ROI_point, label="tau scatter")
plt.title("Region intensity ROI and full row with x in meters")

plt.figure()
plt.title("Tau with a picked side")
plt.scatter(radius_x_m, tau_radius)


# Plot the original integrate_result with error bars
plt.figure()
plt.errorbar(
    radius_for_integration * 1e3,
    kappa_1_cm * 1e-2,
    yerr=integrate_error * 1e-2,
    fmt="o",
    label="Integrate Result",
    capsize=5,
)
# Plot the fitted quadratic function
plt.plot(
    radius_for_integration * 1e3,
    kappa_1_cm_sq_fit * 1e-2,
    label="Fitted Quadratic",
    linestyle="--",
    color="red",
)


plt.figure()
plt.scatter(r_t_K, kappa_intepolated, label="interpolated to the radius T_k")
plt.scatter(radius_for_integration, kappa_1_cm_sq_fit)




# Plot Doplers broadening
plt.figure()
plt.plot(r_t_K * 1e3, lambda_broadening_Doplers * 1e9, "o", alpha=0.5)
plt.xlabel("Radius [mm]")
plt.ylabel(r"Doplers broadening, $\Delta \lambda_{D}$ [$nm$]")
plt.legend()
# ax = plt.gca()
# ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
# ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# plt.savefig(os.path.join(foldername_savefig, 'doplers_broadening.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)


# Population number density
plt.figure()
plt.plot(r_t_K * 1e3, n_i * 1e6, "o", alpha=0.5, label="Concentration")
plt.xlabel("Radius [mm]")
plt.ylabel(r"Population number density $n_{i}$ [m$^{-3}$]")
plt.yscale("log")
plt.ylim(5e18, 1e21)
# plt.savefig(os.path.join(foldername_savefig, 'p_number_density.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)


# Plot number density (m)
plt.figure()
plt.plot(r_t_K * 1e3, n * 1e6, "o", alpha=0.5, label="Concentration")
plt.errorbar(
    r_t_K * 1e3,
    n * 1e6,
    color="black",
    yerr=dn * 1e6,
    fmt="o",
    label="Integrate Result",
    capsize=3,
)
plt.xlabel("Radius [mm]")
plt.ylabel(r"Number density $n$ [m$^{-3}$]")
plt.yscale("log")
plt.ylim(5e19, 5e21)

# plt.savefig(os.path.join(foldername_savefig, 'number_density.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.legend()

plt.show(block=False)
input("Press any key to close all plots...")
plt.close("all")
