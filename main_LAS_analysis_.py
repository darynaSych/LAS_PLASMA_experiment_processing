import matplotlib.pyplot as plt

# Import the classes
from image_preprocess import ImagePreprocess
from intensity_analysis import IntensityAnalysis
from optical_param_analysis import OpticalParamAnalysis
from lib_conc_calculator import concentration_calculator
from config_loader import initialize_config

from plotting_def import *
from utilites import *
from save_txt import *

import sys

config_file = sys.argv[1]
plasma_parameters_file = "plasma_parameters.json"

param = initialize_config(config_file=config_file, plasma_parameters=plasma_parameters_file)

# Instantiate the ImagePreprocess class to read image
image_absorption = ImagePreprocess(
    filepath=param["filepath_img_absorption"],
    image_parameters=param["image_parameters"],
    y_crssctn=param["y_crssctn_absorbtion"],
)
image_absorption.read_image_based_on_extension()
analyse_intensity_abs = IntensityAnalysis(preprocessor_object=image_absorption)
x_pxl_abs, intensity_abs = analyse_intensity_abs.extract_intensity_from_region(
    y_crssctn=param["y_crssctn_absorbtion"]
)
x_pxl_abs_ROI, intensity_abs_ROI = analyse_intensity_abs.extract_intensity_from_region(
    x_min_ROI=param["x_minROI"],
    x_max_ROI=param["x_maxROI"],
    y_crssctn=param["y_crssctn_absorbtion"],
)

# Instantiate the ImagePreprocess class for the ground truth image
image_gt = ImagePreprocess(
    filepath=param["filepath_img_gt"],
    image_parameters=param["image_parameters"],
    y_crssctn=param["y_crssctn_gt"],
)
image_gt.read_image_based_on_extension()
intensity_analysis_gt = IntensityAnalysis(preprocessor_object=image_gt)

# Extract intensity from full width
x_pxl_gt, intensity_gt = intensity_analysis_gt.extract_intensity_from_region(
    y_crssctn=param["y_crssctn_gt"]
)
# Extract intensity from ROI
x_pxl_gt_ROI, intensity_gt_ROI = intensity_analysis_gt.extract_intensity_from_region(
    x_min_ROI=param["x_minROI"],
    x_max_ROI=param["x_maxROI"],
    y_crssctn=param["y_crssctn_gt"],
)

# Transform x-array from pixels to meters
x_m_abs = analyse_intensity_abs.x_array_rescale_to_m(
    x_array_pxl=x_pxl_abs,
    x_array_pxl_ROI=x_pxl_abs_ROI,
    intensity_fit_ROI=intensity_abs_ROI,
)
x_m_abs_ROI = analyse_intensity_abs.x_array_rescale_to_m(
    x_array_pxl_ROI=x_pxl_abs_ROI,
    x_array_pxl=x_pxl_abs_ROI,
    intensity_fit_ROI=intensity_abs_ROI,
)

# Compute tau (optical thickness)
optical_analysis_from_intensity_scatter = OpticalParamAnalysis(
    x_array_m=x_m_abs_ROI,
    intensity_probe=intensity_gt_ROI,
    intensity_absorption=intensity_abs_ROI,
)
tau_ROI_point = optical_analysis_from_intensity_scatter.compute_tau()

# Apply square fit to tau_ROI_point (fix was here)
tau_ROI, tau_ROI_error = apply_square_fit_to_function(
    x_array=x_m_abs_ROI,
    y_array=tau_ROI_point
)

radius_x_m, tau_radius = optical_analysis_from_intensity_scatter.pick_side_of_analysis(
    tau_array=tau_ROI,
    radius_array_m=x_m_abs_ROI,
    right_side=param["right_side_pick_flag"],
)

# Computes derivative
tau_radius_prime = optical_analysis_from_intensity_scatter.tau_derivative(
    radius=radius_x_m, tau=tau_radius
)

# Abel's transform
radius_for_integration, kappa_1_m, integrate_error = (
    optical_analysis_from_intensity_scatter.integrate_Abel(
        number_points=param["number_of_points_for_integration"],
        radius_m=radius_x_m,
        tau_prime=tau_radius_prime,
    )
)
kappa_1_m_sq_fit, kappa_1_m_sq_fit_error = apply_square_fit_to_function(
    radius_for_integration, kappa_1_m
)

compute_plasma_param = concentration_calculator.PlasmaValuesCalculator(
    plasma_parameters=param["plasma_parameters"]
)

# Load temperature profile
t_K, d_T_K, r_t_K = compute_plasma_param.read_t_K(
    filepath_t_K=param["filepath_temperature"]
)
kappa_intepolated_1_m = interpolate_function(
    r_t_K, radius_for_integration, kappa_1_m_sq_fit
)

# Compute plasma parameters
n_m_3, dn_m_3, n_i_m_3, d_lambda_Doplers_m = (
    compute_plasma_param.calculate_plasma_parameters(
        radius=r_t_K,
        T_K=t_K,
        d_T_K=d_T_K,
        kappa_profile_1_m=kappa_intepolated_1_m,
        filepath_statsum=param["filepath_statsum"],
    )
)

# SAVE .TXT
save_output_to_txt = param["save_output_to_txt"]
if save_output_to_txt:
    arrays = [r_t_K, n_m_3, dn_m_3, n_i_m_3]
    column_names = ["r [m]", "n [m^-3]", "dn [m^-3]", "n_i [m^-3]"]
    save_arrays_to_txt(param["filepath_save_results_txt"], arrays, column_names)

    y_position_m = (param["y_crssctn_gt"] - param["image_parameters"].get("y_min_electrode")) * image_gt.dpxl_m
    with open(param["filepath_save_results_txt"], "a") as f:
        f.write(f"# y_crssctn_gt [mm]:\t{y_position_m*1e3:.6e}\n")

# PLOTTING SECTION
plot_image_and_edge_detection(
    image_left=image_absorption.draw_rectangle_with_overlay(
        bgr_image=image_absorption.grayscale_image
    ),
    image_right_edge_detection=image_absorption.draw_rectangle_with_overlay(
        bgr_image=image_absorption.edge_detection()
    ),
)

plot_absorption_gt_image(
    image_absorption=image_absorption.draw_rectangle_with_overlay(
        bgr_image=image_absorption.grayscale_image
    ),
    image_gt=image_gt.draw_rectangle_with_overlay(bgr_image=image_gt.grayscale_image),
)

plot_region_intensity(
    x=x_pxl_abs,
    intensity=intensity_abs,
    x_ROI=x_pxl_abs_ROI,
    intensity_ROI=intensity_abs_ROI,
    region_size=param["image_parameters"].get("region_size"),
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
    region_size=param["image_parameters"].get("region_size"),
)

plt_ROI_intensity_m(
    x_m_abs_ROI=x_m_abs_ROI,
    intensity_abs_ROI=intensity_abs_ROI,
    intensity_gt_ROI=intensity_gt_ROI,
)

plot_optical_thickness(
    x_m_abs_ROI=x_m_abs_ROI,
    tau_ROI=tau_ROI,
    tau_ROI_point=tau_ROI_point,
    radius_x_m=radius_x_m,
    tau_radius=tau_radius,
    side_of_analysis=param["right_side_pick_flag"],
)

plot_absorption_coefficient(
    radius_for_integration=radius_for_integration,
    kappa_1_cm=0.01 * kappa_1_m,
    integrate_error=integrate_error,
    kappa_1_cm_sq_fit=0.01 * kappa_1_m_sq_fit,
)

plot_T_K_and_interpolated_kappa(
    r_t_K=r_t_K,
    t_K=t_K,
    d_T_K=d_T_K,
    kappa_intepolated=kappa_intepolated_1_m * 0.01,
    radius_for_integration=radius_for_integration,
    kappa_1_cm_sq_fit=kappa_1_m_sq_fit * 0.01,
)

plot_Doplers_broadening(r_t_K=r_t_K, d_lambda_Dopler_m=d_lambda_Doplers_m)

plot_population_number_density(r_t_K=r_t_K, n_i_m_3=n_i_m_3)

plot_number_density(r_t_K=r_t_K, n_m_3=n_m_3, dn=dn_m_3)

plot_population_and_number_density(r_t_K, n_i_m_3=n_i_m_3, n_m_3=n_m_3)

if param["show_plots_flag"]:
    plt.show(block=False)
    input("Press any key to close all plots...")
    plt.close("all")
else:
    print("Calculations have been finished!")
