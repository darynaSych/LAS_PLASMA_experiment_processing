import os
import matplotlib.pyplot as plt
from units_constants import *
from utilites import *

# Import the classes
from image_preprocess import ImagePreprocess
from intensity_analysis import IntensityAnalysis
from optical_param_analysis import OpticalParamAnalysis
from concentration_calculator import ConcentrationCalculator


"""
Ця версія ще буде зберігати в .тхт отримані значення заселеностей та концентрацій 
"""

# SET THE PARAMETERS OF IMAGE

#boolean variable to choose which wavelength should be analyzed
wavelength_flag_G_510nm = True
wavelength_flag_Y_578nm = False

# Folders and profiles
foldername_img = 'Photos_19-12'
filename_img_absorption = "_DSC3426.jpg" # зображення з поглинанням
filename_img_gt = "_DSC3417o.jpg" # ground truth image
foldername_savefig = 'Photos_19-12/3426'

foldername = 'plots_and_results' 
filename_statsum = 'Statsum_CuI.txt' # Statistic sum
filename_temperature = 'temperature_profile_3436.txt' # Temperature profile
filename_OES_results = 'oes_results_3428.txt' # File to compare with OES n-profile


filepath_img_absorption = os.path.join(foldername_img, filename_img_absorption)
filepath_img_gt =os.path.join(foldername_img, filename_img_gt)
filepath_statsum = os.path.join(foldername, filename_statsum)
filepath_temperature = os.path.join(foldername, filename_temperature)
filepath_OES_results = os.path.join(foldername, filename_OES_results)


# Create ROI which determines the boundaries of analysis
x_min=2000
x_max=4000
x_minROI = x_min # Define ROI for preanalysis
x_maxROI  = x_max



# SET THE PARAMETERS OF PLASMA AND LASER
image_parameters = {
    'x_min_electrode': 2041,  # Left limit of the electrode
    'x_max_electrode': 4154,  # Right limit of the electrode
    'y_min_electrode': 1078,  # Lower limit of the electrode
    'y_max_electrode': 2820,  # Upper limit of the electrode
    'region_size' : 3
}
y_crssctn_absorbtion = 2046        # Defined y cross-section if None the crssctn will be defined automatically
y_crssctn_gt = 2046


plasma_parameters_Y_578nm = {
    'lambda_m': 587.2*nm,
    'mu_Cu': 64,
    'f_ik': 0.00328,
    'g_i': 6.0,
    'E_i': 1.64,
    'E_k': 3.82
}

plasma_parameters_G_510nm = {
    'lambda_m': 510.5*nm,
    'mu_Cu': 64,
    'f_ik': 0.00328,
    'g_i': 6.0,
    'E_i': 1.38*eV,
    'E_k': 3.82*eV
}
plasma_parameters = plasma_parameters_G_510nm if wavelength_flag_G_510nm else plasma_parameters_Y_578nm

#FLAGS
save_output_to_txt = True
filepath_save_results_txt = os.path.join(foldername, "results_of_plotting.txt")
save_message = ""

# Instantiate the ImagePreprocess class to read image
image_absorption = ImagePreprocess(
    filepath=filepath_img_absorption,  
    image_parameters=image_parameters,  
    y_crssctn=y_crssctn_absorbtion  
)
image_absorption.read_image_based_on_extension()
analyse_intensity_abs = IntensityAnalysis(image_preprocessor=image_absorption)
x_pxl_abs, intensity_abs = analyse_intensity_abs.extract_intensity_from_region(y_crssctn=y_crssctn_absorbtion)
x_pxl_abs_ROI, intensity_abs_ROI = analyse_intensity_abs.extract_intensity_from_region(x_min_ROI=x_minROI, x_max_ROI=x_maxROI,y_crssctn=y_crssctn_absorbtion)


# Instantiate the ImagePreprocess class for the ground truth image
image_gt = ImagePreprocess(
    filepath=filepath_img_gt,  
    image_parameters=image_parameters,  
    y_crssctn=y_crssctn_gt 
)
image_gt.read_image_based_on_extension()
analyse_intensity_gt = IntensityAnalysis(image_preprocessor=image_gt)

x_pxl_gt, intensity_gt = analyse_intensity_gt.extract_intensity_from_region(y_crssctn=y_crssctn_gt)
x_pxl_gt_ROI, intensity_gt_ROI=analyse_intensity_gt.extract_intensity_from_region(x_min_ROI=x_minROI, x_max_ROI=x_maxROI,y_crssctn=y_crssctn_gt)



# DISPLAY IMAGE WITH RECTANGULAR FRAME AND EDGE-DETECTED IMAGE
fig, [ax1, ax2] = plt.subplots(1, 2)
ax1.imshow(
    image_absorption.draw_rectangle_with_overlay(
        bgr_image=image_absorption.grayscale_image), cmap = 'gray'
) # Display grayscale image with rectange
ax1.set_title("Grayscale image")
ax1.set_xlabel("x [pxl]")
ax1.set_ylabel("y [pxl]")

ax2.imshow(
    image_absorption.draw_rectangle_with_overlay(
        bgr_image=image_absorption.grayscale_image), cmap = 'gray'
) # Display grayscale image with rectange
ax2.set_title("Edge-detected image")
ax2.set_xlabel("x [pxl]")
ax2.set_ylabel("y [pxl]")

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

# Далі фітую РОІ квадратичною функцією
intensity_abs_ROI_square_fit = analyse_intensity_abs.apply_square_fit(x_pxl_ROI=x_pxl_abs_ROI, intensity_ROI=intensity_abs_ROI)
intensity_gt_ROI_square_fit = analyse_intensity_gt.apply_square_fit(x_pxl_ROI=x_pxl_gt_ROI, intensity_ROI=intensity_gt_ROI)
plt.figure()
plt.scatter(x_pxl_abs_ROI, intensity_abs_ROI)
plt.scatter(x_pxl_gt_ROI, intensity_gt_ROI)
plt.plot(x_pxl_abs_ROI, intensity_abs_ROI_square_fit)
plt.plot(x_pxl_gt_ROI, intensity_gt_ROI_square_fit)
plt.title("Region intensity ROI squared fit")


#Тепер х з пікселів переводжу в метри для РОІ який буде використовуватися далі і для всього простору аби його потім заплотити поруч з пікселями і зберегти
x_m_abs = analyse_intensity_abs.x_array_rescale_to_m(x_array_pxl=x_pxl_abs, x_array_pxl_ROI= x_pxl_abs_ROI,intensity_fit_ROI=intensity_abs_ROI_square_fit)
x_m_abs_ROI = analyse_intensity_abs.x_array_rescale_to_m(x_array_pxl_ROI =x_pxl_abs_ROI,x_array_pxl=x_pxl_abs_ROI, intensity_fit_ROI=intensity_abs_ROI_square_fit)
plt.figure()
plt.scatter(x_m_abs_ROI, intensity_abs_ROI)
plt.scatter(x_m_abs, intensity_abs)
plt.title("Region intensity ROI and full row with x in meters")

plt.figure()
plt.scatter(x_pxl_abs_ROI, intensity_abs_ROI)
plt.scatter(x_pxl_abs, intensity_abs)
plt.title("Region intensity ROI and full row with x in meters")

#Порахувати тау і вивести тау від радіуса в області РОІ
optical_param_analysis_sq_fit = OpticalParamAnalysis(x_array_m=x_m_abs_ROI, i_probe=intensity_gt_ROI_square_fit, i_absorption=intensity_abs_ROI_square_fit)
optical_param_analysis_point = OpticalParamAnalysis(x_array_m=x_m_abs_ROI, i_probe=intensity_gt_ROI, i_absorption=intensity_abs_ROI)

tau_ROI = optical_param_analysis_sq_fit.compute_tau()
tau_ROI_point = optical_param_analysis_point.compute_tau()

radius_for_analysis_m, tau_for_analysis = optical_param_analysis_sq_fit.analysis_side_picker(tau_array=tau_ROI,radius_array_m=x_m_abs_ROI,right_side=True)
tau_prime_for_analysis = optical_param_analysis_sq_fit.compute_tau_prime(radius=radius_for_analysis_m,tau=tau_for_analysis)

plt.figure()
plt.scatter(x_m_abs_ROI, tau_ROI, label = 'tau square fit')
plt.scatter(x_m_abs_ROI, tau_ROI_point, label = 'tau scatter')
plt.title("Region intensity ROI and full row with x in meters")

plt.figure()
plt.title('Tau with a picked side')
plt.scatter(radius_for_analysis_m, tau_for_analysis)
from utilites import *
# Тут зараз буду шукати перетворення Абеля. 
radius_for_integration, integrate_result, integrate_error = optical_param_analysis_sq_fit.integrate_Abel(number_points=30, radius_m=radius_for_analysis_m, tau_prime=tau_prime_for_analysis)
kappa_1_cm_fit = apply_square_fit_to_function(radius_for_integration, integrate_result)
# Plot the original integrate_result with error bars
plt.figure()
plt.errorbar(radius_for_integration* 1e3, integrate_result* 1e-2, yerr=integrate_error*1e-2, fmt='o', label='Integrate Result', capsize=5)
# Plot the fitted quadratic function
plt.plot(radius_for_integration* 1e3, kappa_1_cm_fit* 1e-2, label='Fitted Quadratic', linestyle='--', color='red')

compute_concentration = ConcentrationCalculator(plasma_parameters=plasma_parameters_G_510nm)


# Load temperature profile
t_K, d_T_K, r_t_K = compute_concentration.read_t_K(filepath_t_K=filepath_temperature)
kappa_intepolated = interpolate_function( r_t_K,radius_for_integration, kappa_1_cm_fit)

plt.figure()
plt.scatter(r_t_K, kappa_intepolated, label = 'interpolated to the radius T_k')
plt.scatter(radius_for_integration, kappa_1_cm_fit)

n, dn, n_i, lambda_broadening_Doplers = compute_concentration.calculate_concentration(radius=r_t_K, T_K=t_K, d_T_K=d_T_K, kappa_profile=kappa_intepolated, filepath_statsum=filepath_statsum)


#Plot Doplers broadening
plt.figure()
plt.plot(r_t_K * 1e3, lambda_broadening_Doplers*1e9, "o", alpha=0.5)
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
plt.plot(r_t_K * 1e3, n_i*1e6, "o", alpha=0.5, label='Concentration')
plt.xlabel("Radius [mm]")
plt.ylabel(r"Population number density $n_{i}$ [m$^{-3}$]")
plt.yscale("log")
plt.ylim(5e18,1e21)
# plt.savefig(os.path.join(foldername_savefig, 'p_number_density.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)


# Plot number density (m)
plt.figure()
plt.plot(r_t_K * 1e3, n*1e6, "o", alpha=0.5, label='Concentration')
plt.errorbar(r_t_K * 1e3, n*1e6, color = 'black', yerr=dn*1e6, fmt='o', label='Integrate Result', capsize=3)
plt.xlabel("Radius [mm]")
plt.ylabel(r"Number density $n$ [m$^{-3}$]")
plt.yscale("log")
plt.ylim(5e19,5e21)
# plt.savefig(os.path.join(foldername_savefig, 'number_density.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)


