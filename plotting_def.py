import os
import sys
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.ticker import ScalarFormatter
from units_constants import *
from utilites import *
from config_loader import initialize_config

config_file = sys.argv[1]
param = initialize_config(config_file=config_file, plasma_parameters="plasma_parameters.json")
# Globals
save_fig_flag = param["save_fig_flag"]
foldername_savefig = param ["foldername_savefig"]


def plot_image_and_edge_detection(image_left, image_right_edge_detection):
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(
        image_left,
        cmap="gray",
    )  # Display grayscale image with rectange
    ax1.set_title("Grayscale image")
    ax1.set_xlabel("x [pxl]",fontsize=16)
    ax1.set_ylabel("y [pxl]",fontsize=16)

    ax2.imshow(
        image_right_edge_detection,
        cmap="gray",
    )  # Display grayscale image with rectange
    ax2.set_title("Edge-detected image")
    ax2.set_ylabel("y [pxl]",fontsize=16)
    ax2.set_xlabel("x [pxl]",fontsize=16)
    plt.tight_layout()
    if save_fig_flag:
        plt.savefig(
        os.path.join(foldername_savefig, "image_and_edge_detection.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )


def plot_absorption_gt_image(image_absorption, image_gt):
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(image_gt, cmap="gray")  # Display grayscale image with rectange
    ax1.set_title("Probe grayscale image")
    ax1.set_xlabel("x [pxl]",fontsize=16)
    ax1.set_ylabel("y [pxl]",fontsize=16)

    ax2.imshow(
        image_absorption,
        cmap="gray",
    )  # Display grayscale image with rectange
    ax2.set_title("Absorption grayscale image")
    ax2.set_xlabel("x [pxl]",fontsize=16)
    ax2.set_ylabel("y [pxl]",fontsize=16)
    plt.tight_layout()

    if save_fig_flag:
        plt.savefig(
        os.path.join(foldername_savefig, "absorption_gt_image.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )


def plot_region_intensity(x, intensity, x_ROI, intensity_ROI, region_size):
    intensity_crssctn_smoothed = savgol_filter(
        intensity_ROI, window_length=51, polyorder=2
    )  # window_length and polyorder can be adjusted

    # DISPLAY CROSSECTION ROI AND FULL of GT image and absorbtion
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(x_ROI, intensity_ROI, label="Initial intensity")
    ax1.plot(x_ROI, intensity_crssctn_smoothed, label="Smoothed intensity")
    ax1.set_title(f"Cropped towards ROI {x_ROI[0]}x{x_ROI[-1]}")
    ax1.set_xlabel("x [pxl]",fontsize=16)
    ax1.set_ylabel("Intensity",fontsize=16)
    ax1.legend()

    ax2.plot(x, intensity, label="full row absorption")
    ax2.set_title(f"Full width. Region size = {region_size}")
    ax2.set_xlabel("x [pxl]",fontsize=16)
    ax2.set_ylabel("Intensity",fontsize=16)
    ax2.legend()

    if save_fig_flag:
        plt.savefig(
        os.path.join(foldername_savefig, "plot_region_intensity.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )


def plot_region_intensity_abs_gt(
    x_abs,
    intensity_abs,
    x_ROI_abs,
    intensity_ROI_abs,
    x_gt,
    intensity_gt,
    x_ROI_gt,
    intensity_ROI_gt,
    region_size,
):

    # DISPLAY CROSSECTION ROI AND FULL of GT image and absorption
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(11, 6))
    ax1.plot(x_ROI_abs, intensity_ROI_abs, label="Absorption", color="#011f4b")
    ax1.plot(x_ROI_gt, intensity_ROI_gt, label="Probe", color="#005b96")
    ax1.set_title(f"ROI {x_ROI_abs[0]}x{x_ROI_abs[-1]}")
    ax1.set_xlabel("x [pxl]", fontsize=16)
    ax1.set_ylabel("Intensity", fontsize=16)
    ax1.legend()


    ax2.plot(x_abs, intensity_abs, label="Absorption")
    ax2.plot(x_gt, intensity_gt, label="Probe")
    ax2.set_title(f"Full width. Region size = {region_size}")
    ax2.set_xlabel("x [pxl]",fontsize=16)
    ax2.set_ylabel("Intensity",fontsize=16)
    ax2.legend()

    if save_fig_flag:
        plt.savefig(
        os.path.join(foldername_savefig, "region_intensity_abs_gt.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )


def plot_ROI_intensity_square_fit(
    x_pxl_abs_ROI,
    x_pxl_gt_ROI,
    intensity_abs_ROI,
    intensity_gt_ROI,
    intensity_abs_ROI_square_fit,
    intensity_gt_ROI_square_fit,
):
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.scatter(x_pxl_abs_ROI, intensity_abs_ROI, s=10, color='#03396c', label=r"Absorption, $I_{abs}$")
    ax1.scatter(x_pxl_gt_ROI, intensity_gt_ROI, s=10, color='#6497b1', label=r"Probe, $I_{ref}$")
    ax1.plot(x_pxl_abs_ROI, intensity_abs_ROI_square_fit, color='#b35517', linewidth=2, label=r"$I_{abs fit}$")
    ax1.plot(x_pxl_gt_ROI, intensity_gt_ROI_square_fit, color='#e56e1e', linewidth=2, label=r"$I_{ref fit}$")
    ax1.set_title("Region intensity ROI squared fit")
    ax1.set_xlabel("x [pxl]", fontsize=16)
    ax1.set_ylabel("Intensity", fontsize=16)
    ax1.legend()
    
    if save_fig_flag:
        plt.savefig(
        os.path.join(foldername_savefig, "ROI_intensity_square_fit.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )


def plt_ROI_intensity_m(
    x_m_abs_ROI, intensity_abs_ROI, intensity_gt_ROI, point_size=10
):
    fig, ax1 = plt.subplots(figsize=(7, 6))

    ax1.scatter(x_m_abs_ROI * 1e3, intensity_abs_ROI, s=point_size, label="Absorption")
    ax1.scatter(x_m_abs_ROI * 1e3, intensity_gt_ROI, s=point_size, label="Probe")

    ax1.set_xlabel("x, мм",fontsize=16)
    ax1.set_ylabel("Intensity",fontsize=16)
    ax1.set_title("Region intensity (x in meters)")
    # Set x-axis ticks every 0.5 mm
    x_min = np.floor(np.min(x_m_abs_ROI * 1e3) * 2) / 2
    x_max = np.ceil(np.max(x_m_abs_ROI * 1e3) * 2) / 2
    ax1.set_xticks(np.arange(x_min, x_max + 0.5, 0.5))

    ax1.legend()

    if save_fig_flag:
        plt.savefig(
        os.path.join(foldername_savefig, ".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )


def plot_optical_thickness(
    x_m_abs_ROI,
    tau_ROI,
    tau_ROI_point,
    radius_x_m,
    tau_radius,
    side_of_analysis,
    point_size=10,
):
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(11, 6))

    ax1.scatter(
        x_m_abs_ROI*1e3, tau_ROI_point, s=point_size,label= r"$\tau_{0}$"
    )
    ax1.plot(x_m_abs_ROI*1e3, tau_ROI, color = 'red', label=r"$\tau_{0}$ квадр. апроксимація інтенсивності")
    ax1.set_title("Optical thickness")
    ax1.set_xlabel("x, мм",fontsize=16)
    ax1.set_ylabel(r"$\tau_0$",fontsize=16)
    ax1.legend()

    if side_of_analysis == True:
        label = r"$\tau_{0}$ (x>0)" 
    elif side_of_analysis == False:
        label = r"$\tau_{0}$ (x<0)"
    else:
        label = "Label with mistake"

    ax2.plot(radius_x_m*1e3, tau_radius,label=label)
    ax2.set_title("Optical thickness (side of analysis)")
    ax2.set_xlabel("r, мм",fontsize=16)
    ax2.set_ylabel(r"$\tau$",fontsize=16)
    ax2.legend()

    if save_fig_flag:
        plt.savefig(
        os.path.join(foldername_savefig, "optical_thickness.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )



def plot_absorption_coefficient(
    radius_for_integration, kappa_1_cm, integrate_error, kappa_1_cm_sq_fit
):
    # Plot results_of_integration
    fig, ax1 = plt.subplots(figsize=(7, 6))

    ax1.errorbar(
        radius_for_integration * 1e3,
        kappa_1_cm,
        yerr=integrate_error * 1e-2,
        fmt="o",
        label= r"$\kappa_{0}$",
        capsize=5,
    )
    # Plot the fitted quadratic function
    ax1.plot(
        radius_for_integration * 1e3,
        kappa_1_cm_sq_fit,
        label="Квадратична апроксимація результатів",
        linestyle="--",
        color="red",
    )

def plot_absorption_coefficient_from_n_i(
    radius_for_integration, kappa_1_cm_from_n_i 
):
    # Plot results_of_integration
    fig, ax1 = plt.subplots(figsize=(7, 6))

    ax1.scatter(
        radius_for_integration * 1e3, kappa_1_cm_from_n_i ,
        label= r"$\kappa_{0}$",

    )

    # ax1.set_title("Inverse Abel Transform")
    ax1.set_xlabel("r, мм",fontsize=16)
    ax1.set_ylabel(r"$\kappa_{0}\; from n_i, 1/см$",fontsize=16)
    ax1.legend()

    if save_fig_flag:
        plt.savefig(
        os.path.join(foldername_savefig, "absorption_coefficient.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )


def plot_T_K_and_interpolated_kappa(
    r_t_K, t_K, d_T_K, kappa_intepolated, radius_for_integration, kappa_1_cm_sq_fit
):
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6))

    ax1.set_title("Temperature profile (OES)")
    ax1.errorbar(
        r_t_K * 1e3,
        t_K,
        color="black",
        yerr=d_T_K,
        fmt="o",
        capsize=3,
    )
    ax1.set_xlabel("r, мм",fontsize=16)
    ax1.set_ylabel(r"$T, K$",fontsize=16)
    ax1.legend()

    ax2.plot(
        radius_for_integration * 1e3,
        kappa_1_cm_sq_fit,
        linestyle="--",
        color="gray",
        label=r"$\kappa_{0}$, квадратична апроксимація",
    )
    ax2.scatter(r_t_K * 1e3, kappa_intepolated, label="Iнтерпольованi значення", color="black")
    ax2.set_title("Interpolated to number of points of T_K (OES)")
    ax2.set_xlabel("r, мм",fontsize=16)
    ax2.set_ylabel(r"$\kappa_{0}\;, 1/см$",fontsize=16)
    ax2.set_xlim(-0.1, 1.1*r_t_K[-1] * 1e3)
    ax2.legend()

    if save_fig_flag:
        plt.savefig(
        os.path.join(foldername_savefig, "T_K_and_interpolated_kappa.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )


def plot_Doplers_broadening(r_t_K, d_lambda_Dopler_m,d_lambda_Doplers_uncertainty_m):
    """
    Plot Doppler broadening vs radius.
    Parameters:
        r_kappa: radius array [m]
        d_lambda_m: Doppler broadening array [m]
    """
    fig, ax1 = plt.subplots(figsize=(7, 6))
    ax1.errorbar(
        r_t_K * 1e3,
        d_lambda_Dopler_m * 1e9,
        yerr=d_lambda_Doplers_uncertainty_m*1e9,
        fmt="o",
        label="",
        capsize=5,
        color = 'black'
    )
    ax1.set_xlabel("r, мм",fontsize=16)
    ax1.set_ylabel(r"$\Delta \lambda_{D}$, нм",fontsize=16)
    ax1.legend()

    ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    
    if save_fig_flag:
        plt.savefig(
            os.path.join(foldername_savefig, "l_Dopler.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )


def plot_population_number_density(r_t_K, n_i_m_3, n_i_m_3_error):
    # Population number density
    fig, ax1 = plt.subplots(figsize=(7, 6))
    ax1.errorbar(
        r_t_K * 1e3,
        n_i_m_3,
        yerr=n_i_m_3_error,
        fmt="o",
        label="",
        capsize=5,
        color = 'black'
    )
    ax1.set_title("Population number density")
    ax1.set_xlabel("r, мм",fontsize=16)
    ax1.set_ylabel(r"$N_{i}$, м$^{-3}$",fontsize=16)
    ax1.set_yscale("log")
    # ax1.set_ylim(5e18,1e20)
    if save_fig_flag:
        plt.savefig(
            os.path.join(foldername_savefig, "p_number_density.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

def plot_number_density(r_t_K, n_m_3, dn):
    # Population number density
    fig, ax1 = plt.subplots(figsize=(7, 6))

    ax1.errorbar(
        r_t_K * 1e3,
        n_m_3,
        yerr=dn,
        fmt="o",
        label="",
        capsize=5,
        color = 'black'
    )
    # ax1.set_title("Number density")

    ax1.set_xlabel("r, мм", fontsize=16)
    ax1.set_ylabel(r"$N_{Cu}$, м$^{-3}$", fontsize=16)
    ax1.tick_params(axis='both', labelsize=14)
    # ax1.set_yscale("log")
    # ax1.set_ylim(1e21,1e22)
    # ax1.set_ylim(5e18,1e20)
    if save_fig_flag:
        plt.savefig(
            os.path.join(foldername_savefig, "number_density.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

def plot_population_and_number_density(r_t_K, n_i_m_3,n_i_m_3_error, n_m_3, n_m_3_error):
    # Population number density
    fig, ax1 = plt.subplots(figsize=(7, 6))
    ax1.errorbar(
        r_t_K * 1e3,
        n_m_3,
        yerr=n_m_3_error,
        fmt="o",
        capsize=5,
        color='black',
        label=r"$N_{Cu}$"
    )

    ax1.errorbar(
        r_t_K * 1e3,
        n_i_m_3,
        yerr=n_i_m_3_error,
        fmt="o",
        capsize=5,
        color='gray',
        label=r"$N_{i}$"
    )

    # ax1.plot(r_t_K * 1e3, n_i_m_3, "o", alpha=0.9, color='gray', label=r"$N_i$")
    # ax1.plot(r_t_K * 1e3, n_m_3, "o", alpha=0.9, color= 'black',label=r"$N_{Cu}$")

    ax1.set_title("")
    ax1.set_xlabel("r, мм",fontsize=16)
    ax1.set_ylabel(r"$N$, м$^{-3}$",fontsize=16)
    ax1.set_yscale("log")
    # ax1.set_ylim(5e18,1e20)
    ax1.legend()
    if save_fig_flag:
        plt.savefig(
            os.path.join(foldername_savefig, "p_and_number_density.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )
