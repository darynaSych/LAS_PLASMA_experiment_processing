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
    ax1.set_xlabel("x [pxl]")
    ax1.set_ylabel("y [pxl]")

    ax2.imshow(
        image_right_edge_detection,
        cmap="gray",
    )  # Display grayscale image with rectange
    ax2.set_title("Edge-detected image")
    ax2.set_xlabel("x [pxl]")
    ax2.set_ylabel("y [pxl]")

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
    ax1.set_xlabel("x [pxl]")
    ax1.set_ylabel("y [pxl]")

    ax2.imshow(
        image_absorption,
        cmap="gray",
    )  # Display grayscale image with rectange
    ax2.set_title("Absorption grayscale image")
    ax2.set_xlabel("x [pxl]")
    ax2.set_ylabel("y [pxl]")

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
    ax1.set_xlabel("x [pxl]")
    ax1.set_ylabel("Intensity")
    ax1.legend()

    ax2.plot(x, intensity, label="full row absorption")
    ax2.set_title(f"Full width. Region size = {region_size}")
    ax2.set_xlabel("x [pxl]")
    ax2.set_ylabel("Intensity")
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

    # DISPLAY CROSSECTION ROI AND FULL of GT image and absorbtion
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(11, 6))
    ax1.plot(x_ROI_abs, intensity_ROI_abs, label="Absorption")
    ax1.plot(x_ROI_gt, intensity_ROI_gt, label="Probe")
    ax1.set_title(f"ROI {x_ROI_abs[0]}x{x_ROI_abs[-1]}")
    ax1.set_xlabel("x [pxl]")
    ax1.set_ylabel("Intensity")
    ax1.legend()

    ax2.plot(x_abs, intensity_abs, label="Absorption")
    ax2.plot(x_gt, intensity_gt, label="Probe")
    ax2.set_title(f"Full width. Region size = {region_size}")
    ax2.set_xlabel("x [pxl]")
    ax2.set_ylabel("Intensity")
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
    ax1.scatter(x_pxl_abs_ROI, intensity_abs_ROI, s=10, alpha=0.5, label="Absorption")
    ax1.scatter(x_pxl_gt_ROI, intensity_gt_ROI, s=10, alpha=0.5, label="Probe")
    ax1.plot(x_pxl_abs_ROI, intensity_abs_ROI_square_fit)
    ax1.plot(x_pxl_gt_ROI, intensity_gt_ROI_square_fit)
    ax1.set_title("Region intensity ROI squared fit")
    ax1.set_xlabel("x [pxl]")
    ax1.set_ylabel("Intensity")
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

    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("Intensity")
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
        x_m_abs_ROI, tau_ROI_point, s=point_size, label="tau from intensty data points"
    )
    ax1.plot(x_m_abs_ROI, tau_ROI, label="tau from square fit intensty")
    ax1.set_title("Optical thickness")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel(r"$\tau$")
    ax1.legend()

    if side_of_analysis == True:
        label = "Optical thickness of right side of image"
    elif side_of_analysis == False:
        label = "Optical thickness of left side of image"
    else:
        label = "Label with mistake"

    ax2.plot(radius_x_m, tau_radius, label=label)
    ax2.set_title("Optical thickness (side of analysis)")
    ax2.set_xlabel("r [m]")
    ax2.set_ylabel(r"$\tau$")
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
        label="Koefficient of absorption",
        capsize=5,
    )
    # Plot the fitted quadratic function
    ax1.plot(
        radius_for_integration * 1e3,
        kappa_1_cm_sq_fit,
        label="Quadratic fit of results",
        linestyle="--",
        color="red",
    )

    ax1.set_title("Inverse Abel Transform")
    ax1.set_xlabel("r [mm]")
    ax1.set_ylabel(r"$\kappa_{0}\;[1/cm]$")
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
    ax1.set_xlabel("r [mm]")
    ax1.set_ylabel(r"$T [K]$")
    ax1.legend()

    ax2.plot(
        radius_for_integration * 1e3,
        kappa_1_cm_sq_fit,
        linestyle="--",
        color="gray",
        label=r"$\kappa_{0}$ square fit",
    )
    ax2.scatter(r_t_K * 1e3, kappa_intepolated, label="Interpolated", color="black")
    ax2.set_title("Interpolated to number of points of T_K (OES)")
    ax2.set_xlabel("r [mm]")
    ax2.set_ylabel(r"$\kappa_{0}\;[1/cm]$")
    ax2.legend()

    if save_fig_flag:
        plt.savefig(
        os.path.join(foldername_savefig, "T_K_and_interpolated_kappa.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )


def plot_Doplers_broadening(r_t_K, d_lambda_Dopler_m):
    """
    Plot Doppler broadening vs radius.
    Parameters:
        r_kappa: radius array [m]
        d_lambda_m: Doppler broadening array [m]
    """
    fig, ax1 = plt.subplots(figsize=(7, 6))

    ax1.plot(r_t_K * 1e3, d_lambda_Dopler_m * 1e9, "o", alpha=0.5, label="")
    ax1.set_xlabel("Radius [mm]")
    ax1.set_ylabel(r"Doppler broadening, $\Delta \lambda_{D}$ [$nm$]")
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


def plot_population_number_density(r_t_K, n_i_m_3):
    # Population number density
    fig, ax1 = plt.subplots(figsize=(7, 6))
    ax1.plot(r_t_K * 1e3, n_i_m_3, "o", alpha=0.5, label="")
    ax1.set_title("Population number density")
    ax1.set_xlabel("Radius [mm]")
    ax1.set_ylabel(r"Population number density $n_{i}$ [m$^{-3}$]")
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
        yerr=dn * 1e6,
        fmt="o",
        label="",
        capsize=5,
    )
    ax1.set_title("Number density")
    ax1.set_xlabel("Radius [mm]")
    ax1.set_ylabel(r"Number density $n_{i}$ [m$^{-3}$]")
    ax1.set_yscale("log")
    # ax1.set_ylim(5e18,1e20)
    if save_fig_flag:
        plt.savefig(
            os.path.join(foldername_savefig, "number_density.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

def plot_population_and_number_density(r_t_K, n_i_m_3, n_m_3):
    # Population number density
    fig, ax1 = plt.subplots(figsize=(7, 6))
    ax1.plot(r_t_K * 1e3, n_i_m_3, "o", alpha=0.5, label="Population number density")
    ax1.plot(r_t_K * 1e3, n_m_3, "o", alpha=0.5, label="Cu number density")

    ax1.set_title("")
    ax1.set_xlabel("Radius [mm]")
    ax1.set_ylabel(r"$N$ [m$^{-3}$]")
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
