import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from units_constants import *
from utilites import *


def plot_image_and_edge_detection(image_left, image_right_edge_detection):
    fig, [ax1, ax2] = plt.subplots(1, 2,figsize=(10, 4))
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

def plot_region_intensity(x,intensity, x_ROI, intensity_ROI, region_size):
    intensity_crssctn_smoothed = savgol_filter(intensity_ROI, window_length=51, polyorder=2)  # window_length and polyorder can be adjusted

    # DISPLAY CROSSECTION ROI AND FULL of GT image and absorbtion
    fig, [ax1, ax2] = plt.subplots(1,2, figsize=(10, 4))
    ax1.plot(x_ROI, intensity_ROI, label = 'Initial intensity')
    ax1.plot(x_ROI,intensity_crssctn_smoothed, label = 'Smoothed intensity')
    ax1.set_title(f"Cropped towards ROI {x_ROI[0]}x{x_ROI[-1]}")
    ax1.set_xlabel("x [pxl]")
    ax1.set_ylabel("Intensity")
    ax1.legend()

    ax2.plot(x,intensity, label = 'full row absorption')
    ax2.set_title(f"Full width. Region size = {region_size}")
    ax2.set_xlabel("x [pxl]")
    ax2.set_ylabel("Intensity")
    ax2.legend()


def plot_region_intensity_abs_gt(x_abs,intensity_abs, x_ROI_abs, intensity_ROI_abs, 
                                 x_gt ,intensity_gt, x_ROI_gt, intensity_ROI_gt,
                                 region_size):

    # DISPLAY CROSSECTION ROI AND FULL of GT image and absorbtion
    fig, [ax1, ax2] = plt.subplots(1,2, figsize=(11, 6))
    ax1.plot(x_ROI_abs, intensity_ROI_abs, label = 'Absorption')
    ax1.plot(x_ROI_gt, intensity_ROI_gt, label = 'Probe')
    ax1.set_title(f"ROI {x_ROI_abs[0]}x{x_ROI_abs[-1]}")
    ax1.set_xlabel("x [pxl]")
    ax1.set_ylabel("Intensity")
    ax1.legend()

    ax2.plot(x_abs, intensity_abs, label = 'Absorption')
    ax2.plot(x_gt, intensity_gt, label = 'Probe')    
    ax2.set_title(f"Full width. Region size = {region_size}")
    ax2.set_xlabel("x [pxl]")
    ax2.set_ylabel("Intensity")
    ax2.legend()
