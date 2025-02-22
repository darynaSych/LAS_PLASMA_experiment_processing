import numpy as np
import matplotlib.pyplot as plt
import rawpy
import cv2
import os
from units_constants import *
from utilites import * 
from image_preprocess import ImagePreprocess

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