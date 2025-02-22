# import all neccessary packages
import numpy as np
import rawpy
import cv2
import os
from units_constants import *
from PIL import Image
from utilites import *  


class ImagePreprocess:
    width_electrode_m = 6e-3  # ВИНЕСТИ ЯК КОНСТАНТУ

    def __init__(
        self,
        filepath: str,
        image_parameters: dict,
        y_crssctn: int,
    ):
        self.filepath = filepath
        self.x_min_electrode = image_parameters.get("x_min_electrode")
        self.x_max_electrode = image_parameters.get("x_max_electrode")
        self.y_min_electrode = image_parameters.get("y_min_electrode")
        self.y_max_electrode = image_parameters.get("y_max_electrode")
        self.region_size = image_parameters.get("region_size")

        self.y_crssctn = (
            y_crssctn
            if y_crssctn is not None
            else y_min_electrode + (y_max_electrode - y_min_electrode) // 2
        )
        self.dpxl_m = self.width_electrode_m / (
            self.x_max_electrode - self.x_min_electrode
        )  # pixel size in meters

        self.rgb_image = None  # Placeholder for RGB image
        self.grayscale_image = None  # Placeholder for grayscale image

    def _convert_to_grayscale(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Convert an RGB image to grayscale using standard weights.
        """
        return np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    def read_raw_image(self) -> None:
        """
        Read a raw image and convert it to grayscale.
        """
        with rawpy.imread(self.filepath) as raw:
            self.rgb_image = raw.postprocess()
        self.grayscale_image = self._convert_to_grayscale(self.rgb_image)

    def read_jpg_image(self) -> None:
        """
        Read a JPG image and convert it to grayscale.
        """
        with Image.open(self.filepath) as img:
            self.rgb_image = np.array(img.convert("RGB"))
        self.grayscale_image = self._convert_to_grayscale(self.rgb_image)

    def read_image_based_on_extension(self):
        """
        Reads an image using the appropriate method based on its file extension.

        Args:
            image_processor (ImagePreprocess): An instance of the ImagePreprocess class.
        """
        # Extract the file extension
        _, file_extension = os.path.splitext(self.filepath)

        # Call the appropriate method based on the extension
        if file_extension.lower() == ".jpg":
            self.read_jpg_image()
        else:
            self.read_raw_image()

    def _draw_rectangle_with_overlay(
        self, bgr_image: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        """
        Draw a rectangle with a semi-transparent overlay on the image.

        Args:
            bgr_image (np.ndarray): Image to modify.
            alpha (float): Transparency factor for the overlay.

        Returns:
            np.ndarray: Image with the overlay.
        """
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
        overlay = bgr_image.copy()

        cv2.rectangle(
            overlay,
            (self.x_min_electrode, self.y_min_electrode),
            (self.x_max_electrode, self.y_max_electrode),
            (255, 0, 0),
            thickness=-1,  # Fill the rectangle
        )
        return cv2.addWeighted(overlay, alpha, bgr_image, 1 - alpha, 0)

    def draw_rectangle_with_overlay(
        self, bgr_image: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

        # Draw a red rectangle on the selected region
        cv2.rectangle(
            bgr_image,
            (self.x_min_electrode, self.y_min_electrode),
            (self.x_max_electrode, self.y_max_electrode),
            (255, 0, 0),
            thickness=4,
        )
        # Draw a red horizontal line in the center
        x_min = max(0, self.x_min_electrode)
        x_max = min(bgr_image.shape[1], self.x_max_electrode)
        y_start = max(0, self.y_crssctn - self.region_size // 2)
        y_end = min(
            bgr_image.shape[0],
            self.y_crssctn + self.region_size // 2 + (self.region_size % 2),
        )
        # Define the rectangular region (top-left and bottom-right corners)
        x_start, y_start = x_min, y_start  # Top-left corner (x, y)
        x_end, y_end = x_max, y_end  # Bottom-right corner (x, y)
        channel_to_modify = 2
        # Modify the red channel (index 2 for OpenCV BGR format) within the region

        # Create a red overlay with the same shape as the image
        overlay = bgr_image.copy()
        overlay[y_start:y_end, x_start:x_end] = [255, 0, 0]  # Red in BGR format

        # Blend the overlay with the original image
        alpha = 0.5  # Transparency factor (0.0 = fully transparent, 1.0 = fully opaque)
        bgr_image[y_start:y_end, x_start:x_end] = cv2.addWeighted(
            bgr_image[y_start:y_end, x_start:x_end],
            1 - alpha,
            overlay[y_start:y_end, x_start:x_end],
            alpha,
            0,
        )
        return bgr_image

    def edge_detection(self) -> np.ndarray:
        """
        Apply Canny edge detection to the grayscale image.

        Returns:
            np.ndarray: Binary image with edges detected.
        """
        return cv2.Canny(
            self.grayscale_image,
            threshold1=195,
            threshold2=230,
            apertureSize=5,
            L2gradient=True,
        )
