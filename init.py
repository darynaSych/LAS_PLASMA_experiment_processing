from utilites import *
import os

def init():   
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
