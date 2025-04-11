import os
from PLASMA_PARAMETERS import *


def load_config(filepath):
    """Reads a configuration file and returns a dictionary of parameters, ignoring comments."""
    config = {}
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):  # Ignore empty lines and full-line comments
                continue
            
            # Remove inline comments
            line = line.split("#", 1)[0].strip()

            if "=" not in line:
                continue  # Skip malformed lines

            key, value = line.split("=", 1)  # Split at the first '='
            key, value = key.strip(), value.strip()

            # Convert types automatically (bool, int, float)
            if value.lower() in ["true", "false"]:
                config[key] = value.lower() == "true"
            elif value.isdigit():
                config[key] = int(value)
            elif "." in value and value.replace(".", "").isdigit():
                config[key] = float(value)
            else:
                config[key] = value  # Keep as string if no conversion
    return config


def initialize_config(config_file):
    config = load_config(config_file)
    
    # Image file paths
    foldername_img = config["foldername_img"]
    foldername = config["foldername"]

    paths = {
        "filepath_img_absorption": os.path.join(foldername_img, config["filename_img_absorption"]),
        "filepath_img_gt": os.path.join(foldername_img, config["filename_img_gt"]),
        "filepath_statsum": os.path.join(foldername, config["filename_statsum"]),
        "filepath_temperature": os.path.join(foldername, config["filename_temperature"]),
        "filepath_OES_results": os.path.join(foldername, config["filename_OES_results"]),
        "filepath_save_results_txt": config["filepath_save_results_txt"],
        "foldername": config["foldername"],
        "foldername_savefig" : config["foldername_savefig"],
        "save_fig_flag" : config["save_fig_flag"]
    }

    # Image Processing Parameters
    image_parameters = {
        "x_min_electrode": config["x_min_electrode"],
        "x_max_electrode": config["x_max_electrode"],
        "y_min_electrode": config["y_min_electrode"],
        "y_max_electrode": config["y_max_electrode"],
        "region_size": config["region_size"],
    }

    # Flags and other parameters
    settings = {
        "x_minROI": config["x_minROI"],
        "x_maxROI": config["x_maxROI"],
        "wavelength_flag_G_510nm": config["wavelength_flag_G_510nm"],
        "wavelength_flag_Y_578nm": config["wavelength_flag_Y_578nm"],
        "save_output_to_txt": config["save_output_to_txt"],
        "save_message": config["save_message"],
        "y_crssctn_absorbtion": config["y_crssctn_absorbtion"],
        "y_crssctn_gt": config["y_crssctn_gt"],
        "right_side_pick_flag": config["right_side_pick_flag"],
        "number_of_points_for_integration": config["number_of_points_for_integration"],
        "plasma_parameters": plasma_parameters_G_510nm if config["wavelength_flag_G_510nm"] else plasma_parameters_Y_578nm
    }

    return {**paths, **settings, "image_parameters": image_parameters}


