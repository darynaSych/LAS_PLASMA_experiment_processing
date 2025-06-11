import os
import json


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


def initialize_config(config_file, plasma_parameters):
    config = load_config(config_file)

    foldername_root = config["foldername_root"]
    
    # Define additional folders
    foldername_temperature = os.path.join(foldername_root, 'Input_files/Input_parameters')
    foldername_statsum = os.path.join(foldername_root, 'Input_files/Input_parameters')
    foldername_results = os.path.join(foldername_root, 'Results')  # <-- results folder
    foldername_results_image = os.path.join(
    foldername_results, 
f"{config['filename_img_absorption']}abs_gt{config['filename_img_gt']}_rs{config['region_size']}_cssctn{config['y_crssctn_absorbtion']}"
)

    # Create folders if they don't exist
    for path in [foldername_root, foldername_temperature, foldername_results, foldername_results_image]:
        os.makedirs(path, exist_ok=True)

    paths = {
        "filepath_img_absorption": os.path.join(foldername_root, "Input_files", config["filename_img_absorption"]),
        "filepath_img_gt": os.path.join(foldername_root, "Input_files", config["filename_img_gt"]),
        "filepath_statsum": os.path.join(foldername_statsum, config['filename_statsum']),
        "filepath_temperature": os.path.join(foldername_temperature, config["filename_temperature"]),
        "filepath_OES_results": os.path.join(foldername_temperature, config["filename_OES_results"]),
        "filepath_save_results_txt": os.path.join(foldername_results_image, 'Results.txt'),
        "foldername_savefig" : foldername_results_image,
        "save_fig_flag": config["save_fig_flag"],
        "show_plots_flag": config["show_plots_flag"]
    }

    # Image Processing Parameters
    image_parameters = {
        "x_min_electrode": config["x_min_electrode"],
        "x_max_electrode": config["x_max_electrode"],
        "y_min_electrode": config["y_min_electrode"],
        "y_max_electrode": config["y_max_electrode"],
        "region_size": config["region_size"],
    }


    with open(plasma_parameters, "r") as f:
        plasma_dict = json.load(f)

    # Select the right parameter set based on flags
    if config["wavelength_flag_G_510nm"]:
        plasma_parameters = plasma_dict["G_510nm"]
    elif config["wavelength_flag_Y_578nm"]:
        plasma_parameters = plasma_dict["Y_578nm"]
    else:
        raise ValueError("No wavelength flag is set to True in the config.")

    settings = {
        "x_minROI": config["x_minROI"],
        "x_maxROI": config["x_maxROI"],
        "wavelength_flag_G_510nm": config["wavelength_flag_G_510nm"],
        "wavelength_flag_Y_578nm": config["wavelength_flag_Y_578nm"],
        "save_output_to_txt": config["save_output_to_txt"],
        "y_crssctn_absorbtion": config["y_crssctn_absorbtion"],
        "y_crssctn_gt": config["y_crssctn_gt"],
        "right_side_pick_flag": config["right_side_pick_flag"],
        "number_of_points_for_integration": config["number_of_points_for_integration"],
        "plasma_parameters": plasma_parameters
    }

    return {**paths, **settings, "image_parameters": image_parameters}


