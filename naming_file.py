#SET THE PARAMETERS OF IMAGE

# Create ROI which determines the boundaries of analysis
x_minROI = 2000
x_maxROI = 4000


# MODIFY (Boolean variable to choose which wavelength should be analyzed)
wavelength_flag_G_510nm = True
wavelength_flag_Y_578nm = False


# Folders and profiles
foldername_img = "Photos_19-12"
filename_img_absorption = "_DSC3426.jpg"  # Absorption image
filename_img_gt = "_DSC3417o.jpg"  # Ground truth image
foldername_savefig = "Photos_19-12/3426"

foldername = "plots_and_results"
filename_statsum = "Statsum_CuI.txt"  # Statistic sum
filename_temperature = "temperature_profile_3436.txt"  # Temperature profile
filename_OES_results = "oes_results_3428.txt"  # File to compare with OES n-profile