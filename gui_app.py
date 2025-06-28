import sys
import json
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QFileDialog, QCheckBox, QScrollArea,QGroupBox,QFormLayout, QRadioButton, QButtonGroup

)
from PyQt5.QtCore import Qt

PARAMS_FILE = "last_params.json"
EXPERIMENT_FILE = "experiment_config.json"


class StartupWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Experiment Folder")
        self.setGeometry(200, 200, 400, 200)

        central = QWidget()
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        self.label = QLabel("Please select folder for experiment (e.g. 12-05-2025)")
        self.folder_path = QLineEdit()
        browse_btn = QPushButton("Browse Folder")
        go_btn = QPushButton("GO!")

        layout.addWidget(self.label)
        layout.addWidget(self.folder_path)
        layout.addWidget(browse_btn)
        layout.addWidget(go_btn)

        browse_btn.clicked.connect(self.choose_folder)
        go_btn.clicked.connect(self.go_to_param_window)

        self.load_last_experiment_folder()

    def load_last_experiment_folder(self):
        if os.path.exists(EXPERIMENT_FILE):
            try:
                with open(EXPERIMENT_FILE, "r") as f:
                    folder = json.load(f).get("experiment_folder", "")
                    self.folder_path.setText(folder)
            except Exception as e:
                print("Failed to load previous experiment folder:", e)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_path.setText(folder)

    def go_to_param_window(self):
        folder = self.folder_path.text().strip()
        if not folder:
            return

        # Save to config
        try:
            with open(EXPERIMENT_FILE, "w") as f:
                json.dump({"experiment_folder": folder}, f)
        except Exception as e:
            print("Failed to save experiment folder:", e)

        self.param_window = ParamWindow()
        self.param_window.show()
        self.close()


class ParamWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plasma Parameters Input")
        self.setGeometry(100, 100, 600, 500)

        # Load experiment folder
        self.experiment_folder = ""
        if os.path.exists(EXPERIMENT_FILE):
            with open(EXPERIMENT_FILE, "r") as f:
                self.experiment_folder = json.load(f).get("experiment_folder", "")

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)

        container = QWidget()
        self.layout = QVBoxLayout(container)
        scroll.setWidget(container)

        # === Група: Межі електродів ===
        electrode_group = QGroupBox("Set limits of electrodes")
        form_layout = QFormLayout()

        self.x_max_electrode = QLineEdit()
        self.y_min_electrode = QLineEdit()
        self.y_max_electrode = QLineEdit()
        self.region_size = QLineEdit()

        form_layout.addRow("x_max_electrode:", self.x_max_electrode)
        form_layout.addRow("y_min_electrode:", self.y_min_electrode)
        form_layout.addRow("y_max_electrode:", self.y_max_electrode)
        form_layout.addRow("region_size:", self.region_size)

        electrode_group.setLayout(form_layout)
        self.layout.addWidget(electrode_group)

        # === Група: межі аналізу ===
        image_group = QGroupBox("Set boundaries of image analysis")
        form_layout = QFormLayout()

        self.x_minROI_input = QLineEdit()
        self.x_maxROI_input = QLineEdit()
        self.y_abs_input = QLineEdit()
        self.y_gt_input = QLineEdit()

        form_layout.addRow("x_minROI:", self.x_minROI_input)
        form_layout.addRow("x_maxROI:", self.x_maxROI_input)
        form_layout.addRow("y_crssctn_absorbtion:", self.y_abs_input)
        form_layout.addRow("y_crssctn_gt:", self.y_gt_input)

        image_group.setLayout(form_layout)
        self.layout.addWidget(image_group)

        # === Група: flags  ===
        flags_group = QGroupBox("Select flags")
        form_layout = QFormLayout()

        # Checkbox for side selection
        # === Горизонтальне розміщення RadioButtons для довжин хвиль ===
        side_layout = QHBoxLayout()
        self.side_right_radio = QRadioButton("Right")
        self.side_left_radio = QRadioButton("Left")

        # Додаємо обидві радіокнопки в групу
        self.side_group = QButtonGroup()
        self.side_group.setExclusive(True)
        self.side_group.addButton(self.side_right_radio)
        self.side_group.addButton(self.side_left_radio)

        side_layout.addWidget(self.side_right_radio)
        side_layout.addWidget(self.side_left_radio)

        form_layout.addRow(QLabel("Select side for analysis:"), side_layout)

        # === Горизонтальне розміщення RadioButtons для довжин хвиль ===
        wavelength_layout = QHBoxLayout()
        self.wavelength_G_radio = QRadioButton("Green λ = 510 nm")
        self.wavelength_Y_radio = QRadioButton("Yellow λ = 578 nm")

        # Додаємо обидві радіокнопки в групу
        self.wavelength_group = QButtonGroup()
        self.wavelength_group.setExclusive(True)
        self.wavelength_group.addButton(self.wavelength_G_radio)
        self.wavelength_group.addButton(self.wavelength_Y_radio)

        wavelength_layout.addWidget(self.wavelength_G_radio)
        wavelength_layout.addWidget(self.wavelength_Y_radio)

        form_layout.addRow(QLabel("Select wavelength:"), wavelength_layout)

        flags_group.setLayout(form_layout)
        self.layout.addWidget(flags_group)




        self.abs_path = QLineEdit()
        self.gt_path = QLineEdit()
        self.temp_path = QLineEdit()

        self.add_file_selector("Absorption Image:", self.abs_path)
        self.add_file_selector("GT Image:", self.gt_path)
        self.add_file_selector("Temperature File:", self.temp_path)

        self.submit_button = QPushButton("Run Analysis")
        self.submit_button.clicked.connect(self.submit_params)
        self.layout.addWidget(self.submit_button)

        self.load_last_params()

    def add_labeled_input(self, label_text, line_edit):
        label = QLabel(label_text)
        self.layout.addWidget(label)
        self.layout.addWidget(line_edit)

    def add_file_selector(self, label_text, line_edit):
        row = QHBoxLayout()
        row.addWidget(QLabel(label_text))
        row.addWidget(line_edit)
        browse = QPushButton("Browse")
        browse.clicked.connect(lambda _, le=line_edit: le.setText(
QFileDialog.getOpenFileName(self, "Select File", os.path.join(self.experiment_folder, "Input_files"))[0]
        ))
        row.addWidget(browse)
        self.layout.addLayout(row)

    def load_last_params(self):
        if os.path.exists(PARAMS_FILE):
            try:
                with open(PARAMS_FILE, "r") as f:
                    data = json.load(f)
                    self.x_minROI_input.setText(str(data.get("x_minROI", "")))
                    self.x_maxROI_input.setText(str(data.get("x_maxROI", "")))
                    self.y_abs_input.setText(str(data.get("y_crssctn_absorbtion", "")))
                    self.y_gt_input.setText(str(data.get("y_crssctn_gt", "")))
                    self.side_checkbox.setChecked(data.get("right_side_pick_flag", True))

                    self.abs_path.setText(data.get("filepath_img_absorption", ""))
                    self.gt_path.setText(data.get("filepath_img_gt", ""))
                    self.temp_path.setText(data.get("filepath_temperature", ""))
            except Exception as e:
                print("Error loading saved params:", e)

    def save_last_params(self, params):
        try:
            with open(PARAMS_FILE, "w") as f:
                json.dump(params, f, indent=2)
        except Exception as e:
            print("Error saving params:", e)

    def submit_params(self):
        try:
            params = {
                "x_minROI": int(self.x_minROI_input.text()),
                "x_maxROI": int(self.x_maxROI_input.text()),
                "y_crssctn_absorbtion": int(self.y_abs_input.text()),
                "y_crssctn_gt": int(self.y_gt_input.text()),
                "right_side_pick_flag": self.side_checkbox.isChecked(),
                "filepath_img_absorption": self.abs_path.text(),
                "filepath_img_gt": self.gt_path.text(),
                "filepath_temperature": self.temp_path.text(),

                "image_parameters": {
                    "x_min_electrode": 2100,
                    "x_max_electrode": 4230,
                    "y_min_electrode": 1370,
                    "y_max_electrode": 3320,
                    "region_size": 50,
                },
                "number_of_points_for_integration": 30,
                "save_output_to_txt": True,
                "show_plots_flag": True,
                "plasma_parameters": {
                    "element": "CuI",
                    "filepath_OES_results": os.path.join(self.experiment_folder, "oes_results_3428.txt"),
                    "filepath_statsum": os.path.join(self.experiment_folder, "Statsum_CuI.txt"),
                },
                "filepath_save_results_txt": os.path.join(self.experiment_folder, "results.txt"),
            }

            self.save_last_params(params)

            # from main_analysis import run_analysis
            # run_analysis(param=params)

        except Exception as e:
            print("❌ Error in parameters or run_analysis:", e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    startup = StartupWindow()
    startup.show()
    sys.exit(app.exec_())
