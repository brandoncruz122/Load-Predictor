import sys
from PyQt5.QtWidgets import QDialog, QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5 import QtCore
from PyQt5.QtWidgets import *

# inputvaluesdict = {}

class VariableEntryWidget(QWidget):
    def __init__(self, label, parent=None):
        super(VariableEntryWidget, self).__init__(parent)
        self.label = QLabel(label)
        self.entry = QLineEdit()
        # Set a fixed width for the QLineEdit widget (e.g., 100 pixels)
        self.entry.setFixedWidth(100)
        self.label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.entry)
        self.setLayout(layout)

class VariableInputDialog(QDialog):
    def __init__(self, labels, parent=None):
        super(VariableInputDialog, self).__init__(parent)
        self.variable_widgets = []

        for label in labels:
            variable_widget = VariableEntryWidget(label)
            self.variable_widgets.append(variable_widget)
    
        self.submit_button = QPushButton("Submit")
        self.use_preset_button = QPushButton("Use Preset")

        self.submit_button.clicked.connect(self.submit_clicked)
        self.use_preset_button.clicked.connect(self.use_preset_clicked)

        layout = QVBoxLayout()
        for widget in self.variable_widgets:
            layout.addWidget(widget)
        layout.addWidget(self.submit_button)
        layout.addWidget(self.use_preset_button)

        self.setLayout(layout)

    def submit_clicked(self):
        values = [float(widget.entry.text()) for widget in self.variable_widgets]
        self.result = dict(zip(variable_names, values))
        # inputvaluesdict = self.result
        inputvaluesdict.update(self.result)
        self.callscript()
        
        '''
        The script containing the code for the simulation must contain the following at the top:
        
        from __main__ import *
        for key, value in inputvaluesdict.items():
            exec(f"{key} = {value}")
            print(f"{key} = {value}")

        print("CLASS.py is running...")
        '''
        self.reject()
    def callscript(self):
        import CLASS
        self.reject()

    def compute_sum(self):
        values = [float(widget.entry.text()) for widget in self.variable_widgets]
        return sum(values)
    
    def use_preset_clicked(self):
        preset_values = [961.3471 / 1000, 620.9414 / 1000, 544.3625 / 1000, -438.9258 / 1000,  # Load setup
                         70e9,  # Young's Modulus
                         986.5268,  # Boundary location
                         150, -151, 50,  # Training setup
                         550, -550, 350,  # Testing setup
                         1,  # Noise level
                         24,  # Thermal Conductivity
                         10, -10,  # Training temperature
                         20, -20]  # Testing temperature

        for widget, value in zip(self.variable_widgets, preset_values):
            widget.entry.setText(str(value))

    def get_values(self):
        return self.result

if __name__ == '__main__':
    variable_names = ["F_l_1", "F_l_2", "F_l_3", "F_l_4", "E", "BL", "max_training_load", "min_training_load",
                      "increment_training", "max_testing_load", "min_testing_load", "increment_testing",
                      "noise_level", "alpha", "max_training_temp", "min_training_temp", "max_testing_temp",
                      "min_testing_temp"]

    inputvaluesdict = {}
    app = QApplication(sys.argv)
    input_dialog = VariableInputDialog(["Load 1 location [mm]", "Load 2 location [mm]:", "Load 3 location [mm]:",
                                        "load 4 location [mm]", "Young's Modulus in [Pa]",
                                        "Boundary location in [mm]", "Max training load in [N]",
                                        "Min training load in [N]", "Training increment",
                                        "Max testing load in [N]", "Min testing load in [N]", "Testing increment",
                                        "Noise level [microstrain]", "Thermal Conductivity [micro-scale]",
                                        "Max training temp [C]", "Min training temp [C]", "Max testing temp [C]",
                                        "Min testing temp [C]"])
    input_dialog.show()

    sys.exit(app.exec_())
