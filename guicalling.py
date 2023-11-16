# By Brandon Cruz last modified 11/15/23

import sys  
from PyQt5.QtWidgets import QDialog, QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5 import QtCore  
from PyQt5 import QtCore, QtWidgets  
from PyQt5.QtWidgets import *  # Import all classes from PyQt5.QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# Class for creating user input boxes
class VariableEntryWidget(QWidget):
    def __init__(self, label, parent=None):
        # super allows you to use methods from the superclass, used here to create qwidget object
        super(VariableEntryWidget, self).__init__(parent)
        self.label = QLabel(label) # create text annoation format
        self.entry = QLineEdit() # create entry box format
        # Set a fixed width for the QLineEdit widget 100 pixels
        self.entry.setFixedWidth(100)
        # align text
        self.label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        
        #
        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.entry)
        self.setLayout(layout)         

# Create the format of the objects in the pyqt window, set labels for buttons and text entries
class VariableInputDialog(QDialog):
    def __init__(self, labels, parent=None):
        super(VariableInputDialog, self).__init__(parent)
        self.variable_widgets = []
        
        # assign each label to each entry box
        for label in labels:
            variable_widget = VariableEntryWidget(label)
            self.variable_widgets.append(variable_widget)
        
        # label each button
        self.submit_button = QPushButton("Submit")
        self.use_preset_button = QPushButton("Use Preset")
        
        # specify what function runs when you click the buttons
        self.submit_button.clicked.connect(self.submit_clicked)
        self.use_preset_button.clicked.connect(self.use_preset_clicked)

        layout = QVBoxLayout()
        
        # add all widgets to the layout
        for widget in self.variable_widgets:
            layout.addWidget(widget)
        
        # remove space between entry boxes
        layout.setSpacing(0)
        layout.addWidget(self.submit_button)
        layout.addWidget(self.use_preset_button)
        
        self.setLayout(layout)
    
    # function runs when the submit button is pressed
    def submit_clicked(self):
        # for each entry in each text box, convert to float and add the contens to a list of values
        values = [float(widget.entry.text()) for widget in self.variable_widgets]
        # store the values and their corresponding variable_names in a dictionary
        self.result = dict(zip(variable_names, values))
        # inputvaluesdict = self.result
        
        # update the dictionary with the one in the global variables so that it gets passed to the CLASS program
        inputvaluesdict.update(self.result)
        
        # Run the function that calls the training program
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
        # Run the code in CLASS.py which contains the training and testing program
        import CLASS  
        self.reject()
    
    # this function returns the sum of all the text entries, useed for debugging the preset values, unused now
    def compute_sum(self):
        values = [float(widget.entry.text()) for widget in self.variable_widgets]
        return sum(values)
    
    # clicking the use preset button loads all of these values into the entry widgets (autofills the boxes)
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
        # for each entry box and corresponding value, fill the box with the value as string (will later get converted to float)
        for widget, value in zip(self.variable_widgets, preset_values):
            widget.entry.setText(str(value))

    def get_values(self):
        return self.result

if __name__ == '__main__':
    # list containing all of the variables names in the same order that they will be collected in the user input boxes
    variable_names = ["F_l_1", "F_l_2", "F_l_3", "F_l_4", "E", "BL", "max_training_load", "min_training_load",
                      "increment_training", "max_testing_load", "min_testing_load", "increment_testing",
                      "noise_level", "alpha", "max_training_temp", "min_training_temp", "max_testing_temp",
                      "min_testing_temp"]

    inputvaluesdict = {}  # Define empty dictionary to modify as user inputs are processed, will send to the training program
    app = QApplication(sys.argv)  # Create a PyQt application
    
    # List containing all of the user entry box labels
    input_dialog = VariableInputDialog(["Load 1 location [mm]", "Load 2 location [mm]:", "Load 3 location [mm]:",
                                        "load 4 location [mm]", "Young's Modulus in [Pa]",
                                        "Boundary location in [mm]", "Max training load in [N]",
                                        "Min training load in [N]", "Training increment",
                                        "Max testing load in [N]", "Min testing load in [N]", "Testing increment",
                                        "Noise level [microstrain]", "Thermal Conductivity [micro-scale]",
                                        "Max training temp [C]", "Min training temp [C]", "Max testing temp [C]",
                                        "Min testing temp [C]"])
    # shows the pyqt window
    input_dialog.show()
    
    # allows the program to run and exit after completing
    sys.exit(app.exec_())  
