#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 16:25:05 2023

@author: adityapanigrahi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from __main__ import *
for key, value in inputvaluesdict.items():
    exec(f"{key} = {value}")
    print(f"{key} = {value}")

print("classcalled.py is running...")

 #%%
 #!/usr/bin/env python3
 # -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 16:25:05 2023

@author: adityapanigrahi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class LoadAnalysis:
    
    def __init__(self, E, y_loc_master, I_master, F_l_1, F_l_2, F_l_3, F_l_4, max_training_load, min_training_load, increment_training, 
             max_testing_load, min_testing_load, increment_testing, noise_level, BL, sen_locs, alpha, 
             max_training_temp, min_training_temp, max_testing_temp, min_testing_temp):
        
        self.E = E
        self.y_loc = y_loc
        self.I = I
        self.F_l_1 = F_l_1
        self.F_l_2 = F_l_2
        self.F_l_3 = F_l_3
        self.F_l_4 = F_l_4
        self.max_training_load = max_training_load
        self.min_training_load = min_training_load
        self.increment_training = increment_training
        self.max_testing_load = max_testing_load
        self.min_testing_load = min_testing_load
        self.increment_testing = increment_testing
        self.noise_level = noise_level
        self.BL = BL
        self.sen_locs = sen_locs
        self.alpha = alpha
        self.max_training_temp = max_training_temp
        self.min_training_temp = min_training_temp
        self.max_testing_temp = max_testing_temp
        self.min_testing_temp = min_testing_temp
        
        
        
    def generate_master_force(self,min_training_load,max_training_load,increment_training):
        load1, load2, load3, load4 = np.meshgrid(np.arange(min_training_load, max_training_load, increment_training),
                                                np.arange(min_training_load, max_training_load, increment_training),
                                                np.arange(min_training_load, max_training_load, increment_training),
                                                np.arange(min_training_load, max_training_load, increment_training))
        
        master_force = np.column_stack((load1.ravel(), load2.ravel(), load3.ravel(), load4.ravel()))
        
        return master_force
    
    def calculate_strain(self, master_force):
        
        sen_locs_mat = (BL-sen_locs.values)/1000
        moment1_training =[]
        moment2_training =[]
        moment3_training =[]
        moment4_training =[]
        F_l = [F_l_1, F_l_2, F_l_3, F_l_4]
        right_load =[]
        left_load =[]

        for i in F_l:
            
            if i > 0:
                right_load.append(i)
            else:
                left_load.append(i)
                
        
        for i, sen_loc in enumerate(sen_locs_mat):
            
            if sen_loc>0:
                
                if sen_loc > right_load[0]:
                    moment1 = np.zeros((master_force[:,0].shape))
                else:
                    moment1 = master_force[:,0]*(right_load[0]-sen_loc)
                moment1_training.append(moment1)    
                
                
                if sen_loc > right_load[1]:
                    moment2 = np.zeros((master_force[:,1].shape))
                else:
                    moment2 = master_force[:,1]*(right_load[1]-sen_loc)
                moment2_training.append(moment2)    
                
                if sen_loc > right_load[2]:
                    moment3 = np.zeros((master_force[:,2].shape))
                else:
                    moment3 = master_force[:,2]*(right_load[2]-sen_loc)
                moment3_training.append(moment3) 
        
        for i,sen in enumerate(sen_locs_mat):
            
            if sen < 0:
               
                if sen < left_load[0]:
                    moment4 =np.zeros((master_force[:,3].shape))
                else:
                    moment4 = master_force[:,3]*(left_load[0]-sen)
                moment4_training.append(moment4)
    
        m1 = np.array(moment1_training)      
        m2 = np.array(moment2_training)       
        m3 = np.array(moment3_training)
        
        master_training_moment_right = m1+m2+m3
        master_training_moment_left = np.array(moment4_training)
        
        master_moment = np.concatenate((master_training_moment_left,master_training_moment_right))
        master_moment = master_moment.T
        
        master_training_strain = []
        
        y_loc_master = np.array(y_loc)
        I_master = np.array(I.values)

        for i in range(0,8):
            strain_training = (master_moment[:,i]*y_loc_master[i])/(E*I_master[i])
            master_training_strain.append(strain_training)
            
        master_training_strain = (np.array(master_training_strain)*(1e6))
        
        return master_training_strain
    
    def generate_C(self, master_training_strain, master_force):
        master_training_force = np.array(master_force).T
        C = master_training_strain @ master_training_force.T @ np.linalg.inv(master_training_force @ master_training_force.T)
        x_labels = [1,2,3,4]
        y_labels = np.arange(1,sen_locs.shape[0]+1)
        B = np.linalg.inv(C.T @ C) @ C.T
        sns.heatmap(B.T, cmap="YlGnBu", fmt="d", xticklabels=x_labels, yticklabels=y_labels)
        

        plt.xlabel('Load')
        plt.ylabel('Sensor No')
        plt.show()
        return C
    
    
    def Validation(self,master_force, master_training_strain,C):
        
    
        val_force = np.linalg.inv(C.T @ C) @ C.T @ master_training_strain
        val_force = val_force.T
        diff=[]
       
        
        
        all_diffs = []

        val_force = np.linalg.inv(C.T @ C) @ C.T @ master_training_strain
        val_force = val_force.T
        diff=[]
       
        all_diffs = []
        print('\n')
        print('Validation')

        for i in range(4):
            
             diff = abs(val_force[:, i] - master_force[:, i])
             all_diffs.append(diff)
             average_diff = np.mean(diff)
             highest_diff = np.max(diff)
             lowest_diff = np.min(diff)
             print(f"Load {i+1} (N) - Average: {average_diff:.2f}, Highest: {highest_diff:.2f}, Lowest: {lowest_diff:.2f}")
        print('\n')

        
        # Create a box and whisker plot
        plt.figure()
        plt.boxplot(all_diffs, labels=['Load 1', 'Load 2', 'Load 3', 'Load 4'])
        
        plt.xlabel('Load')
        plt.ylabel('Differences (N)')
        plt.show()
        
        #Plotfor all Cases 
        x_axis_test = np.arange(1, master_force.shape[0]+1)
        #Whisker Plot
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Validation')
        # Load 1
        plt.subplot(2, 2, 1)
        plt.scatter(x_axis_test, val_force[:, 0], color='k', marker='o', s=155, label='Predicted Load (N)')
        plt.scatter(x_axis_test, master_force[:, 0], color='r', marker='X', s=55, label='Actual Load (N)')
        plt.xlabel('Load Case No.')
        plt.ylabel('Load (N)')
        plt.title('Load 1')
        plt.legend()
        
        # Load 2
        plt.subplot(2, 2, 2)
        plt.scatter(x_axis_test, val_force[:, 1], color='k', marker='o', s=155, label='Predicted Load (N)')
        plt.scatter(x_axis_test, master_force[:, 1], color='r', marker='X', s=55, label='Actual Load (N)')
        plt.xlabel('Load Case No.')
        plt.ylabel('Load (N)')
        plt.title('Load 2')
        plt.legend()
        
        # Load 3
        plt.subplot(2, 2, 3)
        plt.scatter(x_axis_test, val_force[:, 2], color='k', marker='o', s=155, label='Predicted Load (N)')
        plt.scatter(x_axis_test, master_force[:, 2], color='r', marker='X', s=55, label='Actual Load (N)')
        plt.xlabel('Load Case No.')
        plt.ylabel('Load (N)')
        plt.title('Load 3')
        plt.legend()
        
        # Load 4
        plt.subplot(2, 2, 4)
        plt.scatter(x_axis_test, val_force[:, 3], color='k', marker='o', s=155, label='Predicted Load (N)')
        plt.scatter(x_axis_test, master_force[:, 3], color='r', marker='X', s=55, label='Actual Load (N)')
        plt.xlabel('Load Case No.')
        plt.ylabel('Load (N)')
        plt.title('Load 4')
        plt.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for the title
        plt.show()
            
    def generate_master_force_test(self,min_testiing_load,max_testing_load,increment_testing):
        
        load1_test, load2_test, load3_test, load4_test = np.meshgrid(np.arange(min_testiing_load,max_testing_load,increment_testing),
                                                 np.arange(min_testiing_load,max_testing_load,increment_testing),
                                                 np.arange(min_testiing_load,max_testing_load,increment_testing),
                                                 np.arange(min_testiing_load,max_testing_load,increment_testing))

        master_force_test = np.column_stack((load1_test.ravel(), load2_test.ravel(), load3_test.ravel(), load4_test.ravel()))  

        np.random.shuffle(master_force_test)
        
        return(master_force_test)
        
    def calculate_strain_test(self, master_force_test):
        
        F_l = [F_l_1, F_l_2, F_l_3, F_l_4]
        sen_locs_mat = (BL-sen_locs.values)/1000
        right_load =[]
        left_load =[]
        for i in F_l:
            
            if i > 0:
                right_load.append(i)
            else:
                left_load.append(i)
        
       
        moment1_testing =[]
        moment2_testing =[]
        moment3_testing =[]
        moment4_testing =[]

        for i, sen_loc in enumerate(sen_locs_mat):
            
            if sen_loc>0:
                
                if sen_loc > right_load[0]:
                    moment1t = np.zeros((master_force_test[:,0].shape))
                else:
                    moment1t = master_force_test[:,0]*(right_load[0]-sen_loc)
                moment1_testing.append(moment1t)    
                
                
                if sen_loc > right_load[1]:
                    moment2t = np.zeros((master_force_test[:,1].shape))
                else:
                    moment2t = master_force_test[:,1]*(right_load[1]-sen_loc)
                moment2_testing.append(moment2t)    
                
                if sen_loc > right_load[2]:
                    moment3t = np.zeros((master_force_test[:,2].shape))
                else:
                    moment3t = master_force_test[:,2]*(right_load[2]-sen_loc)
                moment3_testing.append(moment3t) 
                
        for i,sen in enumerate(sen_locs_mat):
            
            if sen < 0:
               
                if sen < left_load[0]:
                    moment4t =np.zeros((master_force_test[:,3].shape))
                else:
                    moment4t = master_force_test[:,3]*(left_load[0]-sen)
                moment4_testing.append(moment4t) 
        
        m1t = np.array(moment1_testing)      
        m2t = np.array(moment2_testing)       
        m3t = np.array(moment3_testing)

        master_testing_moment_right = m1t+m2t+m3t
        
        
        master_testing_moment_left = np.array(moment4_testing)
        master_moment_t = np.concatenate((master_testing_moment_left,master_testing_moment_right))
        master_moment_t = master_moment_t.T
        y_loc_master = np.array(y_loc)
        I_master = np.array(I.values)
        master_testing_strain = []
        for i in range(0,sen_locs_mat.shape[0]):
            strain_testing = (master_moment_t[:,i]*y_loc_master[i])/(E*I_master[i])
            master_testing_strain.append(strain_testing)
            
        master_testing_strain = (np.array(master_testing_strain)*(1e6))
        
        return master_testing_strain
    
    def Validation_test(self,master_force_test, master_testing_strain,C):
        
    
        val_force_test = np.linalg.inv(C.T @ C) @ C.T @ master_testing_strain
        val_force_test = val_force_test.T
        diff=[]
        val_force = np.linalg.inv(C.T @ C) @ C.T @ master_testing_strain
        val_force = val_force.T
        diff=[]
       
        
        
        all_diffs = []
        print('Test Cases without Noise')

        for i in range(4):
            
        
            diff = abs(val_force_test[:, i] - master_force_test[:, i])
            all_diffs.append(diff)
            
            average_diff = np.mean(diff)
            highest_diff = np.max(diff)
            lowest_diff = np.min(diff)
            
        
            print(f"Load {i+1} (N) - Average: {average_diff:.2f}, Highest: {highest_diff:.2f}, Lowest: {lowest_diff:.2f}")
        print('\n')
        # Create a box and whisker plot
        plt.figure()
        plt.boxplot(all_diffs, labels=['Load 1', 'Load 2', 'Load 3', 'Load 4'])
        
        plt.xlabel('Load')
        plt.ylabel('Differences (N)')
        plt.show()
        
        #Plot for all Cases 
        x_axis_test = np.arange(1, master_force_test.shape[0]+1)
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Testing with no Noise')
        # Load 1
        plt.subplot(2, 2, 1)
        plt.scatter(x_axis_test, val_force_test[:, 0], color='k', marker='o', s=155, label='Predicted Load (N)')
        plt.scatter(x_axis_test, master_force_test[:, 0], color='r', marker='X', s=55, label='Actual Load (N)')
        plt.xlabel('Load Case No.')
        plt.ylabel('Load (N)')
        plt.title('Load 1')
        plt.legend()
        
        # Load 2
        plt.subplot(2, 2, 2)
        plt.scatter(x_axis_test, val_force_test[:, 1], color='k', marker='o', s=155, label='Predicted Load (N)')
        plt.scatter(x_axis_test, master_force_test[:, 1], color='r', marker='X', s=55, label='Actual Load (N)')
        plt.xlabel('Load Case No.')
        plt.ylabel('Load (N)')
        plt.title('Load 2')
        plt.legend()
        
        # Load 3
        plt.subplot(2, 2, 3)
        plt.scatter(x_axis_test, val_force_test[:, 2], color='k', marker='o', s=155, label='Predicted Load (N)')
        plt.scatter(x_axis_test, master_force_test[:, 2], color='r', marker='X', s=55, label='Actual Load (N)')
        plt.xlabel('Load Case No.')
        plt.ylabel('Load (N)')
        plt.title('Load 3')
        plt.legend()
        
        # Load 4
        plt.subplot(2, 2, 4)
        plt.scatter(x_axis_test, val_force_test[:, 3], color='k', marker='o', s=155, label='Predicted Load (N)')
        plt.scatter(x_axis_test, master_force_test[:,3], color='r', marker='X', s=55, label='Actual Load (N)')
        plt.xlabel('Load Case No.')
        plt.ylabel('Load (N)')
        plt.title('Load 4')
        plt.legend()
        

        return val_force_test
    
    def Validation_test_noise(self,noise,master_force_test, master_testing_strain,C):
        
        std = noise
        noise_strain_test = np.random.normal(0, std, master_testing_strain.shape) + master_testing_strain
        testing_load_noise = np.linalg.inv(C.T @ C) @ C.T @ noise_strain_test
        testing_load_noise = testing_load_noise.T
    
        all_diffs = []
        print('Test Cases with Noise')


        for i in range(4):
            
        
            diff = abs(testing_load_noise[:, i] - master_force_test[:, i])
            all_diffs.append(diff)
            
            average_diff = np.mean(diff)
            highest_diff = np.max(diff)
            lowest_diff = np.min(diff)
            
        
            print(f"Load {i+1} (N) - Average: {average_diff:.2f}, Highest: {highest_diff:.2f}, Lowest: {lowest_diff:.2f}")
        print('\n')
         # Create a box and whisker plot
        plt.figure()
        plt.boxplot(all_diffs, labels=['Load 1', 'Load 2', 'Load 3', 'Load 4'])
        
        plt.xlabel('Load')
        plt.ylabel('Differences (N)')
        plt.show()
        
        
        
        #Plot for all Cases 
        x_axis_test = np.arange(1, master_force_test.shape[0]+1)
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Testing with Noise')
        # Load 1
        plt.subplot(2, 2, 1)
        plt.scatter(x_axis_test, testing_load_noise[:, 0], color='k', marker='o', s=155, label='Predicted Load (N)')
        plt.scatter(x_axis_test, master_force_test[:, 0], color='r', marker='X', s=55, label='Actual Load (N)')
        plt.xlabel('Load Case No.')
        plt.ylabel('Load (N)')
        plt.title('Load 1')
        plt.legend()
        
        # Load 2
        plt.subplot(2, 2, 2)
        plt.scatter(x_axis_test, testing_load_noise[:, 1], color='k', marker='o', s=155, label='Predicted Load (N)')
        plt.scatter(x_axis_test, master_force_test[:, 1], color='r', marker='X', s=55, label='Actual Load (N)')
        plt.xlabel('Load Case No.')
        plt.ylabel('Load (N)')
        plt.title('Load 2')
        plt.legend()
        
        # Load 3
        plt.subplot(2, 2, 3)
        plt.scatter(x_axis_test, testing_load_noise[:, 2], color='k', marker='o', s=155, label='Predicted Load (N)')
        plt.scatter(x_axis_test, master_force_test[:, 2], color='r', marker='X', s=55, label='Actual Load (N)')
        plt.xlabel('Load Case No.')
        plt.ylabel('Load (N)')
        plt.title('Load 3')
        plt.legend()
        
        # Load 4
        plt.subplot(2, 2, 4)
        plt.scatter(x_axis_test, testing_load_noise[:, 3], color='k', marker='o', s=155, label='Predicted Load (N)')
        plt.scatter(x_axis_test, master_force_test[:, 3], color='r', marker='X', s=55, label='Actual Load (N)')
        plt.xlabel('Load Case No.')
        plt.ylabel('Load (N)')
        plt.title('Load 4')
        plt.legend()

        return testing_load_noise


    def Thermal_C(self,calculate_strain,master_force,alpha,max_training_temp,min_training_temp):
        
        #UPDATE THE INPUT MATRIX
        alpha =alpha
        temp_change = (max_training_temp - min_training_temp) * np.random.rand(1, master_force.shape[0]) + min_training_temp
        temp_change = temp_change.T
        updated_input_matrix = np.hstack((master_force, temp_change))
        
        
        #APPLY THERMAL STRAIN
        temp_change_t = temp_change.T
        thermal_strain =[]
        thermal_strain = np.zeros_like(calculate_strain)

        for i in range(calculate_strain.shape[1]):
            thermal_strain[:,i] = (alpha * temp_change_t[:,i]) + calculate_strain[:,i]
            
        #Calculate the C matrix
        
        updated_input_matrix = updated_input_matrix.T
        
        C_thermal = thermal_strain@updated_input_matrix.T@np.linalg.inv(updated_input_matrix@updated_input_matrix.T)
    
        
        return temp_change_t, updated_input_matrix, thermal_strain,C_thermal
    
    def Thermal_test(self,C_thermal,master_force_test,calculate_strain_test,alpha,max_testing_temp,min_testing_temp):
        
        alpha =alpha
        temp_change = (max_testing_temp - min_testing_temp) * np.random.rand(1, master_force_test.shape[0]) + min_testing_temp
        temp_change = temp_change.T
        updated_output_matrix = np.hstack((master_force_test, temp_change))
        
        #APPLY THERMAL STRAIN
        temp_change_t = temp_change.T
        thermal_strain_test =[]
        thermal_strain_test = np.zeros_like(calculate_strain_test)

        for i in range(calculate_strain_test.shape[1]):
            thermal_strain_test[:,i] = (alpha * temp_change_t[:,i]) + calculate_strain_test[:,i]
            
            
        predicted_thermal_load = np.linalg.inv(C_thermal.T@C_thermal) @ C_thermal.T @ thermal_strain_test
        
        
        
        #Plot the thermal Load
        plt.figure()
        
        x_axis_test = (np.arange(1, master_force_test.shape[0]+1)).T
        
        plt.scatter(x_axis_test, predicted_thermal_load[4,:],color='k', marker='o', s=155, label='Predicted Thermal Load (deg C)')
        plt.scatter(x_axis_test.T, updated_output_matrix[:,4],color='r', marker='X', s=55, label='Actual Thermal Load (deg C)')
        plt.xlabel('Load Case No.')
        plt.ylabel('Temperature Change(deg C)')
        plt.title('Predicting Thermal Load')
        plt.legend()
        return  thermal_strain_test, updated_output_matrix, predicted_thermal_load


    def noise_study(self,C):
        
        std_choice_values = np.arange(0.1, 5.1, 0.1)

        
        diagonal_values = np.zeros((4, len(std_choice_values)))
        
        # Calculate var_force for each std_choice value and store the diagonal values
        for i in range(len(std_choice_values)):
            std_choice = std_choice_values[i]
            var_force_matrix = std_choice**2 * np.linalg.inv(np.dot(C.T, C))
            diagonal_values[:, i] = np.diag(var_force_matrix)
        
        # Plotting the results
        plt.figure(figsize=(10, 8))
        
        for j in range(4):
            plt.subplot(2, 2, j + 1)
            plt.plot(std_choice_values, diagonal_values[j, :])
            plt.title(f'Load {j + 1}')
            plt.xlabel('Noise Level (Strain) (με)')
            plt.ylabel(f'Variance(s²(Load{j + 1})), N²')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        


#%% INPUT PARAMETERS

#Material Set-up
# E = 70e9; #Young's Modulus (Pa)
#Geometry Set-up
y_loc = pd.read_excel('forceIC3x32.xlsx',sheet_name='y_loc',header = None,usecols='A', nrows=8)/1000; # Import vertical distance between neutral axis and sensor location [m] 
I = pd.read_excel('forceIC3x32.xlsx',sheet_name='newI',header = None,usecols='D', nrows=8); #Area Moment inertia at each section [m^4]

#Upload Sensor Location (from a origin other than Boundary Location)
sen_locs = pd.read_excel('forceIC3x32.xlsx',sheet_name='newI',header = None,usecols='E', nrows=8)

#Input the Boundary location (Use the same origin as the one used in the sensor location)

# BL = 986.5268; # Boundary condition location in mm


# #Load set-up and BL
# F_l_1 = 961.3471/1000; #Load 1 Location [m]
# F_l_2 = 620.9414/1000; #Load 2 Location [m]
# F_l_3 = 544.3625/1000; #Load 3 Location [m]
# F_l_4 = -438.9258/1000; #Load 4 Location [m]


# #Training Setup
# max_training_load = 150; #Maximum Load (N)
# min_training_load = -151; #Minimum Load (N)
# increment_training = 50; #This decides the size of the training case, lower it is the higher the training size
# #Testing Set-up
# max_testing_load = 550; 
# min_testing_load = -550;
# increment_testing=350;
# noise_level = 1; # Noise level in micro-strain


# # Temperature

# alpha = 24  #Thermal Conductivity in micro-scale
# max_training_temp = 10 #Deg C
# min_training_temp = -10  #Deg C


# max_testing_temp = 20 #deg C
# min_testing_temp = -20 #deg C
#%% 

d = LoadAnalysis(E, y_loc, I, F_l_1, F_l_2, F_l_3, F_l_4, max_training_load, min_training_load, increment_training, max_testing_load, min_testing_load, increment_testing, noise_level, BL, sen_locs,alpha,max_training_temp,min_training_temp,max_testing_temp,min_training_temp)

forcce = d.generate_master_force

master_force_result = d.generate_master_force(d.min_training_load, d.max_training_load, d.increment_training)

master_straintraining = d.calculate_strain(master_force_result)

c_matrix = d.generate_C(master_straintraining, master_force_result)

val = d.Validation( master_force_result,master_straintraining,c_matrix)

test_force = d.generate_master_force_test(min_testing_load,max_testing_load,increment_testing)

test_strain = d.calculate_strain_test(test_force)


f_pred = d.Validation_test(test_force, test_strain, c_matrix)


test_noise_load = d.Validation_test_noise(noise_level,test_force, test_strain, c_matrix)


thermal_training,new_input,thermal_strain,Cthermal = d.Thermal_C(master_straintraining,master_force_result,alpha,max_training_temp,min_training_temp)


test_thermal_strain, updated_out, predicted_therm = d.Thermal_test(Cthermal, test_force, test_strain, alpha, max_testing_temp, min_testing_temp)


variance_study = d.noise_study(c_matrix)