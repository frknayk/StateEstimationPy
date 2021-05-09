import numpy as np
from libs.kalman_filter_linear import KalmanFilter
from libs.commons.utilities import VerboseLevels
from libs.commons.plotter import Plotter

"""
Example source : 
https://github.com/balzer82/Kalman/blob/master/Kalman-Filter-CA-RealMeasurements.ipynb?create=1

Situation covered: You have an acceleration sensor (a_x, a_y) and a position sensor (x,y) 
    states = [x,y,x',y',x'',y'']
    measurements = [x'',y'']
"""


############ Create Kalman Filter Parameters ############

# Time Step between Filter Steps
dt = 0.1

# System matrix
A = np.matrix([[1.0, 0.0, dt, 0.0, 1/2.0*dt**2, 0.0],
              [0.0, 1.0, 0.0, dt, 0.0, 1/2.0*dt**2],
              [0.0, 0.0, 1.0, 0.0, dt, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0, dt],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

# Measurements matrix : Position and acceleration is measured!
H = np.matrix([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

# Initial uncertainty
P = np.matrix([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 10.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 10.0]])*100

# Measurement noise covariance 
ra = 10.0**2
R = np.matrix([[ra, 0.0],
               [0.0, ra]])

# Process Noise Covariance
sj = 0.1
Q = np.matrix([[(dt**6)/36, 0, (dt**5)/12, 0, (dt**4)/6, 0],
               [0, (dt**6)/36, 0, (dt**5)/12, 0, (dt**4)/6],
               [(dt**5)/12, 0, (dt**4)/4, 0, (dt**3)/2, 0],
               [0, (dt**5)/12, 0, (dt**4)/4, 0, (dt**3)/2],
               [(dt**4)/6, 0, (dt**3)/2, 0, (dt**2),0],
               [0, (dt**4)/6, 0, (dt**3)/2, 0, (dt**2)]]) *sj**2

# Input matrix : Model is constant acceleration. 
B = np.zeros((6,1))

############ Create Measurements ############

# Size of measurements
m = 100

# Generate ground truth data
accelx_min = 0
accelx_max = 2
accely_min = 0.0001
accely_max = 0.005
accelx_gt = np.linspace(accelx_min, accelx_max, m)
accely_gt = np.linspace(accely_min, accely_max, m)
velx_gt = np.cumsum(accelx_gt)
vely_gt = np.cumsum(accely_gt)
posx_gt = np.cumsum(velx_gt)
posy_gt = np.cumsum(vely_gt)

# Generate sensory data
accelx = accelx_gt + np.random.normal(0, 0.01,m)
accely = accely_gt + np.random.normal(0, 0.01,m)
velx = np.cumsum(accelx)
vely = np.cumsum(accely)
posx = np.cumsum(velx)
posy = np.cumsum(vely)

import matplotlib.pyplot as plt
def plt_pos():
    fig = plt.figure(figsize=(16,9))
    labels_gt = '$ground truth$'
    labels_noisy = '$sensor output$'
    plt.scatter(posx_gt,posy_gt, label=labels_gt)
    plt.scatter(posx,posy, label=labels_noisy)
    plt.title('Position')
    plt.legend(loc='best',prop={'size':18})

def plt_vel():
    fig = plt.figure(figsize=(16,9))
    labels_gt = '$ground truth$'
    labels_noisy = '$sensor output$'
    plt.scatter(velx_gt,vely_gt, label=labels_gt)
    plt.scatter(velx,vely, label=labels_noisy)
    plt.title('Velocity')
    plt.legend(loc='best',prop={'size':18})

def plt_accel():
    fig = plt.figure(figsize=(16,9))
    labels_gt = '$ground truth$'
    labels_noisy = '$sensor output$'
    plt.scatter(accelx_gt,accely_gt, label=labels_gt)
    plt.scatter(accelx,accely, label=labels_noisy)
    plt.title('Acceleration')
    plt.legend(loc='best',prop={'size':18})  

# plt_pos()
# plt_vel()
# plt_accel()
# plt.show()

# Acceleration
measurements = np.vstack((accelx,accely))

# Initial measurement
x = np.matrix([[posx[0], posy[0], velx[0], vely[0], accelx[0], accelx[0]]]).T

############ Create kalman filter ############
kalman_filter = KalmanFilter(A,B,H,R,Q,x,P,num_measured_states=2,debug_level=VerboseLevels.INFO_BASIC)

# Plotting tool
signal_name_list = ['x','y','vx','vy','ax','ay']
measurements_name_list = ['ax','ay']
units_name_list = ['m','m','m/s','m/s','m/s2','m/s2']
plotter = Plotter(6,4,signal_name_list,measurements_name_list,units_name_list)

############ Simulation loop ############
for n in range(len(measurements[0])):
    kalman_filter.prediction(0)
    z_k = measurements[:,n].reshape(2,1)
    kalman_filter.correction(z_k)
    # Log to plot
    plotter.log(kalman_filter.x,z_k)

# Set ground truth data
signal_gt_name_list = ['x','y','vx','vy','ax','ay']
gt_list = [posx_gt,posy_gt,velx_gt,vely_gt,accelx_gt,accely_gt]
plotter.set_grount_truth_data(gt_list,signal_gt_name_list)

# Plot results
plotter.plot()