import numpy as np
import matplotlib.pyplot as plt
from libs.kalman_filter_linear import KalmanFilter
from libs.commons.utilities import VerboseLevels
from libs.commons.plotter import Plotter
"""
Example source : 
https://github.com/balzer82/Kalman/blob/master/Kalman-Filter-CV.ipynb?create=1
"""


############ Create Kalman Filter Parameters ############
P = np.diag([1000.0, 1000.0, 1000.0, 1000.0])
# Time Step between Filter Steps
dt = 0.1
A = np.matrix([[1.0, 0.0, dt, 0.0],
              [0.0, 1.0, 0.0, dt],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])
H = np.matrix([[0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])
ra = 10.0**2
R = np.matrix([[ra, 0.0],
              [0.0, ra]])
sv = 8.8
Q = np.matrix([[(dt**4)/4, 0, (dt**3)/2, 0],
               [0, (dt**4)/4, 0, (dt**3)/2],
               [(dt**3)/2, 0, dt**2, 0],
               [0, (dt**3)/2, 0, dt**2]]) * sv**2
B = np.matrix([[0]])

############ Create Measurements ############
m = 200 # Measurements
vx= 20 # in X
vy= 10 # in Y
mx = np.array(vx+np.random.randn(m))
my = np.array(vy+np.random.randn(m))
measurements = np.vstack((mx,my))
x = np.matrix([[0.0, 0.0, 0.0, 0.0]]).T

############ Create kalman filter ############
kalman_filter = KalmanFilter(A,B,H,R,Q,x,P,num_measured_states=2,debug_level=VerboseLevels.INFO_BASIC)

# Plotting tool
signal_name_list = ['x','y','vx','vy']
measurements_name_list = ['vx','vy']
units_name_list = ['m','m','m/s','m/s']
plotter = Plotter(4,2,signal_name_list,measurements_name_list,units_name_list)

############ Simulation loop ############
for n in range(len(measurements[0])):
    kalman_filter.prediction(0)
    z_k = measurements[:,n].reshape(2,1)
    kalman_filter.correction(z_k)
    # Log to plot
    plotter.log(kalman_filter.x,z_k)

# Plot results
plotter.plot()

