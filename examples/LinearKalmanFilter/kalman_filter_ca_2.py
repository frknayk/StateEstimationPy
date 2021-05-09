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
A = np.matrix([
              [1.0,     0.0,        dt,         0.0,        0.5*dt*dt,     0],
              [0.0,     1.0,        0.0,        dt,         0,             0.5*dt*dt],
              [0.0,     0.0,        1.0,        0.0,        dt,            0],
              [0.0,     0.0,        0.0,        1.0,        0,             dt],
              [0.0,     0.0,        0.0,        0.0,        1,0,           0],
              [0.0,     0.0,        0.0,        0.0,        0,0,           1.0]
              ])

# Measurements matrix : Position and acceleration is measured!
H = np.matrix(
                [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                ])

# Initial uncertainty
p_init = 10
P = np.diag([p_init, p_init, p_init, p_init, p_init, p_init])

# Measurement noise covariance 
ra = 10.0**2
R = np.matrix([[ra, 0.0, 0.0, 0.0],
               [0.0, ra, 0.0, 0.0],
               [0.0, 0.0, ra, 0.0],
               [0.0, 0.0, 0.0, ra]])

# Process Noise Covariance
sa = 1.0
G = np.matrix([[1/2.0*dt**2],
               [1/2.0*dt**2],
               [dt],
               [dt],
               [1.0],
               [1.0]])
Q = G*G.T*sa**2

# Input matrix : Model is constant acceleration. 
B = np.matrix([[0],[0],[0],[0]])

############ Create Measurements ############
m = 100 # Measurements
# Acceleration
sa= 0.1 # Sigma for acceleration
ax= 0.0 # in X
ay= 0.0 # in Y
mx = np.array(ax+sa*np.random.randn(m))
my = np.array(ay+sa*np.random.randn(m))
measurements = np.vstack((mx,my))

# Initial measurement
x = np.matrix([[0.0,0.0]]).T

# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(16,9))
# plt.plot(range(m),mx, label='$a_x$')
# plt.plot(range(m),my, label='$a_y$')
# plt.ylabel('Acceleration')
# plt.title('Measurements')
# plt.ylim([-1, 1])
# plt.legend(loc='best',prop={'size':18})
# plt.show()

############ Create kalman filter ############
kalman_filter = KalmanFilter(A,B,H,R,Q,x,P,num_measured_states=2,debug_level=VerboseLevels.INFO_BASIC)

# Plotting tool
signal_name_list = ['x','y','vx','vy','ax','ay']
measurements_name_list = ['x','y','ax','ay']
units_name_list = ['m','m','m/s','m/s','m/s2','m/s2']
plotter = Plotter(6,4,signal_name_list,measurements_name_list,units_name_list)

############ Simulation loop ############
for n in range(len(measurements[0])):
    kalman_filter.prediction(0)
    z_k = measurements[:,n].reshape(2,1)
    kalman_filter.correction(z_k)
    # Log to plot
    plotter.log(kalman_filter.x,z_k)

# Plot results
plotter.plot()