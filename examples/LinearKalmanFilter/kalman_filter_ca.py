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

# Uncertainty matrix
P = np.diag([1000.0, 1000.0, 1000.0, 1000.0])

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
datafile = 'examples/data/2014-02-14-002-Data.csv'

temp = np.loadtxt(datafile, delimiter=',', unpack=True,skiprows=1)

s = True
# dlat = np.hstack((0.0, np.diff(latitude)))
# dlon = np.hstack((0.0, np.diff(longitude)))
# dt_s = np.hstack((0.0, np.diff(millis/1000.0)))

# dy = 111.32 * np.cos(latitude * np.pi/180.0) * dlon # in km
# dx = 111.32 * dlat # in km

# mx = np.cumsum(1000.0 * dx) # in m
# my = np.cumsum(1000.0 * dy) # in m
