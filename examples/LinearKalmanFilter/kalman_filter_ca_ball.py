import numpy as np
from libs.kalman_filter_linear import KalmanFilter
from libs.commons.utilities import VerboseLevels
from libs.commons.plotter import Plotter

P = 100.0*np.eye(9)
dt = 0.01 # Time Step between Filter Steps

A = np.matrix([[1.0, 0.0, 0.0, dt, 0.0, 0.0, 1/2.0*dt**2, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 1/2.0*dt**2, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 1/2.0*dt**2],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

rp = 1.0**2  # Noise of Position Measurement
R = np.matrix([[rp, 0.0, 0.0],
               [0.0, rp, 0.0],
               [0.0, 0.0, rp]])


sj = 0.1

Q = np.matrix([[(dt**6)/36, 0, 0, (dt**5)/12, 0, 0, (dt**4)/6, 0, 0],
               [0, (dt**6)/36, 0, 0, (dt**5)/12, 0, 0, (dt**4)/6, 0],
               [0, 0, (dt**6)/36, 0, 0, (dt**5)/12, 0, 0, (dt**4)/6],
               [(dt**5)/12, 0, 0, (dt**4)/4, 0, 0, (dt**3)/2, 0, 0],
               [0, (dt**5)/12, 0, 0, (dt**4)/4, 0, 0, (dt**3)/2, 0],
               [0, 0, (dt**5)/12, 0, 0, (dt**4)/4, 0, 0, (dt**3)/2],
               [(dt**4)/6, 0, 0, (dt**3)/2, 0, 0, (dt**2), 0, 0],
               [0, (dt**4)/6, 0, 0, (dt**3)/2, 0, 0, (dt**2), 0],
               [0, 0, (dt**4)/6, 0, 0, (dt**3)/2, 0, 0, (dt**2)]]) *sj**2

B = np.matrix([[0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0]])
u = 0.0

I = np.eye(9)

Hz = 100.0 # Frequency of Vision System
dt = 1.0/Hz
T = 1.0 # s measuremnt time
m = int(T/dt) # number of measurements

px= 0.0 # x Position Start
py= 0.0 # y Position Start
pz= 1.0 # z Position Start

vx = 10.0 # m/s Velocity at the beginning
vy = 0.0 # m/s Velocity
vz = 0.0 # m/s Velocity

c = 0.1 # Drag Resistance Coefficient
d = 0.9 # Damping

Xr=[]
Yr=[]
Zr=[]
for i in range(int(m)):
    accx = -c*vx**2  # Drag Resistance
    
    vx += accx*dt
    px += vx*dt

    accz = -9.806 + c*vz**2 # Gravitation + Drag
    vz += accz*dt
    pz += vz*dt
    
    if pz<0.01:
        vz=-vz*d
        pz+=0.02
    if vx<0.1:
        accx=0.0
        accz=0.0
        
    Xr.append(px)
    Yr.append(py)
    Zr.append(pz)

### Add Noise to the Real Position
sp= 0.1 # Sigma for position noise

Xm = Xr + sp * (np.random.randn(m))
Ym = Yr + sp * (np.random.randn(m))
Zm = Zr + sp * (np.random.randn(m))
measurements = np.vstack((Xm,Ym,Zm))

# Initial state
x = np.matrix([0.0, 0.0, 1.0, 10.0, 0.0, 0.0, 0.0, 0.0, -9.81]).T

############ Create kalman filter ############
kalman_filter = KalmanFilter(A,B,H,R,Q,x,P,num_measured_states=3,debug_level=VerboseLevels.INFO_BASIC)

# Plotting tool
signal_name_list = ['x','y','z','vx','vy','vz','ax','ay','az']
measurements_name_list = ['x','y','z']
units_name_list = ['m','m','m','m/s','m/s','m/s','m/s2','m/s2','m/s2']
plotter = Plotter(9,3,signal_name_list,measurements_name_list,units_name_list)

############ Simulation loop ############
for n in range(len(measurements[0])):
    kalman_filter.prediction(0)
    z_k = measurements[:,n].reshape(3,1)
    kalman_filter.correction(z_k)
    # Log to plot
    plotter.log(kalman_filter.x,z_k)

# Plot results
plotter.plot()