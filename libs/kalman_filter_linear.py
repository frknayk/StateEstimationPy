import numpy as np
import sys
from libs.commons.utilities import VerboseLevels, Bcolors

# Fancy printing class
bcolors = Bcolors()

class STATUS:
    ERR_WRONG_DIMENSIONS = -1
    ERR_WRONG_MATRIX_TYPE = -2
    NOT_INITATED = 0
    INITATED = 1

class KalmanFilter(object):
    def __init__(self,A, B, H, R, Q, x0, P0,
                num_measured_states,num_states=-1,num_inputs=-1,debug_level=VerboseLevels.NO_VERBOSE):
        """Constructor of linear kalman filter\
            
        Args:
            A (nd.array): System(dynamics) matrix
            B (nd.array): Input matrix
            H (nd.array): Measurement Matrix
            R (nd.array): Measurement noise covariance 
            Q (nd.array): Process noise covariance 
            x0 (nd.array): Initial state vector
            P0 (nd.array): Initial uncertainty
            num_measured_states (int) : Number of measured states. Used for dim check
        """
        # Dynamic (system) matrix 
        self.A = A.copy()
        # Input matrix
        self.B = B.copy()
        # Measurement matrix
        self.H = H.copy()
        # Measurement noise covariance 
        self.R = R.copy()
        # Process Noise Covariance
        self.Q = Q.copy()
        # State vector
        self.x = x0.copy()
        # Uncertainty matrix
        self.P = P0.copy()
        # Kalman gain
        self.K = None
        # Number of states
        self.dim_state = self.A.shape[0] if num_states == -1 else num_states
        # Number of inptus
        self.dim_input = self.B.shape[0] if num_inputs == -1 else num_inputs
        # Number of measured states
        self.dim_state_measured = num_measured_states
        self.verbosity = debug_level
        self.filter_status = STATUS.NOT_INITATED
        self.I = np.eye(self.dim_state)
        # Check if given matrices are valid as dimensions
        self.check_dimensions()

    def check_dimensions(self):
        """Check whether given fiare valid or not"""
        #TODO :Add positivity check for covariance related matrices !
        
        dim_check_arr = []
        dim_check_arr.append( self.check_dim(self.A, self.dim_state,self.dim_state, "A (dynamics matrix)") )
        dim_check_arr.append( self.check_dim(self.B, self.dim_input,1, "B (input matrix)") )
        dim_check_arr.append( self.check_dim(self.H, self.dim_state_measured,self.dim_state_measured, "H (measurement matrix)") )
        dim_check_arr.append( self.check_dim(self.R, self.dim_state_measured,self.dim_state_measured, "R(measurement noise covariance)") )
        dim_check_arr.append( self.check_dim(self.Q, self.dim_state,self.dim_state, "Q (process noise covariance matrix)") )
        dim_check_arr.append( self.check_dim(self.x, self.dim_state,1, "x (state vector)") )
        dim_check_arr.append( self.check_dim(self.P, self.dim_state,self.dim_state, "P (uncertainty matrix)") )

        self.filter_status = STATUS.INITATED

        # If not change filter status
        if False in dim_check_arr:
            self.filter_status = STATUS.ERR_WRONG_DIMENSIONS
            if self.verbosity > VerboseLevels.NO_VERBOSE:
                bcolors.print_error("Check matrix dimensions, filter could not be created!")
        if self.verbosity > VerboseLevels.NO_VERBOSE:
            bcolors.print_ok("Filter is initiated with given matrices.")

    @staticmethod
    def check_dim(matrix,dim_1, dim_2, matrix_name):
        if matrix.shape[0] != dim_1 and matrix.shape[1] != dim_2: 
            err_msg = "Wrong state dimension is given for : {0}".format(matrix_name)
            bcolors.print_warning(err_msg)
            return False
        return True

    def prediction(self, u_k):
        # Project the state ahead
        self.x = self.A*self.x + self.B*u_k
        # Project the error covariance ahead
        self.P = self.A*self.P*self.A.T + self.Q

    def correction(self, z_k):
        # Compute the kalman gain
        part_inv = (self.H*self.P*self.H.T) + self.R
        self.K = self.P*self.H.T * np.linalg.pinv(part_inv)
        # Update the estimate via measurement
        y = z_k - (self.H*self.x)
        self.x = self.x + (self.K*y)
        # Update the error covariance
        self.P = (self.I - (self.K*self.H))*self.P


if __name__ == '__main__':
    pass