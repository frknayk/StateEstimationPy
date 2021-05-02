import unittest
import numpy as np
from pyunitreport import HTMLTestRunner
from libs.kalman_filter_linear import KalmanFilter,VerboseLevels,STATUS

class Test_KalmanFilterLinear(unittest.TestCase):
    def test_1(self):
        # Determine dimensions
        dim_state = 4
        dim_input = 1
        dim_state_measured = 2
        verbose_level = VerboseLevels.NO_VERBOSE

        # Create matrices
        A = np.random.rand(dim_state,dim_state)
        B = np.random.rand(dim_input,1)
        H = np.random.rand(dim_state,dim_state_measured)
        R = np.random.rand(dim_state_measured,dim_state_measured)
        Q = np.random.rand(dim_state,dim_state)
        x0 = np.random.rand(dim_state,1)
        P0 = np.random.rand(dim_state,dim_state)

        # Create filter
        kf_filter = KalmanFilter(A,B,H,R,Q,x0,P0,dim_state_measured,dim_state,dim_input,debug_level=verbose_level)

        self.assertTrue(kf_filter.filter_status == STATUS.INITATED,"KF dimensions are not as expected")

    def test_2(self):
        # Determine dimensions
        dim_state = 4
        dim_input = 1
        dim_state_measured = 2
        verbose_level = VerboseLevels.NO_VERBOSE

        # Create matrices
        A = np.random.rand(dim_state,dim_state)
        B = np.random.rand(dim_input,1)
        H = np.random.rand(dim_state,dim_state_measured)
        R = np.random.rand(dim_state_measured,dim_state_measured)
        Q = np.random.rand(dim_state,dim_state)
        x0 = np.random.rand(dim_state,1)
        P0 = np.random.rand(dim_state,dim_state)

        # Create filter
        kf_filter = KalmanFilter(A,B,H,R,Q,x0,P0,dim_state_measured,debug_level=verbose_level)

        self.assertTrue(kf_filter.filter_status == STATUS.INITATED,"KF dimensions are not as expected")

    def test_3(self):
        # Determine dimensions
        dim_state = 4
        dim_input = 1
        dim_state_measured = 2
        verbose_level = VerboseLevels.NO_VERBOSE

        # Create matrices
        A = np.random.rand(dim_state,dim_state)
        B = np.random.rand(dim_input,1)
        H = np.random.rand(dim_state,dim_state_measured)
        R = np.random.rand(dim_state_measured,dim_state_measured)
        Q = np.random.rand(dim_state,dim_state)
        x0 = np.random.rand(dim_state,1)
        P0 = np.random.rand(dim_state,dim_state)

        # Create filter
        kf_filter = KalmanFilter(A,B,H,R,Q,x0,P0,dim_state_measured,dim_state,dim_input,debug_level=verbose_level)

        self.assertTrue(kf_filter.filter_status == STATUS.INITATED,"KF dimensions are not as expected")

if __name__ == "__main__":
    # unittest.main(testRunner=HTMLTestRunner(output='LinearKalmanFilter_Test_Results'))
    unittest.main()