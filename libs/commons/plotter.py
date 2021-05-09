import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    """Plotts logged states"""
    def __init__(self,num_state,num_measurement,states_name_list=[],measurements_name_list=[],unit_list=[]):
        """Plot logged states with given state names(states_name_list) and physical units(unit_list)

        Args:
            num_state (int): Number of states in measurement model
            num_measurement (int): Number of measured states
            states_name_list (list): List of state names
            unit_list (list): List of state units
        """
        self.num_state = num_state
        self.num_measurement = num_measurement
        self.states_name_list = states_name_list
        self.unit_list = unit_list
        self.meas_name_list = measurements_name_list
        self.ground_truth_name_list = []
        self.ground_truth_list = []
        self.states_list = []
        self.measurement_list = []

    def log(self, kf_state, measurement):
        state = []
        for s in kf_state:
            state.append(s.item(0))
        meas = []
        for m in measurement:
            meas.append(m[0])
        self.states_list.append(state)
        self.measurement_list.append(meas)

    def set_grount_truth_data(self, signals, signals_name_list):
        self.ground_truth_name_list = signals_name_list
        self.ground_truth_list = signals

    def plot(self):
        fig, axs = plt.subplots(self.num_state,figsize=(16,10))
        fig.suptitle('Measurements')

        states_list_np = np.asarray(self.states_list)
        measurement_list_np = np.asarray(self.measurement_list)
        ground_truth_list_np = np.asarray(self.ground_truth_list)
        for x in range(self.num_state):
            plt_label_state = self.states_name_list[x]
            plt_label_unit = self.unit_list[x]
            
            # Total number of measured signals
            num_samples = range(states_list_np.shape[0])

            # Plot tracked state
            axs[x].step(num_samples,states_list_np[:,x], label=plt_label_state)

            # Plot measured state
            if plt_label_state in self.meas_name_list:
                # Index of state
                idx = self.meas_name_list.index(plt_label_state)
                plt_label_meas = plt_label_state + "(measured)"
                axs[x].step(num_samples, measurement_list_np[:,idx], label=plt_label_meas)

            # Plot ground-truth state
            if len(self.ground_truth_name_list)>0 and plt_label_state in self.ground_truth_name_list:
                # Index of state
                idx = self.ground_truth_name_list.index(plt_label_state)
                plt_label_meas = plt_label_state + "(gt)"
                axs[x].step(num_samples, ground_truth_list_np[idx,:], label=plt_label_meas)

            axs[x].legend(loc=2)
            axs[x].set_ylabel(plt_label_unit)
        plt.show()

