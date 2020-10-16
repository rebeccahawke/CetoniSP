import os
from time import time, time_ns, sleep
import numpy as np
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets

from equip import CetoniSP
from data_handling import fit_sinusoid

app = QtWidgets.QApplication([])

'''
from msl.equipment import Config

config = r'config.xml'
cfg = Config(config)               # loads cfg file
db = cfg.database()           # loads database
equipment = db.equipment      # loads subset of database with equipment being used


for record in equipment():
    print(record)
'''


class Experimenter(object):

    def __init__(self):

        self.sp = None
        self.t_data = []        # time in seconds
        self.fl_data = []       # syringe plunger fill level in unit set (mL)

    def initialise_pump(self, vol=None, flow=None):

        self.sp = CetoniSP(r'C:\Users\Public\Documents\QmixElements\Projects\default_project\Configurations\LowPressure_wValve')
        self.sp.look_for_devices()
        self.sp.connect_to_pump("neMESYS_Low_Pressure_1_Pump")

        ### essential start-up routine ###
        self.sp.configure_syringe(inner_dia=23.0329, stroke=60.0)  # 25 mL syringe
        self.sp.set_units('mL', 'mL/min')
        self.sp.calibrate_pump()  # Only run this command with no syringe installed

        self.sp.install_syringe()

        if vol is not None:
            self.set_syringe_level(vol, flow=flow)

    def set_syringe_level(self, vol, flow=None):

        self.sp.set_syringe_level(vol, flow=flow)

    def clear_data(self):
        self.t_data = []
        self.fl_data = []

    def update_data(self, ts, fls):
        for t in ts:
            self.t_data.append(t)
        for fl in fls:
            self.fl_data.append(fl)

    def collect_steady_state_data(self, wait_time, time_int):
        print("Collecting steady-state data for {} seconds".format(wait_time))

        nmeas = int(np.ceil(wait_time/time_int*1000))
        for i in range(nmeas):
            t = time_ns() / 1e9
            fl = self.sp.syringe_fill_level
            self.t_data.append(t)
            self.fl_data.append(fl)
            sleep(time_int/1000)

    def run_single_step(self, start_vol, stop_vol, flow, time_int=50, wait_time=5):
        """Aspirate or dispense a given volume at the specified flow rate
        while recording the syringe fill level according to the system clock

        Parameters
        ----------
        start_vol : float
            initial volume of syringe
        stop_vol : float
            final volume of syringe
        flow : float
        time_int : float
            approximate time in ms between data points
        wait_time : float
            time in seconds to collect steady-state data

        Returns
        -------
        """
        self.sp.set_syringe_level(start_vol)

        self.clear_data()

        # collect data for {wait_time} at start_vol
        self.collect_steady_state_data(wait_time, time_int)

        vol = start_vol - stop_vol      # because a 'positive volume' is dispensed
        # collect data while pumping
        t_data, fl_data = self.sp.pump_vol(vol, flow, poll_int=time_int)
        self.update_data(t_data, fl_data)

        # collect data for {wait_time} at stop_vol
        self.collect_steady_state_data(wait_time, time_int)

        print(self.t_data, self.fl_data)

    def run_triangle_wave(self):

        print('triangle wave')

    def run_osc_mode(self, A=10, T=10, cycles=1, phi=0, c=0):

        filename = os.path.join(
            os.path.dirname(__file__),
            r'data_files\Osc_A{}-T{}-C{}_{}.xlsx'.format(A, T, cycles, int(time()))
        )

        done = self.sp.generate_osc_flow(A, T, cycles=3, save_to=filename)

        if done:
            fit_sinusoid(filename, 2, A, T, phi, c)
            fit_sinusoid(filename, 1, A, T, phi, c)

    def plot_data(self):
        plt.plot(self.t_data, self.fl_data)
        plt.scatter(self.t_data, self.fl_data)
        plt.show()

    def finish(self):

        self.sp.disconnect()


if __name__ == "__main__":
    '''from experiment_methods import Experimenter'''
    exp = Experimenter()
    exp.initialise_pump(1)
    exp.run_single_step(1, 2, 10, time_int=20, wait_time=5)
    exp.plot_data()