import os
from time import time

# from PyQt5 import QtWidgets

from equip import CetoniSP, RFCounter
from data_handling import fit_sinusoid

# app = QtWidgets.QApplication([])
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
        self.rfcounter = None

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

    def run_single_step(self, start_vol, stop_vol, flow, trig_interval):

        time_diff_min = (stop_vol - start_vol)/flow
        time_diff_sec = time_diff_min*60

        n_meas = time_diff_sec*trig_interval
        self.rfcounter.configure_rfcounter(n_meas, trig_interval)

        self.sp.set_syringe_level(start_vol)

        vol = start_vol - stop_vol      # because a 'positive volume' is dispensed
        # thread 1:
        self.sp.pump_volume(vol, flow)
        # thread 2:
        rf_data = self.rfcounter.read_n_raw_readings(n_meas)



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

    def finish(self):

        self.sp.disconnect()