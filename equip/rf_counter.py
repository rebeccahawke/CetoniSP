# ===========================================================================
# Frequency counter configure and get data
# note that this set-up gives calibrated data for input data in microseconds (not kHz as implied)
# previous script had open and close for each reading - check it works without having to do this
# ===========================================================================
from time import time_ns
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import pyqtgraph as pg
from msl.qt import QtWidgets, application
import pyqtgraph.exporters


class RFCounter(object):

    def __init__(self, record):
        self.record = record
        self.rfcounter = self.record.connect()
        self.check_connection()
        self.triggerer = None

    def check_connection(self):
        # self.rfcounter = rm.open_resource("USB0::0x0957::0x1707::MY57510158::0::INSTR")  # changed to USB address
        self.rfcounter.write("*RST")

        identity = self.rfcounter.query("*IDN?").strip("\n")
        print("Measurement using {}".format(identity))

    def configure_rfcounter(self, n_meas, trig_interval):
        """Configure rf_counter to respond to a trigger by collecting n_meas measurements,
        at time intervals of trig_interval seconds, using external trigger

        Parameters
        ----------
        n_meas : int
            number of measurements to collect
        trig_interval : float
            trigger pulse repetition interval in seconds, using external trigger

        Returns
        -------
        bool on completion
        """
        gate_time = trig_interval - 0.002 # allows for dead time
        self.rfcounter.timeout = 5000

        self.rfcounter.write("*CLS")  # clear status.
        # can include *SRE 0; *ESE 0 to enable status registers but not necessary with triggered read

        self.rfcounter.write("CONF:FREQ 25E3, 0.1, (@1)")   # configure expected frequency and resolution, @ channel 1

        # self.rfcounter.write(":STATus:PRESet; :FORMat:DATA ASCii; :CONFigure:SCALar:VOLTage:PERiod")
        self.rfcounter.write("SENS:FREQ:GATE:SOUR TIME")    # sets the gate source to TIME so mmt starts after trigger
        self.rfcounter.write("SENS:FREQ:GATE:TIME {}".format(gate_time))  # set gate time in seconds allowing for dead time
        # print(self.rfcounter.query("SENS:FREQ:GATE:TIME?"))
        #myinst.write(":FUNC 'FREQ 1'")

        self.rfcounter.write("SAMP:COUN 1")                     # setting sample count to 1 per trigger
        self.rfcounter.write("TRIG:COUN {}".format(n_meas))     # set the number of triggers (here readings)
        self.rfcounter.write("TRIG:SOUR EXT")                   # using an external trigger
        self.rfcounter.write("TRIG:SLOP NEG")                   # triggers on negative slope
        self.rfcounter.write("TRIG:DEL MIN")                    # sets trigger delay to minimum (here 0)

        return True

    def set_triggerer(self, triggerer):
        self.triggerer = triggerer

    def read_n_raw_readings(self, n_meas=250, trig_interval=0.02):
        """

        Parameters
        ----------
        n_meas : int
            number of measurements to collect

        Returns
        -------
        tuple of t0_s, data.
        t0_s is the initial time in number of seconds passed since epoch;
        data is a list of n_meas raw values from the RF counter, in Hz
        """
        # set up for fast graphing
        app = application()
        mw = QtWidgets.QMainWindow()
        mw.setWindowTitle("Capacitor raw data")
        cw = QtWidgets.QWidget()
        mw.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout()
        cw.setLayout(layout)
        pw1 = pg.PlotWidget(name='Capacitor raw data')
        curve = pw1.plot()
        layout.addWidget(pw1)
        mw.show()

        self.rfcounter.write("INPUT:LEVEL:AUTO ONCE")   # only need to get frequency level once
        self.rfcounter.write("INIT")                    # starts waiting for a trigger
        data = np.empty(n_meas)

        if self.triggerer is not None:
            self.triggerer.start_trigger()

        t0_s = time_ns()/1e9
        rdgs_per_s = 1/trig_interval

        for i in range(n_meas):
            a = self.rfcounter.query("DATA:REM? 1,WAIT")  # a is a string
            # read one data value taken from memory to buffer; remove value from memory after reading
            data[i] = float(a.strip("\n"))

            if i % rdgs_per_s == 0:  # update plot every second
                curve.setData(data[:i])  # show only the collected data
                app.processEvents()

        if self.triggerer is not None:
            self.triggerer.stop_trigger()

        t1_s = time_ns() / 1e9
        print("Elapsed time: {}".format(t1_s - t0_s))

        return t0_s, data

    def convert_cal_rf(self, data):
        # Calibration at 20/1/2020: y = 0.001135274x - 25.139001220
        # where y = height and x is RFC value in Hz (typically 25-35 kHz)

        # Calibration from 6/11/2020 using dial gauge and comparing several runs:
        pars = [-7.67043811e-12,  5.66480566e-07, -1.27538994e-02,  8.91495880e+01]
        # quartic coeffs = [2.30559894e-15, - 2.39545023e-10, 9.29559827e-06, - 1.58540497e-01,   1.00054383e+03]
        # typical error is around 20 um

        heights = []

        for raw_val in data:
            x = float(raw_val)
            height = pars[0]*x**3 + pars[1]*x**2 + pars[2]*x + pars[3]
            heights.append(height)

        return heights

    def close(self):
        self.rfcounter.write("*RST")
        self.rfcounter.close()


class Triggerer(object):

    def __init__(self, record):
        self.record = record
        self.trig = record.connect()

    def configure_triggering(self, trig_pulse_period):
        trig_pulse_width = 1.5E-6   # frequency counter accepts TTL trigger pulses of width > 1 us

        trig_pulse_Vpp = 2.4
        trig_pulse_V_offset = 1.2   # need to shift to positive voltage for TTL pulse

        self.trig.write("*RST")
        self.trig.write("SOURCE1:FUNC PULS")
        self.trig.write("SOURCE1:FUNC:PULS:TRAN:BOTH 1E-8")  #
        self.trig.write("SOURCE1:FUNC:PULS:WIDTH {}".format(trig_pulse_width))
        self.trig.write("SOURCE1:FUNC:PULS:HOLD WIDTH")     # keep pulse width constant when changing freq/period
        self.trig.write("SOURCE1:FUNC:PULS:PERIOD {}".format(trig_pulse_period))
        self.trig.write("SOURCE1:VOLT {}".format(trig_pulse_Vpp))
        self.trig.write("SOURCE1:VOLT:OFFSET {}".format(trig_pulse_V_offset))

        return True

    def start_trigger(self):
        self.trig.write("OUTP1 ON")

    def stop_trigger(self):
        self.trig.write("OUTP1 OFF")

    def close_comms(self):
        self.trig.close()


if __name__ == '__main__':
    from msl.equipment import Config

    config = r'C:\Users\r.hawke\PycharmProjects\CetoniSP\config.xml'
    cfg = Config(config)  # loads cfg file
    db = cfg.database()  # loads database
    equipment = db.equipment  # loads subset of database with equipment being used

    save = True
    trig_interval = 0.01        # trigger pulse repetition interval in seconds, using external trigger
    meas_time = 45            # duration of measurement in seconds
    n_meas = int(meas_time/trig_interval)      # number of measurements to collect
    print("Number of measurements: {}".format(n_meas))

    rfc = RFCounter(equipment['rfc_lh'])
    rfc.configure_rfcounter(n_meas, trig_interval)

    trig = Triggerer(equipment['wfg'])
    trig.configure_triggering(trig_interval)

    rfc.set_triggerer(trig)

    # take readings
    t0_s, raw_data = rfc.read_n_raw_readings(n_meas, trig_interval)

    rfc.close()
    trig.close_comms()

    height_data = rfc.convert_cal_rf(raw_data)

    # process and plot data
    times = [x * trig_interval for x in range(0, n_meas)]
    savepath = ""
    if save:
        savepath = r'../data_files/RF-data_{}.csv'.format(t0_s)
        with open(savepath, mode='w') as fp:
            fp.write("Timestamp,Frequency (Hz),Height (mm)\n")
            for a, b, c in zip(times, raw_data, height_data):
                fp.write("{},{},{}\n".format(datetime.fromtimestamp(t0_s + a), b, c))
            fp.close()
        print("Data saved to {}".format(savepath))

    # plt.plot(times, height_data)
    plt.scatter(times, height_data)
    plt.title("RFC position")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (mm)")
    plt.tight_layout()
    if save:
        plt.savefig(savepath.strip("csv")+"png")
    plt.show()
