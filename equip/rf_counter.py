
import pyvisa
from time import sleep

rm = pyvisa.ResourceManager()

# ===========================================================================
# Frequency counter configure and get data
# note that this set-up gives calibrated data for input data in microseconds (not kHz as implied)
# previous script had open and close for each reading - check it works without having to do this


class RFCounter(object):

    def __init__(self):

        self.rfcounter = rm.open_resource("GPIB1::3::INSTR")

    def configure_rfcounter(self, timeint):

        self.rfcounter.timeout = 5000
        self.rfcounter.write("*RST")

        identity = self.rfcounter.query("*IDN?").strip("\n")
        print("Measurement using {}".format(identity))

        self.rfcounter.write("*CLS; *SRE 0; *ESE 0")

        self.rfcounter.write(":STATus:PRESet; :FORMat:DATA ASCii; :CONFigure:SCALar:VOLTage:PERiod")
        self.rfcounter.write("SENS:FREQ:GATE:TIME {}".format(timeint))
        #print(rfcounter.query("SENS:FREQ:GATE:TIME?"))
        #myinst.write(":FUNC 'FREQ 1'")

    def read_raw_rf(self):

        return self.rfcounter.ask(":READ?").strip('\n')

    def read_raw_rf_n(self, n):

        self.rfcounter.write("SAMP:COUN {}".format(n))  # ask for n readings // set readings/trigger

        return self.rfcounter.ask(":READ?").strip('\n').split(",")

    def read_cal_rf(self):

        period = self.read_raw_rf()

        period_cal1 = -993
        period_cal2 = 0.0321
        height = 1000*(float(period)*period_cal1+period_cal2)  # calibration RF --> height
        # for 'freq' reading in microseconds (not kHz as implied)

        return period, height

    def get_freq_n(self, n, dt):

        freq = []
        for i in range(n):
            reading = self.read_raw_rf()
            freq.append(float(reading))
            sleep(dt)

        return freq

