
import pyvisa

rm = pyvisa.ResourceManager()

# ===========================================================================
# Frequency counter configure and get data
# note that this set-up gives calibrated data for input data in microseconds (not kHz as implied)
# previous script had open and close for each reading - check it works without having to do this


class RFCounter(object):

    def __init__(self, record=None):
        # self.record = record
        # self.rfcounter = self.record.connect()

        self.rfcounter = None
        self.connect_to_rfcounter()

    def connect_to_rfcounter(self):
        self.rfcounter = rm.open_resource("USB0::0x0957::0x1707::MY57510158::0::INSTR")  # changed to USB address
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

        """
        gate_time = trig_interval - 0.002 # allows for dead time
        self.rfcounter.timeout = 5000

        self.rfcounter.write("*CLS")  # clear status.
        # can include *SRE 0; *ESE 0 to enable status registers but not necessary with triggered read

        self.rfcounter.write("CONF:FREQ 30E3, 0.1, (@1)")   # configure expected frequency and resolution, @ channel 1

        # self.rfcounter.write(":STATus:PRESet; :FORMat:DATA ASCii; :CONFigure:SCALar:VOLTage:PERiod")
        self.rfcounter.write("SENS:FREQ:GATE:SOUR TIME")    # sets the gate source to TIME so mmt starts after trigger
        self.rfcounter.write("SENS:FREQ:GATE:TIME {}".format(gate_time))  # set gate time in seconds allowing for dead time
        print(self.rfcounter.query("SENS:FREQ:GATE:TIME?"))
        #myinst.write(":FUNC 'FREQ 1'")

        self.rfcounter.write("SAMP:COUN 1")                     # setting sample count to 1 per trigger
        self.rfcounter.write("TRIG:COUN {}".format(n_meas))     # set the number of triggers (here readings)
        self.rfcounter.write("TRIG:SOUR EXT")                   # using an external trigger
        self.rfcounter.write("TRIG:SLOP NEG")                   # triggers on negative slope
        self.rfcounter.write("TRIG:DEL MIN")                    # sets trigger delay to minimum (here 0)

        return True

    def read_n_raw_readings(self, n_meas):

        self.rfcounter.write("INPUT:LEVEL:AUTO ONCE")   # only need to get frequency level once
        self.rfcounter.write("INIT")                    # starts waiting for a trigger
        data = []

        ### need to set trigger to begin generating pulses (if not already) ###

        for i in range(n_meas):
            a = self.rfcounter.query("DATA:REM? 1,WAIT")  # a is a string
            # read one data value taken from memory to buffer; remove value from memory after reading
            data.append(float(a.strip("\n")))

        return data

    def convert_cal_rf(self, raw_val):

        period_cal1 = -993
        period_cal2 = 0.0321
        height = 1000*(float(raw_val)*period_cal1+period_cal2)  # calibration RF --> height
        # for 'freq' reading in microseconds (not kHz as implied)

        return raw_val, height

    def close(self):
        self.rfcounter.write("*RST")
        self.rfcounter.close()


class Triggerer(object):

    def __init__(self):

        self.trig = rm.open_resource("")

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

    def start_trigger(self):
        self.trig.write("OUTP1 ON")

    def stop_trigger(self):
        self.trig.write("OUTP1 OFF")

    def close_comms(self):
        self.trig.close()


if __name__ == '__main__':

    n_meas = 20                # number of measurements to collect
    trig_interval = 0.02        # trigger pulse repetition interval in seconds, using external trigger

    rfc = RFCounter()
    print(rfc.configure_rfcounter(n_meas, trig_interval))
    data = rfc.read_n_raw_readings(n_meas)
    print(data)
    rfc.close()