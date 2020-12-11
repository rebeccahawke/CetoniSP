from time import time_ns
import matplotlib.pyplot as plt

import pyvisa

rm = pyvisa.ResourceManager()


class LVDT_HP3458A(object):

    def __init__(self, record=None, reset=False):
        self.record = record
        self.lvdt = self.record.connect()

        # self.lvdt = rm.open_resource("GPIB0::23::INSTR")

        if reset:
            self.lvdt.write("*RST")

        self.lvdt.read_termination = "\r\n"
        self.lvdt.timeout = 1000000  # Need to set timeout for when waiting for start of triggers. Think this is in ms.
        self.lvdt.clear()
        identity = self.lvdt.query("ID?")
        print('LVDT measurement via ' + identity)

        self.scale_fac = 0

    def configure_lvdt(self, n):
        """Configure for normal operation"""
        self.lvdt.write("PRESET NORM")
        self.lvdt.write("DISP OFF")
        self.lvdt.write("AZERO OFF")
        self.lvdt.write("MATH OFF")
        self.lvdt.write("DCV 10")
        self.lvdt.write("NPLC {}".format(n))
        # self.lvdt.write("OFORMAT DINT")
        # self.lvdt.write("MFORMAT DINT")
        self.lvdt.write("TIMER 1")
        self.lvdt.read_termination = "\r\n"
        self.lvdt.timeout = 5000
        identity = self.lvdt.query("ID?")
        print('LVDT measurement via ' + identity)

        return True

    def config_lvdt_ext_trig(self, trig_int, range):
        """Configure reading of LVDT from HP3458A for external triggering"""
        aper_time = trig_int - 0.000030  # allow for 30 us dead time
        self.lvdt.write("END ALWAYS")  # End or Identify set to true when last byte of all readings sent
        self.lvdt.write("PRESET NORM")  # sets TARM AUTO, TRIG SYN, MEM OFF, MATH OFF
        self.lvdt.write("TARM HOLD")
        self.lvdt.write("TRIG AUTO")
        # self.lvdt.write("RQS 16")  # request for service on event(s); here message available
        self.lvdt.write("OFORMAT ASCII")
        self.lvdt.write("MFORMAT DINT")
        # print("DCV {}".format(range))
        self.lvdt.write("DCV {}".format(range))
        # print("APER {}".format(aper_time))
        self.lvdt.write("APER {}".format(aper_time))
        self.lvdt.write("TIMER {}".format(trig_int))
        self.lvdt.write("AZERO OFF")

        # read scale factor for DINT readings
        # n_bytes = 100
        self.scale_fac = self.lvdt.query("ISCALE?")
        self.lvdt.write("AZERO ON")

    def read_prep_ext_trig(self, n_meas):
        """Prepare LVDT for external triggering"""
        self.lvdt.write("DISP OFF")
        self.lvdt.write("AZERO OFF")
        self.lvdt.write("MEM FIFO")  # enable reading memory in first in first out mode
        self.lvdt.write("NRDGS {},EXT".format(n_meas))  # sample n_meas readings using an external trigger
        self.lvdt.write("TARM SGL")  # arm trigger once then return to hold when finished

    def read_lvdt_ext_trig(self):
        n = int(self.lvdt.query("MCOUNT?"))
        print("MCOUNT:", n)
        readings = [0]*n
        for i in range(n):
            a = lvdt.read_raw_lvdt()
            # print(a)
            readings[i] = a

        return readings

    def read_end_ext_trig(self):
        self.lvdt.write("AZERO ON")
        self.lvdt.write("DISP ON")
        self.lvdt.write("NRDGS 1,AUTO")
        self.lvdt.write("TARM AUTO")

    def read_raw_lvdt(self):
        return float(self.lvdt.read().strip('\n'))

    def read_cal_lvdt(self):
        # TODO: add calibration information here
        raise NotImplementedError
        #return self.read_raw_lvdt()

    def close_comms(self):
        self.read_end_ext_trig()
        self.lvdt.close()


class LVDT_HP34970A(object):

    VCAL1 = -0.0009755
    VCAL2 = -0.003229

    def __init__(self):
        self.lvdt = rm.open_resource("GPIB1::9::INSTR")

    def configure_lvdt(self, n):

        self.lvdt.write("*RST")
        self.lvdt.write("*CLS")
        self.lvdt.write("CONFigure:VOLTage:DC 10,(@113)\n")

        identity = self.lvdt.query("*IDN?")
        print('LVDT measurement via ' + identity)

        return True

    def read_raw_lvdt(self):
        return float(self.lvdt.ask("MEAS:VOLT:DC? (@112)").strip('\n'))

    def read_cal_lvdt(self):

        raw = self.read_raw_lvdt()

        height = 1000 * (float(raw) * self.VCAL1 + self.VCAL2)

        return raw, height

    def close_lvdt(self):

        self.lvdt.close()


if __name__ == "__main__":
    from msl.equipment import Config

    config = r'C:\Users\r.hawke\PycharmProjects\CetoniSP\config.xml'
    cfg = Config(config)  # loads cfg file
    db = cfg.database()  # loads database
    equipment = db.equipment  # loads subset of database with equipment being used

    lvdt = LVDT_HP3458A(equipment['lvdt'])
    # lvdt.configure_lvdt(1)
    # print(lvdt.read_raw_lvdt())

    save = True
    trig_interval = 0.01        # trigger pulse repetition interval in seconds, using external trigger
    meas_time = 45            # duration of measurement in seconds
    n_meas = int(meas_time/trig_interval)      # number of measurements to collect
    print("Number of measurements: {}".format(n_meas))

    lvdt.config_lvdt_ext_trig(trig_int=trig_interval, range=10)

    t0_s = time_ns() / 1e9
    lvdt.read_prep_ext_trig(n_meas=n_meas)
    t1_s = time_ns() / 1e9
    print("Elapsed time: {}".format(t1_s - t0_s))

    readings = lvdt.read_lvdt_ext_trig()
    # print(readings)

    lvdt.close_comms()

    # process and plot data
    times = [x * trig_interval for x in range(0, n_meas)]
    savepath = ""
    if save:
        savepath = r'../data_files/LVDT-data_{}.csv'.format(t0_s)
        with open(savepath, mode='w') as fp:
            fp.write("Timestamp,Voltage (V)\n")
            for a, b in zip(times, readings):
                fp.write("{:.3f},{}\n".format(a, b))
            fp.close()
        print("Data saved to {}".format(savepath))

    # plt.plot(times, readings)
    plt.scatter(times, readings)
    plt.title("LVDT readings")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.tight_layout()
    if save:
        plt.savefig(savepath.strip("csv")+"png")
    plt.show()
