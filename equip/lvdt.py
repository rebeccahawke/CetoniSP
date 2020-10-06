import pyvisa

rm = pyvisa.ResourceManager()


class LVDT_HP3458A(object):

    def __init__(self):
        self.lvdt = rm.open_resource("GPIB0::23::INSTR")

    def configure_lvdt(self, n):

        self.lvdt.write("RESET")
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

    def read_raw_lvdt(self):
        return -1 * float(self.lvdt.read().strip('\n'))

    def read_cal_lvdt(self):
        # TODO: add calibration information here
        raise NotImplementedError
        #return self.read_raw_lvdt()

    def close_lvdt(self):
        self.lvdt.write("RESET")
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