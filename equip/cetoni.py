import sys
import os
import numpy as np
from time import sleep, perf_counter, time_ns
from openpyxl import Workbook

qmixsdk_dir = r"C:\Users\Public\QmixSDK_MSVC17-64bit_Setup_v20200902"
# updated QmixSDK using MSVC compiler for dlls; note that these libraries are in PyQt5 version 12
sys.path.append(qmixsdk_dir + "/lib/python")
os.environ['PATH'] += os.pathsep + qmixsdk_dir
sys.path.append(qmixsdk_dir)

os.environ['LOG4CPLUS_LOGLOG_QUIETMODE'] = 'false'

from qmixsdk import qmixbus
from qmixsdk import qmixpump
from qmixsdk import qmixvalve
# from qmixsdk.qmixbus import UnitPrefix, TimeUnit


timeout = 60*60*12


class CetoniSP(object):
    def __init__(self, deviceconfig):
        # Open bus with deviceconfig
        # N.B. create the deviceconfig file using the QmixElements software
        self.bus = qmixbus.Bus()
        self.bus.open(deviceconfig, 0)

        self.vol_unit = None
        self.flow_unit = None

        self.syringe = None
        self.calibrated = False

    def look_for_devices(self):
        print("Looking up devices...")
        pumpcount = qmixpump.Pump.get_no_of_pumps()
        print("Number of pumps: {}".format(pumpcount))
        for i in range(pumpcount):
            pump2 = qmixpump.Pump()
            pump2.lookup_by_device_index(i)
            print("Name of pump {} is {}".format(i, pump2.get_device_name()))

    def device_name_lookup(self, name):
        # Find specific pump
        self.pump = qmixpump.Pump()
        self.pump.lookup_by_name(name)

    def bus_start(self):
        # Start bus communication...
        self.bus.start()

    def enable_pump(self):
        # Enable pump drive
        if self.pump.is_in_fault_state():
            self.pump.clear_fault()
        assert(not self.pump.is_in_fault_state())
        if not self.pump.is_enabled():
            self.pump.enable(True)
        assert(self.pump.is_enabled())

    def connect_to_pump(self, name):
        self.device_name_lookup(name)
        self.bus_start()
        self.enable_pump()

    @staticmethod
    def wait_calibration_finished(pump, timeout_seconds):
        """
        The function waits until the given pump has finished calibration or
        until the timeout occurs.
        """
        timer = qmixbus.PollingTimer(timeout_seconds * 1000)
        result = False
        while (not result) and not timer.is_expired():
            sleep(0.1)
            result = pump.is_calibration_finished()
        return result

    @staticmethod
    def wait_dosage_finished(pump, timeout_seconds):
        """
        The function waits until the last dosage command has finished or
        until the timeout occurs.
        """
        timer = qmixbus.PollingTimer(timeout_seconds * 1000)
        message_timer = qmixbus.PollingTimer(500)
        result = True
        while result and not timer.is_expired():
            sleep(0.1)
            if message_timer.is_expired():
                fl = pump.get_fill_level()
                print("Fill level: {}".format(fl))
                message_timer.restart()
            result = pump.is_pumping()
        return not result

    @staticmethod
    def collect_position_data(pump, timeout_seconds, time_int):
        """Collect syringe fill level information during dosage,
        following the protocol above.

        Parameters
        ----------
        pump : pump
            reference to syringe pump, here self.pump
        timeout_seconds : int
            when to give up waiting for the pump to stop pumping
        time_int : float
            time interval in ms (?) between readings of fill level

        Returns
        -------
        tuple of t_data, fl_data; time in seconds and fill level in unit set for pump
        """
        print("Collecting position data every {} ms".format(time_int))
        timer = qmixbus.PollingTimer(timeout_seconds * 1000)
        message_timer = qmixbus.PollingTimer(time_int)
        result = True
        t_data = []
        fl_data = []
        while result and not timer.is_expired():
            if message_timer.is_expired():
                t = time_ns() / 1e9
                fl = pump.get_fill_level()
                t_data.append(t)
                fl_data.append(fl)
                # print("Timer (s): {}; Fill level: {}".format(t, fl))
                message_timer.restart()
            result = pump.is_pumping()
        return t_data, fl_data

    def configure_syringe(self, inner_dia, stroke=60.0):
        """Syringe configuration

        Parameters
        ----------
        inner_dia : float
            inner diameter in mm (e.g. 23.0329 mm for 25 mL syringe)
        stroke : float
            length of scale in mm (typically 60.0)

        Returns
        -------
        tuple of inner diameter and piston stroke setting
        """
        self.pump.set_syringe_param(inner_dia, stroke)
        self.syringe = self.pump.get_syringe_param()
        assert(inner_dia == self.syringe.inner_diameter_mm)
        assert(stroke == self.syringe.max_piston_stroke_mm)

        return self.syringe.inner_diameter_mm, self.syringe.max_piston_stroke_mm

    def set_units(self, vol_unit, flow_unit):
        """Set units for volume and flow

        Parameters
        ----------
        vol_unit : str
            allowed 'mL', 'L'
        flow_unit : str
            allowed 'mL/s', 'mL/min'

        Returns
        -------
        tuple of volume and flow units as set on the device
        """
        vol_units = {
            'mL': (qmixpump.UnitPrefix.milli, qmixpump.VolumeUnit.litres),
            'L': (qmixpump.UnitPrefix.unit, qmixpump.VolumeUnit.litres),
        }
        self.pump.set_volume_unit(vol_units.get(vol_unit)[0], vol_units.get(vol_unit)[1])
        print("Max. volume: {} {}".format(round(self.pump.get_volume_max(), 3), vol_unit))

        flow_units = {
            'mL/s': (qmixpump.UnitPrefix.milli, qmixpump.VolumeUnit.litres, qmixpump.TimeUnit.per_second),
            'mL/min': (qmixpump.UnitPrefix.milli, qmixpump.VolumeUnit.litres, qmixpump.TimeUnit.per_minute),
        }
        self.pump.set_flow_unit(flow_units.get(flow_unit)[0], flow_units.get(flow_unit)[1], flow_units.get(flow_unit)[2])
        print("Max. flow: {} {}".format(round(self.pump.get_flow_rate_max(), 3), flow_unit))

        self.vol_unit = vol_unit
        self.flow_unit = flow_unit

        s_v = self.pump.get_volume_unit()
        for i, u in enumerate(vol_units.get(vol_unit)):
            assert u == s_v[i]
        s_f = self.pump.get_flow_unit()
        for i, u in enumerate(flow_units.get(flow_unit)):
            assert u == s_f[i]

        return self.pump.get_volume_unit(), self.pump.get_flow_unit()

    def calibrate_pump(self):
        """Run the self-calibration routine to set internal limits"""

        print("Calibrating pump...")
        # check no syringe in pump
        if not self.check_no_syringe_in_pump('self-calibration'):
            return

        self.pump.calibrate()
        sleep(0.2)
        self.calibrated = self.wait_calibration_finished(self.pump, timeout)
        print("Pump calibrated: {}".format(self.calibrated))
        assert self.calibrated

    @property
    def is_configured(self):
        if not self.vol_unit:
            print('set volume unit')
            return False
        if not self.flow_unit:
            print('set flow unit')
            return False
        if not self.syringe:
            print('configure syringe')
            return False
        if not self.calibrated:
            print('Calibration required')
            return False
        return True

    @property
    def syringe_params(self):
        if self.syringe:
            return self.syringe.inner_diameter_mm, self.syringe.max_piston_stroke_mm
        else:
            return None, None

    @property
    def syringe_fill_level(self):
        """

        Returns
        -------
        float of fill level, or None if pump not configured
        """
        if self.is_configured:
            return self.pump.get_fill_level()
        else:
            return None

    @property
    def flow_rate_is(self):
        """Poll syringe pump for current flow rate

        Returns
        -------
        float of current flow rate, or None if pump not configured
        """
        if self.is_configured:
            return self.pump.get_flow_is()
        else:
            return None

    def test_aspirate(self):
        if not self.is_configured:
            return False

        print("Testing aspiration...")
        # check no syringe in pump
        if not self.check_no_syringe_in_pump('test'):
            return

        max_volume = self.pump.get_volume_max() / 2
        if max_volume > self.pump.get_fill_level():
            max_volume -= self.pump.get_fill_level()
        max_flow = self.pump.get_flow_rate_max()
        self.pump.aspirate(max_volume, max_flow)
        finished = self.wait_dosage_finished(self.pump, timeout)
        assert finished

    def aspirate_vol_flow(self, vol, flow):
        """Aspirate (withdraw) specified volume at specified flow rate

        Parameters
        ----------
        vol : float >= 0
            volume in units previously set using set_units
        flow : float
            flow in units previously set using set_units

        Returns
        -------
        bool for completion
        """
        if not self.is_configured:
            return False
        allowed_vol = self.pump.get_volume_max() - self.pump.get_fill_level()
        if vol > allowed_vol:
            print('Requested volume too large; only {} {} remaining'.format(round(allowed_vol, 3), self.vol_unit))
            return False
        elif flow > self.pump.get_flow_rate_max():
            print(
                'Requested flow rate too large; must be less than {} {}'.format(
                    round(self.pump.get_flow_rate_max(), 3), self.flow_unit)
            )
            return False
        else:
            print('Aspirating {} {} at {} {}'.format(vol, self.vol_unit, flow, self.flow_unit))
            self.pump.aspirate(vol, flow)
            finished = self.wait_dosage_finished(self.pump, timeout)
            return finished

    def test_dispense(self):
        if not self.is_configured:
            return False

        print("Testing dispensing...")
        # check no syringe in pump
        if not self.check_no_syringe_in_pump('test'):
            return

        max_volume = self.pump.get_volume_max() / 10
        if max_volume > self.pump.get_fill_level():
            max_volume -= self.pump.get_fill_level()
        max_flow = self.pump.get_flow_rate_max() / 2
        self.pump.dispense(max_volume, max_flow)
        finished = self.wait_dosage_finished(self.pump, timeout)
        assert finished

    def dispense_vol_flow(self, vol, flow):
        """Dispense (infuse) specified volume at specified flow rate

        Parameters
        ----------
        vol : float >= 0
            volume in units previously set using set_units
        flow : float
            flow in units previously set using set_units

        Returns
        -------
        bool for completion
        """
        if not self.is_configured:
            return False
        allowed_vol = self.pump.get_fill_level()
        if vol > allowed_vol:
            print('Requested volume too large; only {} {} remaining'.format(round(allowed_vol, 3), self.vol_unit))
            return False
        elif flow > self.pump.get_flow_rate_max():
            print(
                'Requested flow rate too large; must be less than {} {}'.format(
                    round(self.pump.get_flow_rate_max(), 3), self.flow_unit)
            )
            return False
        else:
            print('Dispensing {} {} at {} {}'.format(vol, self.vol_unit, flow, self.flow_unit))
            self.pump.dispense(vol, flow)
            finished = self.wait_dosage_finished(self.pump, timeout)
            return finished

    def test_pump_volume(self):
        # tests pump for aspirate and then dispense, at same flow rate
        if not self.is_configured:
            return False

        print("Testing pumping volume...")
        # check no syringe in pump
        if not self.check_no_syringe_in_pump('test'):
            return

        pumped_volume = self.pump.get_volume_max() / 10
        max_flow = self.pump.get_flow_rate_max() / 3

        allowed_vol = self.pump.get_volume_max() - self.pump.get_fill_level()
        if pumped_volume > allowed_vol:
            pumped_volume = allowed_vol
        self.pump.pump_volume(-pumped_volume, max_flow)  # negative volume aspirates
        finished = self.wait_dosage_finished(self.pump, timeout)
        assert finished

        self.pump.pump_volume(pumped_volume, max_flow)  # positive volume dispenses
        finished = self.wait_dosage_finished(self.pump, timeout)
        assert finished

    def pump_vol(self, volume, flow, poll_int=50):
        """Pump a specified volume at a given flow rate.
        A negative volume will be aspirated/withdrawn; a positive volume will be dispensed.

        Parameters
        ----------
        volume : float
        flow : float
        poll_int : int
            time interval in ms for data collection

        Returns
        -------
        bool for completion
        """
        if not self.is_configured:
            return False
        if not 0 <= flow <= self.pump.get_flow_rate_max():
            print('Flow of {} requested is outside bounds {}'.format(flow, self.pump.get_flow_rate_max()))
        if not 0 <= self.syringe_fill_level - volume <= self.pump.get_volume_max():
            print('Forbidden volume')
            print('Asked for {}, but would reach {}'.format(volume, self.syringe_fill_level - volume))
            return False

        print("Pumping {} {} at {} {}".format(volume, self.vol_unit, flow, self.flow_unit))
        self.pump.pump_volume(volume, flow)

        return self.collect_position_data(self.pump, timeout, poll_int)

    def test_generate_flow(self):
        if not self.is_configured:
            return False

        print("Testing generating flow...")
        # check no syringe in pump
        if not self.check_no_syringe_in_pump('test'):
            return

        max_flow = self.pump.get_flow_rate_max()/3
        self.pump.generate_flow(max_flow)
        sleep(1)
        flow_is = self.pump.get_flow_is()
        assert(round(max_flow, 1) == round(flow_is, 1))
        finished = self.wait_dosage_finished(self.pump, timeout)
        assert finished

    def generate_flow(self, flow):
        if not self.is_configured:
            return False
        if not -self.pump.get_flow_rate_max() <= flow <= self.pump.get_flow_rate_max():
            print('Flow of {} requested is outside bounds {}'.format(flow, self.pump.get_flow_rate_max()))

        self.pump.generate_flow(flow)
        print("Generating flow of {} {}".format(flow, self.flow_unit))
        sleep(0.2)
        flow_is = self.pump.get_flow_is()
        assert(round(flow, 2) == round(flow_is, 2))

        return flow_is

    def generate_osc_flow(self, A, T, cycles, save_to=None):
        """Generate sinusoidally oscillating flow with amplitude A in flow units where time is in minutes,
        and time period T in seconds, lasting for a duration in seconds

        Parameters
        ----------
        A : float
            amplitude of sinusoid in whatever flow units have been set for the pump
        T : float
            period of sinusoid in seconds
        cycles : float
            number of full oscillations

        Returns
        -------
        Bool on completion
        """
        if not self.is_configured:
            return False
        if not 0 <= A <= self.pump.get_flow_rate_max():
            print('Flow of {} requested is outside bounds {}'.format(A, self.pump.get_flow_rate_max()))

        volume_change = A * T * 1/60 * 1/np.pi
        print("Volume change in volume unit if time is in minutes", volume_change)
        if volume_change > self.syringe_fill_level:
            print('Requested volume change too large; only {} {} remaining'.format(round(self.syringe_fill_level, 3), self.vol_unit))
            return False

        wb = Workbook()
        ws = wb.active

        print("Beginning oscillating flow with max flow rate of {} {} "
              "and period of {} s, for {} cycles".format(A, self.flow_unit, T, cycles))
        header = [
            "Time (s)",
            "Syringe fill level ({})".format(self.vol_unit),
            "Flow ({})".format(self.flow_unit)
            ]
        print(header)
        if save_to is not None:
            ws.append(header)

        t0 = perf_counter()
        while perf_counter() - t0 < cycles * T:
            t = perf_counter() - t0
            flow = A * np.sin(2 * np.pi * t / T)
            self.pump.generate_flow(flow)
            sleep(0.15)
            row = [perf_counter() - t0, self.syringe_fill_level, self.flow_rate_is]
            print(row)
            if save_to is not None:
                ws.append(row)

        self.generate_flow(0)

        if save_to is not None:
            wb.save(save_to)

        return True

    def test_set_syringe_level(self):
        if not self.is_configured:
            return False

        print("Testing set syringe fill level...")
        # check no syringe in pump
        if not self.check_no_syringe_in_pump('test'):
            return

        max_flow = self.pump.get_flow_rate_max() / 2
        max_volume = self.pump.get_volume_max() / 2
        self.pump.set_fill_level(max_volume, max_flow)
        finished = self.wait_dosage_finished(self.pump, timeout)
        assert finished

        fill_level_is = self.pump.get_fill_level()
        print(max_volume, fill_level_is)

        self.pump.set_fill_level(0, max_flow)
        finished = self.wait_dosage_finished(self.pump, timeout)
        assert finished

        fill_level_is = self.pump.get_fill_level()
        print(0, fill_level_is)

    def set_syringe_level(self, vol, flow=None):
        if not self.is_configured:
            return False
        if not flow:
            flow = self.pump.get_flow_rate_max() / 3
        if 0 <= vol <= self.pump.get_volume_max():
            print('Setting syringe level to {} {}'.format(vol, self.vol_unit))
            self.pump.set_fill_level(vol, flow)
            finished = self.wait_dosage_finished(self.pump, timeout)
            assert finished

    def test_valve(self):
        print("Testing valve...")
        if not self.pump.has_valve():
            print("No valve installed")
            return

        valve = self.pump.get_valve()
        valve_pos_count = valve.number_of_valve_positions()
        print("Valve positions: {}".format(valve_pos_count))
        for i in range(valve_pos_count):
            valve.switch_valve_to_position(i)
            sleep(0.2)  # give valve some time to move to target
            valve_pos_is = valve.actual_valve_position()
            assert(i == valve_pos_is)

    def valve_to_pos(self, pos):
        """Move valve to specified position

        Parameters
        ----------
        pos : int
            Counting starts from 0, e.g. 0 or 1 for nemesys pump
        Returns
        -------
        bool: True if move complete, False if not
        """
        valve = self.pump.get_valve()
        valve_pos_count = valve.number_of_valve_positions()
        if 0 <= pos < valve_pos_count:
            valve.switch_valve_to_position(pos)
            sleep(0.2)
            print('Valve is in position {}'.format(valve.actual_valve_position()))
            return True
        else:
            print('Valve directed to illegal position target: {} (valve has {} positions).'.format(pos, valve_pos_count))
            return False

    def disconnect(self):
        print("Closing bus")
        self.bus.stop()
        self.bus.close()

    @staticmethod
    def check_no_syringe_in_pump(operation):
        # check no syringe in pump
        ok = input("Remove syringe from pump before commencing {}."
                   "\nPress enter to continue or press c to cancel".format(operation))
        if 'c' in ok.lower():
            return False
        return True

    @staticmethod
    def install_syringe():
        ok = input("OK to install syringe. Press enter when syringe installed or press c to cancel")
        if 'c' in ok.lower():
            return False
        return True

