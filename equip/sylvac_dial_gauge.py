"""
Class for reading a Sylvac Dial Gauge
"""


class DialGauge(object):
    def __init__(self, record):
        self.record = record

    def get_reading(self):
        syl = record.connect()
        syl.serial.flush()
        reading = syl.query('PRI?')
        syl.disconnect()

        return reading.strip("~")


if __name__ == "__main__":
    from time import time_ns
    from msl.equipment import Config

    config = r'C:\Users\r.hawke\PycharmProjects\CetoniSP\config.xml'
    cfg = Config(config)  # loads cfg file
    db = cfg.database()  # loads database
    equipment = db.equipment  # loads subset of database with equipment being used
    record = db.equipment['sylvac75']

    syl = DialGauge(record)


    savepath = r'C:\Users\r.hawke\PycharmProjects\CetoniSP\data_files\RFC_cal_vacuum/DialGauge_{}_{}.csv'.format(12, time_ns()/ 1e9)
    with open(savepath, mode='w') as fp:
        fp.write("Mmmt No.,Timestamp (s),Height (mm)\n")
        for i in range(50):
            print(i)
            fp.write("{},{},{}\n".format(i, time_ns()/ 1e9, syl.get_reading()))
        fp.close()
    print("Data saved to {}".format(savepath))

