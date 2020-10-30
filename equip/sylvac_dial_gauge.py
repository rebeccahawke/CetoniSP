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

        return reading


if __name__ == "__main__":
    from msl.equipment import Config

    config = r'C:\Users\r.hawke\PycharmProjects\CetoniSP\config.xml'
    cfg = Config(config)  # loads cfg file
    db = cfg.database()  # loads database
    equipment = db.equipment  # loads subset of database with equipment being used
    record = db.equipment['sylvac75']

    syl = DialGauge(record)

    for i in range(20):
        print(syl.get_reading())
