"""
A collection of assorted helper functions to collate and process data files from Igor Pro.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook, Workbook

from data_handling.plot_data import fit_linear, fit_quadratic, fit_sinusoid


def convert_cal_rf(data):
    # copied from RFCounter class because it wasn't letting me import it ?!
    # Calibration at 20/1/2020: y = 0.001135274x - 25.139001220
    # where y = height and x is RFC value in Hz (typically 25-35 kHz)
    # this calibration was done using a two-point calibration; may wish to redo with more points
    heights = []

    for raw_val in data:
        a = 0.001135274
        b = -25.139001220

        height = a*float(raw_val)+b
        heights.append(height)

    return heights


def convert_cal_lvdt(data):
    # where y = height and x is LVDT value in V (typically 1 - 6 V)
    # using quadratic fit from RFCounter calibration values as at 27/10/2020
    # see ...MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\SyringePumpTests\20201023_RFC-LVDT_IgorData for raw data
    heights = []

    for raw_val in data:
        a = -0.03542046
        b = 1.07974847
        c = -0.39624465

        height = a*float(raw_val)**2 + b*float(raw_val) + c
        heights.append(height)

    return heights


def get_all_xlsx_fnames(mydir):
    files = []
    igor_data = []

    for file in os.listdir(mydir):
        if file.endswith(".xlsx"):
            files.append(file)
        if "dP_volt" in file:
            igor_data.append(file)

    return files, igor_data


def get_all_LVDTxlsx_fnames(mydir):
    files = []

    for file in os.listdir(mydir):
        if file.endswith("_LVDT.xlsx"):
            files.append(file)

    return files


def read_in_data(folder, fname):
    f1 = os.path.join(folder, fname)

    xls = pd.ExcelFile(f1)
    df1 = pd.read_excel(xls, 'SP data')
    df2 = pd.read_excel(xls, 'RFC data')

    return df1, df2


def collate_igordata(folder):
    xlsxfiles, igorfiles = get_all_xlsx_fnames(folder)
    print("{} files to process".format(len(xlsxfiles)))

    for x, i in zip(xlsxfiles, igorfiles):
        print(x, i)
        i_data = pd.read_csv(os.path.join(folder, i), header=None)

        wb = load_workbook(os.path.join(folder, x))
        ws = wb["RFC data"]
        ws["E1"] = "LVDT (V)"
        for r in range(6000):
            d = float(i_data.loc[r])
            ws["E"+str(r+3)] = d

        newname = x.strip(".xlsx") + "_LVDT.xlsx"
        wb.save(os.path.join(folder, newname))


def get_rough_maxmin(sp_df, rfc_df):
    spA = 0.5*(max(sp_df['SP Position (mL)'][800:1500]) - min(sp_df['SP Position (mL)'][800:1500]))
    rfcA = 0.5*(max(rfc_df['Height (mm)'][2000:4000]) - min(rfc_df['Height (mm)'][2000:4000]))
    lvdtA = 0.5*(max(rfc_df['LVDT (V)'][2000:4000]) - min(rfc_df['LVDT (V)'][2000:4000]))

    return spA, rfcA, lvdtA


def plot_roughmaxmin(flows, RFCamplitudes, LVDTamplitudes):
    plt.scatter(flows, RFCamplitudes, label="RFC")
    plt.scatter(flows, LVDTamplitudes, label="LVDT")

    plt.ylabel('Rough amplitude of motion (mm)')
    plt.xlabel('Flow at syringe pump (mL/min)')

    plt.legend()

    plt.tight_layout()
    plt.show()


def do_rough_maxmin(folder):
    fnames = get_all_LVDTxlsx_fnames(folder)

    flows = []
    RFCamplitudes, LVDTamplitudes = [], []

    wb = Workbook()
    ws = wb.active
    ws.append(["Flow (mL/min)", "RFC amplitude (mm)", "LVDT amplitude (V)"])

    for fname in fnames:
        code = fname.split("_")
        flows.append(float(code[2]))
        sp1, rfc1 = read_in_data(folder, fname)

        sp, rfcA, lvdtA = get_rough_maxmin(sp1, rfc1)

        RFCamplitudes.append(rfcA)
        LVDTamplitudes.append(lvdtA)

        ws.append([float(code[2]), rfcA, lvdtA])

    savepath = os.path.join(folder, "RoughMaxMin.xlsx")

    wb.save(savepath)
    print(savepath)

    plot_roughmaxmin(flows, RFCamplitudes, LVDTamplitudes)


def plot_timeseries_data(sp_df, rfc_df):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    # fig.suptitle('Piston height over time')
    ax1.set_ylabel('Height (mm)')
    # ax1.set_title('From Capacitor')
    ax2.set_ylabel('Syringe fill level (mm)')
    ax2.set_xlabel('Timestamp')
    ax1.plot(pd.to_datetime(rfc_df['Timestamp']), rfc_df['Height (mm)'])
    ax2.plot(pd.to_datetime(sp_df['Timestamp']), sp_df['SP Position (mL)'])
    fig.autofmt_xdate()

    # plt.plot(pd.to_datetime(sp_data['Timestamp']), sp_data[' SP Position (mL)'])
    # plt.plot(pd.to_datetime(rfc_data['Timestamp']), rfc_data[" Frequency (Hz)"]/7500)
    plt.show()


def combine_lvdt_rfc_files(mydir):
    # get both lvdt and rfc data files from Igor Pro
    files = []
    igor_data = []

    for file in os.listdir(mydir):
        if file.startswith("LH_hgt") and file.endswith(".csv"):
            files.append(file)
        if "dP_volt" in file:
            igor_data.append(file)

    return files, igor_data


def combine_lvdt_rfc_data(folder):
    # e.g. use with
    # folder = r'I:\MSL\Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\SyringePumpTests\20201023_RFC-LVDT_IgorData'
    rfcf, lvdtf = combine_lvdt_rfc_files(folder)

    all_rfc_data = pd.DataFrame()
    all_lvdt_data = pd.DataFrame()

    for rf, lf in zip(rfcf[4:], lvdtf[4:]):
        rfc_data = pd.read_csv(os.path.join(folder, rf), header=None)
        rfc_data.columns = ["RF"]
        # print(rfc_data)
        all_rfc_data = all_rfc_data.append(rfc_data)

        lvdt_data = pd.read_csv(os.path.join(folder, lf), header=None)
        lvdt_data.columns = ["LVDT"]
        # print(lvdt_data)
        all_lvdt_data = all_lvdt_data.append(lvdt_data)

        # plt.scatter(rfc_data, lvdt_data)
        # plt.savefig(os.path.join(folder, rf.strip("csv")+"png"))

        # print(rf, lf, float(rfc_data.loc[50])/float(lvdt_data.loc[50]))

    plt.scatter(all_rfc_data, all_lvdt_data)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("LVDT Voltage (V)")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "CollatedData"))
    plt.show()

    all_data = pd.concat([all_rfc_data, all_lvdt_data], axis=1)
    all_data.to_excel(os.path.join(folder, "CollatedData.xlsx"))


def get_LVDTcal_from_RFCcal(datafile):
    # used with
    # folder = r'I:\MSL\Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\SyringePumpTests\20201023_RFC-LVDT_IgorData'
    # fname = "CollatedData.xlsx"

    df = pd.read_excel(datafile)
    # headers = list(df.columns.values.tolist())

    cal_height_rf = convert_cal_rf(df["RF"])  # headers[1]; get RFC data as Hz and convert it to position in mm

    plt.scatter(df["LVDT"], cal_height_rf)  # plot LVDT (headers[2]) against RFC data as position in mm
    plt.show()
    fit_quadratic(df["LVDT"], cal_height_rf, 0.1, 1, 0.1)

    # returns calibration [a, b, c] parameters: [-0.03542046  1.07974847 -0.39624465]
    # linear fit returns [ 0.82622535 -0.01726006]; Standard deviations: [0.00022915 0.00075441] but not a great fit.


def plot_rfc_lvdt():
    folder = r'I:\MSL\Shared\MSL Kibble Balance\_p_PressureManifoldConsiderations\Flow and Pressure control\SyringePumpTests\20201023_RFC-LVDT_IgorData'

    rf = "LH_hgt_scan0009.csv"
    lf = "dP_volt_scan0009.csv"

    rfc_data = pd.read_csv(os.path.join(folder, rf), header=None)
    rfc_data.columns = ["RF"]
    print(rfc_data)

    lvdt_data = pd.read_csv(os.path.join(folder, lf), header=None)
    lvdt_data.columns = ["LVDT"]
    print(lvdt_data)

    plt.plot(convert_cal_rf(rfc_data["RF"]), label="RF")
    plt.plot(convert_cal_lvdt(lvdt_data["LVDT"]), label="LVDT")
    plt.xlabel("Measurement number")
    # plt.xlabel("Frequency (Hz) converted to Height (mm)")
    plt.ylabel("Height (mm)")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, lf.strip(".csv")+"_overTime.png"))

    plt.show()


def fit_sine_to_height():
    pass



if __name__ == "__main__":
    folder_0p5 = r"C:\Users\r.hawke\PycharmProjects\CetoniSP\data_files\2020-10-23 TriangleWaves 0.5mL"
    folder_1Hz = r"C:\Users\r.hawke\PycharmProjects\CetoniSP\data_files\2020-10-23 TriangleWaves 1Hz"
    fname = "Tri_0.1_12_1603423105.3048003_LVDT.xlsx"

    sp1, df2 = read_in_data(folder, fname)

    rfcA = convert_cal_rf(df2['Frequency (Hz)'][2000:4000])
    rfcB = df2['Height (mm)'][2000:4000]
    lvdtA = df2['LVDT (V)'][2000:4000]
    lvdtB = convert_cal_lvdt(lvdtA, offset=-3.9)

    plt.plot(rfcB, label="RF_asHeight")
    plt.plot(rfcA, label="RF_fromFreq")
    plt.plot(lvdtA, label="LVDT_V")
    plt.plot(lvdtB, label="LVDT_asHeight")
    plt.legend()
    plt.show()

