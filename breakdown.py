import sys
import pandas
import matplotlib.pyplot as plt
import matplotlib.patches
import datetime
import scipy.interpolate
import scipy.optimize
import numpy
from matplotlib.backends.backend_pdf import PdfPages
import glob
import multiprocessing
import itertools

# Select gas pressure in cell from one of three pressure gauges, based on their ranges, and correct offsets
def selectAndCorrectPressureGauge(df, pressureOffset):
    print('Pressure offsets: {0}, {1}, {2}'.format(pressureOffset['MKS 627F_mbar_[20mTorr]'], pressureOffset['MKS  626C_mbar_[20Torr]'], pressureOffset['MicroBee_CDM900_mbar_[50Torr]']))
    series = df['MKS 627F_mbar_[20mTorr]'] - pressureOffset['MKS 627F_mbar_[20mTorr]']
    series.mask(series < 0, 0., inplace = True)
    series.mask(series > 26e-3, df['MKS  626C_mbar_[20Torr]'] - pressureOffset['MKS  626C_mbar_[20Torr]'], inplace = True)
    series.mask(series > 26., df['MicroBee_CDM900_mbar_[50Torr]'] - pressureOffset['MicroBee_CDM900_mbar_[50Torr]'], inplace = True)
    series.mask(series > 66., 1000., inplace = True)
    return series

# read data from LabVIEW program into pandas dataframe,
# generating the datetime index, pressure, and voltage-ramp columns
def readLabVIEWdata(filenames):
    print(filenames)
    data = pandas.concat((pandas.read_csv(filename) for filename in filenames), ignore_index = True)
    data.dropna(axis = 0)
    data['dateTime'] = data.apply(lambda row: datetime.datetime(int(row['Year']), int(row['Month']), int(row['Day']), int(row['Hour']), int(row['Minute']), int(row['Seconds'])), axis = 1)
    data.set_index('dateTime', inplace = True)
    countRamps(data)
#    countSampleRateChanges(data)
#    print('Found {0} ramps'.format(data['ramp'].max()))
    return data

# read data from individual RGA scan file
def readRGAdata(filename):
    starttime = datetime.datetime.strptime(filename, 'RGAdata\\rga1_%b_%d_%Y_%I-%M-%S_%p.ana.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()
        endtime = datetime.datetime.strptime(lines[0], '%b %d, %Y    %I:%M:%S %p\r')
        
        for _ in range(5):
            f.readline()
            
        dataPointsInScan = int(lines[6].split(',')[1])
        units = lines[7].split(',')[1]
        noiseFloor = lines[8].split(',')[1]
        CEMstatus = lines[9].split(',')[1]
        
        for _ in range(12):
            f.readline()
            
        masses = []
        currents = []
        for line in lines[22:]:
            values = line.split(',')
            masses.append(float(values[0]))
            currents.append(float(values[1]))
            
    return (starttime, endtime), (numpy.array(masses), numpy.array(currents))

# convert individual date and time columns in LabVIEW dataframe row to Python DateTime
def toDateTime(row):
    return datetime.datetime(int(row['Year']), int(row['Month']), int(row['Day']), int(row['Hour']), int(row['Minute']), int(row['Seconds']))

# Detect voltage ramp start and stop by comparing a voltage sample to the preceding and following samples.
# Return 1 if a ramp has started or stopped, 0 otherwise.
# Parameter samples is a set of three samples to be compared.
def rampStartStop(samples):
    if samples[0] == samples[1] and samples[1] != samples[2]: # ramp starts
        return 1
    elif samples[0] != samples[1] and samples[1] == samples[2]: # ramp ends
        return 1
    else:
        return 0
        
# Detect voltage ramp start and stop by changes in sample rate (LabVIEW program increases sample rate during ramps).
# This was used before there was a dedicated column indicating ramping.
# Return 1 if a ramp has started or stopped, 0 otherwise.
# Parameter samples is a set of three samples to be compared.
def sampleRateChange(samples):
    if samples[1] - samples[0] > 1 and samples[2] - samples[1] < 1: # sample rate increased
        return 1
    elif samples[1] - samples[0] < 1 and samples[2] - samples[1] > 1: # sample rate reduced
        return 1
    else:
        return 0
        
# Assign a unique number to each ramp period by counting the number of changes in the "ramping?" column of a LabVIEW dataframe
# Non-ramp periods will typically have an even number (starting at 0), ramp period will typically have an odd number.
# Before the "ramping?" columns was added, this was done using the rampStartStop method above.
def countRamps(dataframe):
    dataframe['ramp'] = dataframe['ramping?'].diff().abs().cumsum()
#    dataframe['ramp'] = dataframe['PS_SetVoltage_kV'].rolling(window = 3, min_periods = 3, center = True).apply(rampStartStop, raw = True).cumsum()
    
# Assign a unique number to each ramp period by counting the number of changes in sample rate using the sampleRateChange method above.
# Non-ramp periods will have an even number (starting at 0), ramp period will have an odd number.
# This has become obsolete since the "ramping?" column was added.
def countSampleRateChanges(dataframe):
    dataframe['srChanged'] = dataframe['SecondsElapsed'].rolling(window = 3, min_periods = 3, center = True).apply(sampleRateChange, raw = True).cumsum()
    
# Parameterized Paschen curve
def Paschen(pd, A, B, C):
    return numpy.where(numpy.log(pd + A) + C > 0, B*(pd + A)/(numpy.log(pd + A) + C), numpy.inf)

# Baseline of the current-voltage curve. This will be fit to the reference curve measured in vacuum
# to compensate for the strange behavior of the Gamma HV power supply.
# V: voltage; a,b,c,d: fit parameters
def Baseline(V, a, b, c, d):
    return -a*numpy.log(V, where = V > 0) + b*V**2 + c*V - d
     
# Go through all ramp periods and look for breakdowns by comparing the current-voltage curve to the reference curve.
# If their difference surpasses a threshold we classify it as a breakdown.
# filename: file name for output
# data: dataframe containing LabVIEW data
# referenceRampIndex: number of ramp period containing the reference curve in vacuum
def FindBreakdowns(filename, data, referenceRampIndex, gas, plot = False):
    breakdownPressures = []
    breakdownVoltages = []

    if plot:
        pdf = PdfPages(filename + '.pdf')

    # Get reference ramp and fit baseline. Calculate difference of measure current-voltage curve to baseline and store it in "corrected" column
    reference = data[(data['ramp'] == referenceRampIndex) & (data['PS_Voltage_kV'] > 0.3) & (data['PS_Voltage_kV'] < 70)]
    blopt, blcov = scipy.optimize.curve_fit(Baseline, reference['PS_Voltage_kV'], reference['PS_Current_mA'])
#    print(blopt, numpy.sqrt(numpy.diag(blcov)))
    data['baseline'] = Baseline(data['PS_Voltage_kV'], *blopt)
    data['corrected'] = (data['PS_Current_mA'] - data['baseline'])#.rolling(50, center = True).mean()
    reference = data[data['ramp'] == referenceRampIndex]
    pressureOffsets = reference.mean(numeric_only = True, skipna = True) # determine offsets of pressure gauges by averaging over all data during vacuum reference measurement
    data['pressure'] = selectAndCorrectPressureGauge(data, pressureOffsets)
    
    if plot:
        fig, axes = plt.subplots()
        reference.plot(x = 'PS_Voltage_kV', y = 'PS_Current_mA', ax = axes)
        reference.plot(x = 'PS_Voltage_kV', y = 'baseline', ax = axes, label = '-{0:.3g} ln(V) + {1:.3g} V$^2$ + {2:.3g} V - {3:.3g}'.format(*blopt))
        #axes.set_xscale('log')
        axes.set_xlabel('Voltage (kV)')
        axes.set_ylabel('Current (mA)')
        axes.set_title('{0}: reference curve in vacuum'.format(referenceRampIndex))
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    breakdownVoltages = []
    breakdownPressures = []
    rampIndex = 0
    # loop over all ramp periods
#    print(data['ramp'].max())
    for rampIndex in range(int(data['ramp'].max()) + 1):
        ramp = data[(data['ramp'] == rampIndex) & (data['PS_Voltage_kV'] > 0.1)]
        error = ''
        
        if len(ramp) < 100 or (ramp['PS_SetVoltage_kV'] == ramp['PS_SetVoltage_kV'].iloc[0]).all(): # skip if voltage in ramp period is constant
            #print(rampIndex, 'No ramp')
            continue

        # determine breakdown voltage based on three different current thresholds, these will be used to estimate uncertainty in breakdown voltage
        breakdown = ramp[ramp['corrected'] < 1e-4]['PS_Voltage_kV'].max() # find last sample where current is below lower threshold (-0.1uA)
        breakdown_hi = ramp[ramp['corrected'] < 2e-4]['PS_Voltage_kV'].max() # find last sample where current is below upper threshold (0.1uA)
        breakdown_lo = breakdown - (breakdown_hi - breakdown)
        
        # calculate average min and max gas pressure during breakdown
        breakdownData = ramp[(ramp['PS_Voltage_kV'] >= breakdown_lo) & (ramp['PS_Voltage_kV'] <= breakdown_hi)]
        pressure_lo = breakdownData['pressure'].min()
        pressure_hi = breakdownData['pressure'].max()

        if ramp['PS_Voltage_kV'].diff().max() > 1: # skip if there was a jump in voltage
            error = 'Voltage jump'    
        elif numpy.isnan(breakdown_lo) or numpy.isnan(breakdown_hi) or breakdown_hi == ramp['PS_Voltage_kV'].max():
            error = 'No breakdown found' # if the found breakdown is the last sample in the data it probably wasn't real
        elif breakdown_hi > 70. or breakdown_lo < 0.1:
            # above 70kV we cannot be sure if we found a breakdown in the cell or in the insulation vacuum
            # Below 0.1kV we are dominated by noise
            error = 'Breakdown outside sensible range'
        elif breakdown_hi - breakdown_lo > 1 and breakdown_hi - breakdown_lo > 0.5*breakdown:
            error = 'Large uncertainty'
            
        if plot:
            # plot current-voltage curves for found breakdowns, store in pdf file
            ax = ramp.plot(x = 'PS_Voltage_kV', y = ['PS_Current_mA', 'baseline', 'corrected'], xlabel = 'Voltage (kV)', ylabel = 'Current (mA)')
            ramp.plot(x = 'PS_Voltage_kV', y = 'pressure', ax = ax, secondary_y = True, alpha = 0.2)
            ax.right_ax.set_ylabel('Pressure (mbar)')
            lims = ax.get_ylim()
            if not numpy.isnan(breakdown_lo):
                ax.vlines(breakdown_lo, lims[0], lims[1], colors = 'k', linestyle = '--')
            if not numpy.isnan(breakdown_hi):
                ax.vlines(breakdown_hi, lims[0], lims[1], colors = 'k')
            ax.set_title('{0}: {1:.3g} mbar {2} - {3}'.format(rampIndex, (pressure_lo + pressure_hi)/2, gas, error))
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        if not error:
            # add breakdown voltage and gas pressure to list
            breakdownVoltages.append([breakdown_lo, breakdown_hi])
            breakdownPressures.append([pressure_lo, pressure_hi])
#        else:
#            print(rampIndex, error)

    if plot:
        pdf.close()
    return breakdownPressures, breakdownVoltages


def readDataAndFindBreakdowns(gas, filenames, electrodeDistance, referenceRampIndex, marker):
    data = readLabVIEWdata(filenames)
    pressures, voltages = FindBreakdowns(filenames[0], data, referenceRampIndex, gas, True)
    
    # Read, label, and plot any RGA scans that were performed during the breakdown measurements.
    # This became obsolete when RGA data was included in the LabVIEW readout.
    RGAdata = [readRGAdata(filename) for filename in glob.glob('RGAdata\*.txt')]
    RGAscans = [(scan) for scan in RGAdata if not data[scan[0][0]:scan[0][1]].empty]
    if RGAscans:
        print('Found {0} RGA scans'.format(len(RGAscans)))
        with PdfPages(filenames[0] + '_RGA.pdf') as pdf:
            for scan in RGAscans:
                scanData = data[scan[0][0]:scan[0][1]]
                fig, axes = plt.subplots(figsize = (16, 9))
                pressure = scanData['pressure'].mean()
                voltage = scanData['PS_Voltage_kV'].mean()
                axes.semilogy(scan[1][0], scan[1][1])
                total = sum(scan[1][1])
                for mass, width in zip([2, 4, 14, 16, 17, 18, 28, 32, 40, 44, 65, 130], [10, 5, 5, 5, 5, 5, 5, 5, 5, 5, 35, 65]):
                    x = mass*10 - 10
                    if x + width < len(scan[1][1]):
                        partialpressure = scan[1][1][x - width : x + width]
                        axes.annotate('{0:.1f}%'.format(sum(partialpressure)*100/total), (mass, max(partialpressure)))
                helium = sum(scan[1][1][25:34])
                nitrogen = sum(scan[1][1][265:274])
                xenon = sum(scan[1][1][1225:1354]) + sum(scan[1][1][605:674])
                axes.set_title('{0:.3g} mbar cell pressure, {1:.3g} kV\n{2:.3g} torr, {3:.1f}% He, {4:.1f}% N2, {5:.1f}% Xe'.format(pressure, voltage, total, helium/total*100., nitrogen/total*100., xenon/total*100.))
                axes.set_ylim(1e-10, 1e-5)
                pdf.savefig()
                plt.close()
                    
    return pressures, voltages
    
        

if __name__ == '__main__':
    # list of breakdown measurements
    runs = [
           # Label                 List of data files                                   electrode distance
           #                                                                                   index of reference ramp in vacuum
           #                                                                                      matplotlib marker style and color
            ('N$_2$ (-, glass)',  ['THELabSlowControl_2021_08_11_14_58_47.txt'],        4.78,  1, 'bo'),
            ('He (-, glass)',     ['THELabSlowControl_2021_08_12_15_33_05.txt'],        4.78,  1, 'go'),
            ('Xe (-, glass)',     ['THELabSlowControl_2021_08_16_12_46_25.txt'],        4.78,  1, 'ro'),
            ('Xe (-, glass)',     ['THELabSlowControl_2021_08_20_13_29_22.txt'],        4.78,  1, 'rs'),
            ('He (-, glass)',     ['THELabSlowControl_2021_08_23_16_02_57.txt'],        4.78,  1, 'gs'),
            ('He+H$_2$+N$_2$/CO', ['THELabSlowControl_2021_08_26_14_17_09_helium.txt'], 4.78,  1, 'g+'),
            ('Xe (-, glass)',     ['THELabSlowControl_2021_08_26_14_17_09_xenon.txt'],  4.78,  1, 'rv'),
            ('N$_2$ (-, glass)',  ['THELabSlowControl_2021_09_07_18_21_33.txt'],        4.78, 51, 'bs'),
            ('He (-, glass)',     ['THELabSlowControl_2021_09_13_15_22_33.txt'],        4.78,  1, 'g^'),
            ('Xe (-, glass)',     ['THELabSlowControl_2021_09_15_15_14_36.txt'],        4.78, 41, 'r^'),
            ('He (-, glass)',     ['THELabSlowControl_2021_09_15_17_58_30.txt'],        4.78, 27, 'g^'),
            ('Xe (-, PS)',        ['THELabSlowControl_2023_03_20_16_27_52.txt'],        4.78, 55, 'r>'),
            ('Xe (-, PS)',        ['THELabSlowControl_2023_03_24_18_54_35.txt', 'THELabSlowControl_2023_03_24_19_47_00.txt', 'THELabSlowControl_2023_03_24_20_06_07.txt'], 4.78, 33, 'r<'),
            ('Xe (+, PS)',        ['THELabSlowControl_2023_03_27_15_00_05.txt'],        4.78,  1, 'r^'),
            ('N$_2$ (+, PS)',     ['THELabSlowControl_2023_03_30_10_18_44.txt'],        4.78, 39, 'bv'),
            ('He (+,PS)',         ['THELabSlowControl_2023_04_05_15_25_52.txt'],        4.78,  1, 'gv'),
            ('N$_2$ (+,PS)',      ['THELabSlowControl_2023_05_11_14_31_34.txt'],        7.28,  0, 'b^'),
			('Xe (+,PS)',         ['THELabSlowControl_2023_05_12_11_31_19.txt'],        7.28,  1, 'rD'),
            ('He (+,PS)',         ['THELabSlowControl_2023_05_15_09_50_37.txt'],        7.28,  1, 'gD'),
           ]
    
    # loop over selection of LabVIEW data and find breakdowns, assigning gas type, reference ramp number, and plot marker style for each
    
    with multiprocessing.Pool() as p:                           # multithreaded reading and analysis. 
        breakdowns = p.starmap(readDataAndFindBreakdowns, runs) # Replace these two lines with single-threaded analysis below for debugging
    #breakdowns = itertools.starmap(readDataAndFindBreakdowns, runs)
    
    pfig, paxes = plt.subplots(1, 2, figsize = (14, 5))
    
    for r, b in zip(runs, breakdowns):
        pressures = b[0]
        voltages = b[1]
        gas = r[0]
        electrodeDistance = r[2]
        marker = r[4]
        x = numpy.array([(p[0] + p[1])/2 for p in pressures])
        xerr = numpy.array([abs(p[0] - p[1])/2 for p in pressures])
        y = numpy.array([(V[0] + V[1])/2 for V in voltages])
        yerr = numpy.array([(V[1] - V[0])/2 for V in voltages])
        paxes[0].errorbar(x * electrodeDistance, y, xerr = xerr * electrodeDistance, yerr = yerr, fmt = marker, label = gas, linewidth = 1, markersize = 3)
        paxes[1].errorbar(x, y / electrodeDistance, xerr = xerr, yerr = yerr / electrodeDistance, fmt = marker, label = gas, linewidth = 1, markersize = 3)
    
    
    # plot breakdown voltages vs pressure*distance
    paxes[0].set_xlabel('Pressure * Distance (mbar cm)')
    paxes[1].set_xlabel('Pressure (mbar)')
    paxes[0].set_ylabel('Breakdown voltage (kV)')
    paxes[1].set_ylabel('Breakdown field (kV/cm)')
    paxes[0].set_xlim(1e-2, 4e2)
    paxes[1].set_xlim(1e-3, 1e2)
    paxes[0].set_ylim(0.2, 100)
    paxes[1].set_ylim(0.05, 20)
    #paxes.fill_between([1e-3*10, 1e-2*15], [100., 100.], [200., 200], color = black, hatch = '//', fill = False, alpha = 0.2, label = 'Comag. (voltage)')
    
    # add secondary axes with breakdown field vs pressure
#    x2 = paxes.secondary_xaxis('top', functions = (lambda pd: pd/electrodeDistance, lambda p: p*electrodeDistance))
#    x2.set_xlabel('Pressure (mbar)')
#    y2 = paxes.secondary_yaxis('right', functions = (lambda V: V/electrodeDistance, lambda E: E*electrodeDistance))
#    y2.set_ylabel('Breakdown field (kV/cm)')
    #paxes.fill_between([1e-3*10, 1e-2*15], [10*electrodeDistance, 10*electrodeDistance], [20*electrodeDistance, 20*electrodeDistance], color = black, hatch = '\\\\', fill = False, alpha = 0.2, label = 'Comag. (field)')
    
    for ax in paxes:
      ax.set_xscale('log')
      ax.set_yscale('log')
      ax.legend(loc = 'upper right', ncol = 2)
    
    plt.tight_layout()
    
    #try:
    #    popt, pcov = scipy.optimize.curve_fit(Paschen, x, y, sigma = yerr, absolute_sigma = True)
    #    x = numpy.logspace(-1, 2.5, num = 1000)
    #    paxes.plot(x, Paschen(x, *popt))
    #    print(popt, numpy.sqrt(numpy.diag(pcov)))
    #except:
    #    continue
    pfig.savefig('Paschen.pdf')
    plt.close()
    

