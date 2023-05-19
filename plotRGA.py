import sys
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import glob
import scipy.signal
import numpy

data = {}

with PdfPages('RGA.pdf') as pdf:
  for arg in sys.argv[1:]:
    for filename in glob.glob(arg):
      with open(filename, 'r') as f:
        line = f.readline()
        time = datetime.datetime.strptime(line, '%b %d, %Y  %I:%M:%S %p\r')
        
        for _ in range(5):
          f.readline()
          
        dataPointsInScan = int(f.readline().split(',')[1])
        units = f.readline().split(',')[1]
        noiseFloor = f.readline().split(',')[1]
        CEMstatus = f.readline().split(',')[1]
        
        for _ in range(12):
          f.readline()
          
        masses = []
        currents = []
        while True:
          line = f.readline()
          if not line:
            break
          values = line.split(',')
          masses.append(float(values[0]))
          currents.append(float(values[1]))
          
        data[time] = (masses, currents, CEMstatus)
          
  for time1, time2 in zip(list(data)[0::2], list(data)[1::2]):
    fig, axes = plt.subplots(2, 1, figsize = (16, 9))
    axes[0].plot(data[time1][0], data[time1][1])
    
    axes[1].plot(data[time2][0], data[time2][1])
    peak_indices, peak_properties = scipy.signal.find_peaks(data[time2][1], height = 1e-9, distance = 8)
    xs = numpy.take(data[time2][0], peak_indices)
    ys = numpy.take(data[time2][1], peak_indices)
#    axes[1].plot(xs, ys, 'o')
    for x, y in zip(xs, ys):
      axes[1].annotate(round(x), (x, y))
    
    for ax in axes:
      ax.set_ylim(1e-11, 1.5e-7)
    fig.suptitle('{0}, {1:.3g} torr'.format(time1.isoformat(), sum(data[time2][1])))
    pdf.savefig()

  fig, axes = plt.subplots(figsize = (16, 9))
  time1 = list(data)[3]
  time2 = list(data)[-1]
  x = data[time1][0]
  y = numpy.subtract(data[time1][1], data[time2][1])
  axes.plot(x, y)
  fig.suptitle('{0} background subtracted, {1:.1f}% xenon'.format(time1, sum(y[600:]/sum(y)*100)))
  pdf.savefig()
  
  fig, axes = plt.subplots(figsize = (16, 9))
  time1 = list(data)[-3]
  x = data[time1][0]
  y = numpy.subtract(data[time1][1], data[time2][1])
  axes.plot(x, y)
  fig.suptitle('{0} background subtracted, {1:.1f}% xenon'.format(time1, sum(y[600:]/sum(y)*100)))
  pdf.savefig()
