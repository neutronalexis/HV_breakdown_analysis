# HV_breakdown_analysis
Python scripts to analyze data from breakdown tests with high-voltage test setup at TRIUMF

Add run data and run file paths to the run list in breakdown.py, then run `python breakdown.py`.

The script will load the data from the run files, partition it into ramps, and then detect breakdowns by comparing the voltage-current curves of each ramp with a designated reference ramp in vacuum. All ramps will be plotted to PDF files.

If RGA scans during any ramps are found in the subfolder `RGAdata` the script will generate additional PDF plots with the RGA data.

Finally, all breakdown voltages and pressures are plotted into `Paschen.pdf`

Required Python libraries:
- pandas
- matplotlib
- scipy
