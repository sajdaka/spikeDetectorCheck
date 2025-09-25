A simple toolkit for automating spike detection and data preprocessing for EEG and photometry data in the Mattis Research Lab

Installation:
```console
git clone https://github.com/sajdaka/spikeDetectorCheck.git
cd spikeDetectorCheck

pip install -r requirements.txt
```
Run the GUI (Recommended):
```console
python main.py --gui
```
Run via CLI: 
(Example)
```console
python main.py --eeg-file data/recording.oebin --photometry-file data/photometry.ppd --plot
```

Usage Guide:

- run the gui command as shown above
- select the files and channel required for your usage
- select the preprocessing pipeline you wish to run for this iteration
- move to parameters and change as required
- finally, use Run Spike Detection under analysis and either select Show Plots to view in a browser or Export Results for the plots to be stored locally

Inputs:
- EEG folder (OpenEphys data output)
- Photometry file (.ppd)

Output:
- Plots containing both raw signal and preprocessed (.html)
