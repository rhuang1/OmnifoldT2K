# OmnifoldT2K

Running the OmniFold procedure on a T2K public dataset for a muon-neutrino CC0pi analysis (https://zenodo.org/doi/10.5281/zenodo.15183090)

FormatData.py will read in the public data files and put them into numpy array formats suitable for use as input to OmniFold. This script includes data standardization, outlier removal, and reco/truth matching.

./runOmnifold.sh will run the full procedure, using config_omnifold.json as a simple configuration. This script is also configured to generate diagnostic plots showing the reweighting result after each step 1 and step 2. It will also save the pull and push weights in numpy array format after each iteration. The relevant files are:

t2k.py: Executable script to set up the procedure and load relevant configurations

omnifold.py: OmniFold procedure for unfolding. Based off of script from https://github.com/ViniciusMikuni/AlephOmniFold

utils.py: Various utilities for data handling and plot creation
