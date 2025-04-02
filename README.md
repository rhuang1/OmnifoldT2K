# OmnifoldT2K

Running the OmniFold procedure on an example dataset (data files currently stored on NERSC perlmutter)

./runOmnifold.sh will run the full procedure, using config_omnifold.json as a simple configuration. This script is also configured to generate diagnostic plots showing the reweighting result after each step 1 and step 2. The relevant files are:

t2k.py: Executable script to set up the procedure and load relevant configurations

omnifold.py: OmniFold procedure for unfolding

utils.py: Various utilities for data handling and plot creation