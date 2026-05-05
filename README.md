# OmnifoldT2K

Running the OmniFold procedure on a T2K public dataset for a muon-neutrino CC0pi analysis (https://zenodo.org/doi/10.5281/zenodo.15183090)

FormatData.py will read in the public data files and put them into numpy array formats suitable for use as input to OmniFold. This script includes data standardization, outlier removal, and reco/truth matching.

./runOmnifold.sh will run the full procedure, using config_omnifold.json as a simple configuration. This script is also configured to generate diagnostic plots showing the reweighting result after each step 1 and step 2. It will also save the pull and push weights in numpy array format after each iteration. Applying the pull weights as reweighting factors on the reco-level simulated events should result in a distribution similar to the data. Applying the push weights as reweighting factors on the truth-level events provides the unfolded result. The relevant files are:

config_omnifold.json:
	FILES_MC_RECO: Reconstructed simulated events with shape (nEventsReco, nFeaturesReco)
	FILES_MC_GEN: Truth simulated events with shape (nEventsTruth, nFeaturesTruth)
	FILES_MC_FLAG_RECO: Flags for reco events passing selection. This can be all True if the selection was preapplied to the MC_RECO file.
	FILE_MC_FLAG_GEN: Flags for truth events that are reconstructed. Should have length (nEventsTruth) and sum to nEventsReco, such that applying this flag to the MC_GEN file gives the events corresponding to events in the MC_RECO file.
	FILE_DATA_RECO: Data events with shape (nEventsData, nFeaturesReco)
	FILE_DATA_FLAG_RECO: Flags for data events passing selection
	FILE_DATA_WEIGHT: Weights for data
	FILE_MC_RECO_WEIGHT: Weights for reco-level simulated events
	FILE_MC_GEN_WEIGHT: Weights for truth-level simulated events
	NTRIAL: Number of classifiers to train per iteration. Weights are obtained by averaging the results of each classifier

t2k.py: Executable script to set up the procedure and load relevant configurations

omnifold.py: OmniFold procedure for unfolding. Based off of script from https://github.com/ViniciusMikuni/AlephOmniFold

utils.py: Various utilities for data handling and plot creation

