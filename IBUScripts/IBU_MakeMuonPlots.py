import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataDir', type=str,default='/global/cfs/projectdirs/m4045/users/rhuang/T2K/onoffaxisdata/FormattedData_v11/', help='Input data directory')
parser.add_argument('--FDS', type=int, default=0, help="FDS number (-1 for none)")
parser.add_argument('--plotDir', type=str, default='./', help="Directory to store plots")
parser.add_argument('--resultFile', type=str, default='', help="IBU unfolded result file to analyze")

flags = parser.parse_args()
dataDir = flags.dataDir
fakeIdx = flags.FDS
plotDir = flags.plotDir
unfoldedResults = np.load(flags.resultFile)

data_truth_weight = np.load(dataDir+'mc_weights_truth_Nominal.npy')
data_truth = np.load(dataDir+'mc_vals_truth_ReactionCodesIncluded.npy')
data_pass_truth = np.load(dataDir+'mc_pass_truth_Nominal.npy')

if fakeIdx >= 0:
    data_truth_weight = np.load(dataDir+'mc_weights_truth_FakeDataStudy%d.npy' % fakeIdx)

analysisBins = []
with open("/global/cfs/projectdirs/m4045/users/rhuang/T2K/onoffaxisdata/analysis_binning.txt", 'r') as f:
    lines = f.readlines()
    for row in lines:
        analysisBins.append([float(val) for val in row.split()])

dataTruthBinned = [0 for val in analysisBins]
data_mask = data_truth[:,8]<3
tmpWeights = data_truth_weight[data_mask]
for idx,val in enumerate(data_truth[data_mask,0:2]):
    for binIdx, bins in enumerate(analysisBins):
        if val[0] >= bins[2] and val[0] <= bins[3] and val[1] >= bins[0] and val[1] <= bins[1]:
            dataTruthBinned[binIdx] += tmpWeights[idx]

chi2 = []
meanPreunfold = np.mean(unfoldedResults[:,0,:], axis=0)
sigmaPreunfold = np.std(unfoldedResults[:,0,:], axis=0)

for it in range(10):
    fig = plt.figure(dpi=200)

    meanUnfold = np.mean(unfoldedResults[:,it,:], axis=0)
    sigmaUnfold = np.std(unfoldedResults[:,it,:], axis=0)

    cov = np.cov(unfoldedResults[:,it,:].T)
    chi2.append(np.dot(meanUnfold - dataTruthBinned, np.dot(np.linalg.pinv(cov), meanUnfold - dataTruthBinned)))
    
    corr = np.copy(cov)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            corr[i,j] /= sigmaUnfold[i]*sigmaUnfold[j]
    plt.clf()
    plt.imshow(corr,origin='lower')
    plt.title("IBU 2D Muon Correlation Matrix, Iteration %d" % it)
    plt.colorbar()
    plt.savefig("%s/IBU_2DMuon_Correlation_Iter%d.png" % (plotDir, it), dpi=200)

    plt.clf()
    plt.errorbar([val + 0.5 for val in range(58)], meanUnfold / dataTruthBinned - 1, yerr=sigmaUnfold / dataTruthBinned, marker='o', linestyle='none', label="IBU Unfolded Result", capsize=2)
    _=plt.hist(np.array(range(len(analysisBins)))+0.5, bins=range(len(analysisBins)+1), weights=np.array(meanPreunfold) / np.array(dataTruthBinned) - 1, histtype="step", label="Pre-unfolding")
    plt.fill_between(range(len(analysisBins)+1), np.insert((meanPreunfold - sigmaPreunfold) / dataTruthBinned, 0, (meanPreunfold[0] - sigmaPreunfold[0]) / dataTruthBinned[0]) - 1, np.insert((meanPreunfold + sigmaPreunfold) / dataTruthBinned, 0, (meanPreunfold[0] + sigmaPreunfold[0]) / dataTruthBinned[0]) - 1,step='pre', color='k', alpha=0.15)
    plt.axhline(y=0,color='black')
    plt.xlabel("Kinematic Bin")
    plt.ylabel("MC / Data Ratio - 1")
    plt.title("Muon (p,cos $\\theta$) Truth Space")
    plt.legend()
    plt.savefig("%s/IBU_2DMuon_UnfoldedRatiosIter%d.png" % (plotDir, it), dpi=200)
    
plt.clf()
plt.plot(np.array(chi2) / len(analysisBins), label="$\chi^2$/dof")
plt.title(r"IBU Performance - 2D Muon $(p, \cos \theta)$")
plt.xlabel("Iteration number")
plt.ylabel("Performance")
plt.legend(loc='right')
plt.savefig("%s/IBU_2DMuon_Chi2.png" % plotDir, dpi=200)
