import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fds', type=int,default=0, help='Fake Dataset number')
parser.add_argument('--resultsDir', type=str,default="/global/homes/r/rhuang94/OmniFold/AnalyzeResults/BinnedResults/", help='Directory with binned results')
parser.add_argument('--dataDir', type=str, default="/global/cfs/projectdirs/m4045/users/rhuang/T2K/onoffaxisdata/FormattedData_v12/", help="Data directory")
parser.add_argument('--plotDir', type=str, default="./", help="Directory to save plots")
parser.add_argument('--ibuIter', type=int, default=4, help="Final IBU iteration to plot")
parser.add_argument('--omniIter', type=int, default=6, help="Final Omnifold iteration to plot")

flags = parser.parse_args()

dataDir = flags.dataDir
fakeIdx = flags.fds
plotDir = flags.plotDir
resultsDir = flags.resultsDir
ibuIt = flags.ibuIter
omniIt = flags.omniIter

data_truth = np.load(dataDir+'mc_vals_truth_ReactionCodesIncluded.npy')
data_pass_truth = np.load(dataDir+'mc_pass_truth_Nominal.npy')
data_truth_weight = np.load(dataDir+'mc_weights_truth_FakeDataStudy%d.npy' % fakeIdx)

analysisBins = []
with open("/global/cfs/projectdirs/m4045/users/rhuang/T2K/onoffaxisdata/analysis_binning.txt", 'r') as f:
    lines = f.readlines()
    for row in lines:
        analysisBins.append([float(val) for val in row.split()])

data_mask = data_truth[:,8] < 3
dataTruthBinned = [0 for val in analysisBins]
for idx,val in enumerate(data_truth[:,0:2]):
    mask = data_mask[idx]
    if mask:
        for binIdx, bins in enumerate(analysisBins):
            if val[0] >= bins[2] and val[0] <= bins[3] and val[1] >= bins[0] and val[1] <= bins[1]:
                dataTruthBinned[binIdx] += data_truth_weight[idx]
                break

# Plot Muon result
ibuMuon = np.load("%s/IBU_2DMuon_FakeData%d_SystStat.npy" % (resultsDir, fakeIdx))
omniMuon = np.load("%s/Omnifold_2DMuon_FDS%s_SystStat_NNAverage.npy" % (resultsDir, fakeIdx))
meanOmni = np.mean(omniMuon[omniIt,:,:], axis=0)
sigmaOmni = np.std(omniMuon[omniIt,:,:], axis=0)
meanPreunfold = np.mean(ibuMuon[0,:,0,:], axis=0)
sigmaPreunfold = np.std(ibuMuon[0,:,0,:], axis=0)
meanIBU = np.mean(ibuMuon[0,:,ibuIt,:], axis=0)
sigmaIBU = np.std(ibuMuon[0,:,ibuIt,:], axis=0)

plt.clf()
plt.errorbar([val + 0.5 for val in range(58)], meanIBU / dataTruthBinned - 1, yerr=sigmaIBU / dataTruthBinned, marker='o', markersize=4, linestyle='none', label="IBU Result", capsize=2)
plt.errorbar([val + 0.5 for val in range(58)], meanOmni / dataTruthBinned - 1, yerr=sigmaOmni / dataTruthBinned, marker='o', linestyle='none', label="Omnifold Result", markersize=4, capsize=2)
_=plt.hist(np.array(range(len(analysisBins)))+0.5, bins=range(len(analysisBins)+1), weights=np.array(meanPreunfold) / np.array(dataTruthBinned) - 1, histtype="step", label="Pre-unfolding")
plt.axhline(y=0,color='black')
plt.xlabel("True Muon (p,cos $\\theta$) Kinematic Bin")
plt.ylabel("MC / Data Ratio - 1")
plt.title("Muon (p,cos $\\theta$) Unfolded Ratio to Truth\nFake Dataset %d" % fakeIdx)
plt.legend(loc='upper right')
plt.xlim(0,58)
plt.ylim(-0.5,0.5)
plt.savefig("%s/2DMuon_UnfoldedRatioFDS%d.png" % (plotDir, fakeIdx), dpi=200, bbox_inches='tight')

plt.clf()
plt.errorbar([val + 0.5 for val in range(58)], meanIBU, yerr=sigmaIBU, marker='o', linestyle='none', label="IBU Result", capsize=2, markersize=4, zorder=0)
plt.errorbar([val + 0.5 for val in range(58)], meanOmni, yerr=sigmaOmni, marker='o', linestyle='none', label="Omnifold Result", capsize=2, markersize=4, zorder=1)
_=plt.hist(np.array(range(len(analysisBins)))+0.5, bins=range(len(analysisBins)+1), weights=np.array(dataTruthBinned), histtype="step", label="Data Truth", zorder=2)
plt.axhline(y=0,color='black')
plt.ylabel("Events / Bin")
plt.xlabel("Muon (p,cos $\\theta$) Kinematic Bin")
plt.title("Muon (p,cos $\\theta$) Unfolded Distributions\nFake Dataset %d" % fakeIdx)
plt.legend()
plt.yscale("log")
plt.xlim(0,58)
plt.ylim(10, 2e4)
plt.savefig("%s/2DMuon_UnfoldedDistributionFDS%d.png" % (plotDir, fakeIdx), dpi=200, bbox_inches='tight')

varSTV = ["Pt", "Alpha", "Phi"]
titles = ["$\delta p_{T}$", "$\delta \\alpha_{T}$", "$\delta \phi_{T}$"]
units = ["[MeV/c]", "[Radians]", "[Radians]"]
ptBin = [0,50,100,150,200,250,300,350,400,450,500,560,630,750,30000]
alphaBin = np.linspace(0,3.14,19)
phiBin = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.725,0.875, 1.05, 1.275, 1.6, 2, 2.55,3.15]
bins = [ptBin, alphaBin, phiBin]
for varIdx in range(3):
    data_truth_mask = [all(tup) for tup in zip(data_truth[:,8]<3, data_truth[:,3]>450)]
    dataTruthVar = data_truth[:,10]
    if varIdx == 1:
        dataTruthVar = np.arccos(np.clip(data_truth[:,11],-1,1))
    elif varIdx == 2:
        dataTruthVar = np.arccos(np.clip(data_truth[:,12],-1,1))
    bb,xx = np.histogram(dataTruthVar[data_truth_mask],weights=data_truth_weight[data_truth_mask],bins=bins[varIdx])

    ibuResult = np.load("%s/IBU_Binned%s_FakeData%s_SystStat.npy" % (resultsDir, varSTV[varIdx],fakeIdx))
    omniResult = np.load("%s/Omnifold_Binned%s_FDS%s_SystStat_NNAverage.npy" % (resultsDir, varSTV[varIdx],fakeIdx))

    meanUnfoldIBU = np.mean(ibuResult[0,:,ibuIt,:], axis=0)
    sigmaUnfoldIBU = np.std(ibuResult[0,:,ibuIt,:], axis=0)
    
    meanPreunfold = np.mean(ibuResult[0,:,0,:], axis=0)
    sigmaPreunfold = np.std(ibuResult[0,:,0,:], axis=0)

    meanUnfold = np.mean(omniResult[omniIt,:,:], axis=0)
    sigmaUnfold = np.std(omniResult[omniIt,:,:], axis=0)

    plotBins = 0.5*(xx[0:-1]+xx[1:])
    if varIdx == 0:
        plotBins[-1] = 1125
    plt.clf()
#    plt.errorbar([val + 0.5 for val in range(len(bins[varIdx])-1)], meanUnfoldIBU / bb - 1, yerr=sigmaUnfoldIBU / bb, marker='o', linestyle='none', label="IBU Result", capsize=2)
#    plt.errorbar([val + 0.5 for val in range(len(bins[varIdx])-1)], meanUnfold / bb - 1, yerr=sigmaUnfold / bb, marker='o', linestyle='none', label="Omnifold Result", capsize=2)
#    _=plt.hist(np.array(range(len(bins[varIdx])-1))+0.5, bins=range(len(bins[varIdx])), weights=np.array(meanPreunfold) / bb - 1, histtype="step", label="Pre-unfolding")
    plt.errorbar(plotBins, meanUnfoldIBU / bb - 1, yerr=sigmaUnfoldIBU / bb, marker='o', linestyle='none', label="IBU Result", capsize=2)
    plt.errorbar(plotBins, meanUnfold / bb - 1, yerr=sigmaUnfold / bb, marker='o', linestyle='none', label="Omnifold Result", capsize=2)
    plt.stairs(meanPreunfold / bb - 1, bins[varIdx], label="Pre-unfolding")
    plt.axhline(y=0,color='black')
    plt.legend(loc='lower left')
#    plt.xlabel("%s Kinematic Bin Index" % titles[varIdx])
    plt.xlabel("True %s %s" % (titles[varIdx], units[varIdx]))
    plt.ylabel("MC / Data Ratio - 1")
    plt.title("%s Unfolded Ratio to Truth\nFake Dataset %d" % (titles[varIdx], fakeIdx))

    if varIdx == 0:
        plt.xlim(0,1500)
    elif varIdx == 2 or varIdx == 1:
        plt.xlim(0,3.14)

    plt.savefig("%s/Binned%s_UnfoldedRatioFDS%d.png" % (plotDir, varSTV[varIdx], fakeIdx), dpi=150)

    plt.clf()
    plt.errorbar(plotBins, meanUnfoldIBU, yerr=sigmaUnfoldIBU, marker='o', linestyle='none', label="IBU Result", capsize=2)
    plt.errorbar(plotBins, meanUnfold, yerr=sigmaUnfold, marker='o', linestyle='none', label="Omnifold Result", capsize=2)
    plt.stairs(bb, bins[varIdx], label="Data Truth")
    #_=plt.hist(plotBins, bins=bins[varIdx], weights=np.array(meanPreunfold), histtype="step", label="Pre-unfolding")
    #plt.fill_between(range(len(bins[varIdx])), np.insert((meanPreunfold - sigmaPreunfold) / bb, 0, (meanPreunfold[0] - sigmaPreunfold[0]) / bb[0]) - 1, np.insert((meanPreunfold + sigmaPreunfold) / bb, 0, (meanPreunfold[0] + sigmaPreunfold[0]) / bb[0]) - 1,step='pre', color='k', alpha=0.15)
    if varIdx == 1:
        plt.legend(loc='upper left')
    else:
        plt.legend(loc='upper right')
    plt.xlabel("True %s %s" % (titles[varIdx], units[varIdx]))
    plt.ylabel("Events / Bin")
    plt.title("%s Unfolded Distribution\nFake Dataset %d" % (titles[varIdx], fakeIdx))

    if varIdx == 0:
        plt.xlim(0,1500)
    elif varIdx == 2 or varIdx == 1:
        plt.xlim(0,3.14)
    plt.savefig("%s/Binned%s_UnfoldedDistributionFDS%d.png" % (plotDir, varSTV[varIdx], fakeIdx), dpi=200)
