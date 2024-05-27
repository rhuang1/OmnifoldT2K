import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fds', type=int,default=0, help='Fake Dataset number')
parser.add_argument('--resultsDir', type=str,default="/global/homes/r/rhuang94/OmniFold/AnalyzeResults/BinnedResults/", help='Directory with binned results')
parser.add_argument('--dataDir', type=str, default="/global/cfs/projectdirs/m4045/users/rhuang/T2K/onoffaxisdata/FormattedData_v12/", help="Data directory")
parser.add_argument('--plotDir', type=str, default="./", help="Directory to save plots")
parser.add_argument('--finalIt', type=int, default=9, help="Final iteration to print chi2 values for")

flags = parser.parse_args()

dataDir = flags.dataDir
fakeIdx = flags.fds
plotDir = flags.plotDir
resultsDir = flags.resultsDir

data_truth = np.load(dataDir+'mc_vals_truth_ReactionCodesIncluded.npy')[0::2]
data_pass_truth = np.load(dataDir+'mc_pass_truth_Nominal.npy')[0::2]
data_truth_weight = np.load(dataDir+'mc_weights_truth_FakeDataStudy%d.npy' % fakeIdx)[0::2]

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

# Evaluate chi2 for both IBU and Omnifold
ibuMuon = np.load("%s/IBU_2DMuon_FakeData%d_SystStatDataSplit.npy" % (resultsDir, fakeIdx))
ibuChi2 = [[] for i in range(4)]
omniChi2 = [[] for i in range(4)]
uniChi2 = [[] for i in range(4)]
for it in range(ibuMuon.shape[2]):
    meanUnfold = np.mean(ibuMuon[0,:,it,:], axis=0)
    cov = np.cov(ibuMuon[0,:,it,:].T)
    ibuChi2[0].append(np.dot(meanUnfold - dataTruthBinned, np.dot(np.linalg.pinv(cov), meanUnfold - dataTruthBinned)) / len(analysisBins))
omniChi2[0].append(ibuChi2[0][0])
omniMuon = np.load("%s/Omnifold_2DMuon_FDS%s_SystStatDataSplit_NNAverage.npy" % (resultsDir, fakeIdx))
for it in range(omniMuon.shape[0]):
    thisSlice = omniMuon[it,:,:]
    meanUnfold = np.mean(thisSlice, axis=0)
    cov = np.cov(thisSlice.T)
    cov = np.cov(ibuMuon[0,:,6,:].T)
    omniChi2[0].append(np.dot(meanUnfold - dataTruthBinned, np.dot(np.linalg.pinv(cov), meanUnfold - dataTruthBinned)) / len(analysisBins))
uniChi2[0].append(ibuChi2[0][0])
#uniMuon = np.load("%s/Omnifold_2DMuon_FDS%s_ReqProton_SystStatDataSplit_NNAverage.npy" % (resultsDir, fakeIdx))
uniMuon = np.load("%s/Omnifold_2DMuon_FDS%s_Multifold_SystStatDataSplit_NNAverage.npy" % (resultsDir, fakeIdx))
for it in range(uniMuon.shape[0]):
    thisSlice = uniMuon[it,:,:]
    meanUnfold = np.mean(thisSlice, axis=0)
    cov = np.cov(thisSlice.T)
#    cov = np.cov(omniMuon[it,:,:].T)
    uniChi2[0].append(np.dot(meanUnfold - dataTruthBinned, np.dot(np.linalg.pinv(cov), meanUnfold - dataTruthBinned)) / len(analysisBins))
    
varSTV = ["Pt", "Alpha", "Phi"]
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
    dataSTVTruth,xx = np.histogram(dataTruthVar[data_truth_mask],weights=data_truth_weight[data_truth_mask],bins=bins[varIdx])

    ibuResult = np.load("%s/IBU_Binned%s_FakeData%s_SystStatDataSplit.npy" % (resultsDir, varSTV[varIdx],fakeIdx))
    for it in range(ibuResult.shape[2]):
        meanUnfold = np.mean(ibuResult[0,:,it,:], axis=0)
        cov = np.cov(ibuResult[0,:,it,:].T)
        ibuChi2[varIdx+1].append(np.dot(meanUnfold - dataSTVTruth, np.dot(np.linalg.pinv(cov), meanUnfold - dataSTVTruth)) / (len(bins[varIdx])-1))

    omniChi2[varIdx+1].append(ibuChi2[varIdx+1][0])
    omniResult = np.load("%s/Omnifold_Binned%s_FDS%s_SystStatDataSplit_NNAverage.npy" % (resultsDir, varSTV[varIdx],fakeIdx))
    for it in range(omniResult.shape[0]):
        meanUnfold = np.mean(omniResult[it,:,:], axis=0)
        cov = np.cov(omniResult[it,:,:].T)
        omniChi2[varIdx+1].append(np.dot(meanUnfold - dataSTVTruth, np.dot(np.linalg.pinv(cov), meanUnfold - dataSTVTruth)) / (len(bins[varIdx])-1))

    uniTitles = ['Unifold3', 'Unifold4', 'Unifold5']
#    uniTitles = ['ReqProton', 'ReqProton','ReqProton']
#    uniTitles = ['Multifold','Multifold','Multifold']
    uniResult = np.load("%s/Omnifold_Binned%s_FDS%s_%s_SystStatDataSplit_NNAverage.npy" % (resultsDir, varSTV[varIdx],fakeIdx, uniTitles[varIdx]))
    uniChi2[varIdx+1].append(ibuChi2[varIdx+1][0])
    uniMask = np.array([i for i in range(100) if i != 64 and i != 94])
#    uniMask = np.array([i for i in range(100)])
    for it in range(omniResult.shape[0]):
        meanUnfold = np.mean(uniResult[it,uniMask,:], axis=0)
        cov = np.cov(uniResult[it,uniMask,:].T)
#        cov = np.cov(omniResult[it,:,:].T)
        uniChi2[varIdx+1].append(np.dot(meanUnfold - dataSTVTruth, np.dot(np.linalg.pinv(cov), meanUnfold - dataSTVTruth)) / (len(bins[varIdx])-1))
        
# Plot results for IBU only, Omnifold only, and together
colors = ['blue', 'orange', 'green', 'red']
labels = ['Muon (p, cos $\\theta$)', '$\delta p_{T}$', '$\delta \\alpha_{T}$', '$\delta \phi_{T}$']
plt.clf()
for idx in range(4):
    plt.plot(ibuChi2[idx], color=colors[idx], linestyle='dashed', label=labels[idx])
plt.title("IBU FDS %d $\chi^2$/dof Convergence" % fakeIdx)
plt.ylabel("$\chi^2$/dof")
plt.xlabel("Number of iterations")
plt.legend(loc='upper right')
plt.savefig("%s/Chi2FDS%d_IBU.png" % (plotDir, fakeIdx), dpi=200)

plt.clf()
for idx in range(4):
    plt.plot(omniChi2[idx], color=colors[idx], linestyle='solid', label="Omnifold " + labels[idx])
    plt.plot(uniChi2[idx], color=colors[idx], linestyle='dotted', label="Unifold " + labels[idx])
plt.title("Omnifold FDS %d $\chi^2$/dof Convergence" % fakeIdx)
plt.ylabel("$\chi^2$/dof")
plt.xlabel("Number of iterations")
plt.legend(loc='upper right')
plt.savefig("%s/Chi2FDS%d_Omnifold.png" % (plotDir, fakeIdx), dpi=200)

plt.clf()
for idx in range(4):
    plt.plot(omniChi2[idx], color=colors[idx], linestyle='solid', label="Omnifold " + labels[idx])
    plt.plot(uniChi2[idx], color=colors[idx], linestyle='dotted', label="Unifold " + labels[idx])
    plt.plot(ibuChi2[idx], color=colors[idx], linestyle='dashed', label="IBU " + labels[idx])
plt.title("FDS %d $\chi^2$/dof Convergence Comparison" % fakeIdx)
plt.ylabel("$\chi^2$/dof")
plt.xlabel("Number of iterations")
plt.legend(loc='upper right')
plt.savefig("%s/Chi2FDS%d_Comparison.png" % (plotDir, fakeIdx), dpi=200)

finalIt = flags.finalIt
print("Muon chi2 (Prior, IBU, Omnifold, Unifold): (%.1f, %.1f, %.1f, %.1f)" % (ibuChi2[0][0]*58, ibuChi2[0][finalIt]*58, omniChi2[0][finalIt]*58, uniChi2[0][finalIt]*58))
print("dpT chi2 (Prior, IBU, Omnifold, Unifold): (%.1f, %.1f, %.1f, %.1f)" % (ibuChi2[1][0]*15, ibuChi2[1][finalIt]*15, omniChi2[1][finalIt]*15, uniChi2[1][finalIt]*15))
print("dalphaT chi2 (Prior, IBU, Omnifold, Unifold): (%.1f, %.1f, %.1f, %.1f)" % (ibuChi2[2][0]*19, ibuChi2[2][finalIt]*19, omniChi2[2][finalIt]*19, uniChi2[2][finalIt]*19))
print("dphiT chi2 (Prior, IBU, Omnifold, Unifold): (%.1f, %.1f, %.1f, %.1f)" % (ibuChi2[3][0]*15, ibuChi2[3][finalIt]*15, omniChi2[3][finalIt]*15, uniChi2[3][finalIt]*15))

        

    


