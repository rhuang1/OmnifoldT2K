import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataDir', type=str,default='/global/cfs/projectdirs/m4045/users/rhuang/T2K/onoffaxisdata/FormattedData_v11/', help='Input data directory')
parser.add_argument('--FDS', type=int, default=0, help="FDS number (-1 for none)")
parser.add_argument('--saveDir', type=str, default='./', help='Directory to save result in')

flags = parser.parse_args()
dataDir = flags.dataDir
fakeIdx = flags.FDS
saveDir = flags.saveDir

def ibu(data, r, init, it=10):
    
    # initialize the truth distribution to the prior
    phis = [init]
    
    # iterate the procedure
    for i in range(it):
        
        # update the estimate for the matrix m
        m = r * phis[-1]
        m /= (m.sum(axis=1)[:,np.newaxis] + 10**-50)

        # update the estimate for the truth distribution
        phis.append(np.dot(m.T, data))
        
    return phis

data = np.load(dataDir+'mc_vals_reco_Multifold0_Nominal.npy')
data_weight = np.load(dataDir+'mc_weights_reco_Nominal.npy')
data_truth_weight = np.load(dataDir+'mc_weights_truth_Nominal.npy')
data_truth = np.load(dataDir+'mc_vals_truth_ReactionCodesIncluded.npy')
data_pass_truth = np.load(dataDir+'mc_pass_truth_Nominal.npy')

if fakeIdx >= 0:
    data_weight = np.load(dataDir+'mc_weights_reco_FakeDataStudy%d.npy' % fakeIdx)
    data_truth_weight = np.load(dataDir+'mc_weights_truth_FakeDataStudy%d.npy' % fakeIdx)

mc_reco = np.load(dataDir+'mc_vals_reco_Multifold0_Nominal.npy')
mc_truth = np.load(dataDir+'mc_vals_truth_Multifold0.npy')
mc_truth_Reaction = np.load(dataDir+'mc_vals_truth_ReactionCodesIncluded.npy')
pass_truth = np.load(dataDir+'mc_pass_truth_Nominal.npy' )

# v10 data normalization factors
means = [6.59189684, 0, 0, 6.47929684, 0, 0, 0]
stds = [0.83999279, 1, 1, 0.39165757, 1, 1, 1]

# Ordered as (low cos-theta, high cos-theta), (low p, high p)
analysisBins = []
with open("/global/cfs/projectdirs/m4045/users/rhuang/T2K/onoffaxisdata/analysis_binning.txt", 'r') as f:
    lines = f.readlines()
    for row in lines:
        analysisBins.append([float(val) for val in row.split()])

# Rescale data back into "normal" values
dataScaled = np.copy(data[:,0:2])
mcScaled = np.copy(mc_reco[:,0:2])
truthScaled = np.copy(mc_truth_Reaction[:,0:2])

dataScaled[:,0] = [np.exp(val*stds[0] + means[0]) if val!=0 else -10000  for val in data[:,0]]
mcScaled[:,0] = [np.exp(val*stds[0] + means[0]) if val!=0 else -10000  for val in mc_reco[:,0]]

# Results will be 3-dimensional: (throw, IBU iteration number, bin number)
unfoldedResults = []
truthBinning = np.load("/global/cfs/projectdirs/m4045/users/rhuang/T2K/onoffaxisdata/TruthBinning.npy")
obs = [0 for val in analysisBins]
for idx,val in enumerate(dataScaled):
    for binIdx, bins in enumerate(analysisBins):
        if val[0] >= bins[2] and val[0] <= bins[3] and val[1] >= bins[0] and val[1] <= bins[1]:
            obs[binIdx] += data_weight[idx]
            break

for throw in range(100):
    print("Throw %d" % throw)

#    mc_reco_weight = np.load(dataDir+'mc_weights_reco_Throw%d.npy' % throw)
#    mc_truth_weight = np.load(dataDir+'mc_weights_truth_Throw%d.npy' % throw)
    mc_reco_weight = np.load('/global/cfs/projectdirs/m4045/users/rhuang/T2K/onoffaxisdata/test_recov8_truthv8/mc_weights_reco_Throw%d.npy' % throw)        
    mc_truth_weight = np.load('/global/cfs/projectdirs/m4045/users/rhuang/T2K/onoffaxisdata/test_recov8_truthv8/mc_weights_truth_Throw%d.npy' % throw)
                       
    mcRecoSignal = [0 for val in analysisBins]
    mcRecoBkg = [0 for val in analysisBins]
    recoIsSignal = np.array([False for val in mcScaled[:,0]])
    signal_mask = [tup for tup in zip(pass_truth, mc_truth_Reaction[:,8] < 3)]
    recoIdx = 0
    prior = [0 for val in analysisBins]
    truthAllSignal = [0 for val in analysisBins]
    mcRecoBinAssignments = []
    priorBinAssignments = []
    for idx,mask in enumerate(signal_mask):
        if mask[0]:
            val = mcScaled[recoIdx]
            for binIdx, bins in enumerate(analysisBins):
                if val[0] >= bins[2] and val[0] <= bins[3] and val[1] >= bins[0] and val[1] <= bins[1]:
                    if mask[1]:
                        mcRecoSignal[binIdx] += mc_reco_weight[recoIdx]
                        recoIsSignal[recoIdx] = True
                        mcRecoBinAssignments.append(binIdx)
                    else:
                        mcRecoBkg[binIdx] += mc_reco_weight[recoIdx]
                    break
            recoIdx += 1
        if mask[1]:
            binIdx = truthBinning[0,idx]
            weight = mc_truth_weight[idx]    
            if mask[0]:
                prior[binIdx] += weight
                priorBinAssignments.append(binIdx)
            truthAllSignal[binIdx] += weight
                
    responseMatrix = np.zeros((len(analysisBins), len(analysisBins)))
        
    responseWeights = mc_truth_weight[[all(val) for val in signal_mask]]
    signalRecoWeights = mc_reco_weight[recoIsSignal]
    for idx, val in enumerate(zip(mcRecoBinAssignments, priorBinAssignments)):
        responseMatrix[val[0]][val[1]] += signalRecoWeights[idx]
        prior[val[1]] += (signalRecoWeights[idx] - responseWeights[idx])
    responseMatrix /= (responseMatrix.sum(axis=0) + 10**-50)

    signalEfficiency = [val[0] / val[1] for val in zip(prior, truthAllSignal)]

    unfolded = ibu(np.clip(np.array(obs) - np.array(mcRecoBkg), 0, 1e9), responseMatrix, prior)
#    unfolded = ibu(np.clip(np.array(obs), 0, 1e9), responseMatrix, prior)
    unfoldedResults.append([np.array(val) / np.array(signalEfficiency) for val in unfolded])

if fakeIdx >=0:
    np.save("%s/IBU_2DMuon_FakeData%d_Syst_v8.npy" % (saveDir,fakeIdx), unfoldedResults)
else:
    np.save("%s/IBU_2DMuon_Nominal_Syst.npy" % (saveDir), unfoldedResults)
