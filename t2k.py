import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import horovod.tensorflow.keras as hvd
#import tensorflow.keras as hvd
import tensorflow as tf
import utils
from omnifold import  Multifold,LoadJson
import tensorflow.keras.backend as K

utils.SetStyle()

hvd.init()
# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


parser = argparse.ArgumentParser()

parser.add_argument('--config', default='config_omnifold.json', help='Basic config file containing general options')
parser.add_argument('--plot_folder', default='../plots/', help='Folder used to store plots')
parser.add_argument('--weights_folder', default='../weights/', help='Folder used to store weights')
parser.add_argument('--file_path', default='/global/cfs/projectdirs/m4045/users/rhuang/T2K/FormattedData_AllThrowsFakeEventIDs/', help='Folder containing input files')
parser.add_argument('--nevts', type=float,default=-1, help='Dataset size to use during training')
parser.add_argument('--verbose', action='store_true', default=False,help='Run the scripts with more verbose output')
parser.add_argument('--require_proton', action='store_true', default=False,help='Only use reconstructed events where 0th index value is != 0 (equivalent to requiring a reconstructed proton for Unifold3/4/5)')
parser.add_argument('--shape_only', action='store_true', default=False,help='Normalize the data/mc reco distributions to be equivalent, so that unfolding is done on shape only')
parser.add_argument('--signal_only', action='store_true', default=False,help='Only use signal events in both reco and truth')
parser.add_argument('--cheat', action='store_true', default=False,help='Cheat network')
parser.add_argument('--start_trial', type=int,default=0, help='Starting trial number')
parser.add_argument('--split', action='store_true', default=False,help='Cheat network')

flags = parser.parse_args()
nevts=int(flags.nevts)
opt = LoadJson(flags.config)

if not os.path.exists(flags.plot_folder):
    os.makedirs(flags.plot_folder)


data, mc_reco,mc_gen,reco_mask,gen_mask,data_weights,mc_weights,mc_weights_reco = utils.DataLoader(flags.file_path,opt,nevts)

if flags.shape_only:
    mc_weights_reco *= np.sum(data_weights) / np.sum(mc_weights_reco)

if flags.signal_only:
    signal_mask = np.load(flags.file_path + "mc_vals_truth_ReactionCodesIncluded.npy")[:,8] < 3
    signal_mask_reco = signal_mask[gen_mask]
    gen_mask = gen_mask[signal_mask]
    mc_weights = mc_weights[signal_mask]
    mc_gen = mc_gen[signal_mask, :]
    
    reco_mask = reco_mask[signal_mask_reco]
    data_weights = data_weights[signal_mask_reco]
    mc_weights_reco = mc_weights_reco[signal_mask_reco]
    data = data[signal_mask_reco, :]
    mc_reco = mc_reco[signal_mask_reco, :]

if flags.split:
    mc_gen = mc_gen[1::2,:]
    mc_weights = mc_weights[1::2]
    recoSelect = np.array([False for i in reco_mask])
    data_mask = np.copy(recoSelect)
    recoIdx = 0
    for idx, val in enumerate(gen_mask):
        if not val:
            continue
        if idx % 2 == 1:
            recoSelect[recoIdx] = True
        else:
            data_mask[recoIdx] = True
        recoIdx += 1
    gen_mask = gen_mask[1::2]
    data = data[data_mask,:]
    data_weights = data_weights[data_mask]
    reco_mask = reco_mask[recoSelect]
    mc_weights_reco = mc_weights_reco[recoSelect]
    mc_reco = mc_reco[recoSelect, :]

reqProtonIdx = 0
if flags.require_proton:
    data_weights = data_weights[data[:,reqProtonIdx]!=0]
    data = data[data[:,reqProtonIdx]!=0,:]
    recoIdx = 0
    for idx, val in enumerate(gen_mask):
        if val:
            gen_mask[idx] = mc_reco[recoIdx,reqProtonIdx] != 0
            recoIdx += 1
    mc_weights_reco = mc_weights_reco[mc_reco[:,reqProtonIdx] != 0]
    mc_reco = mc_reco[mc_reco[:,reqProtonIdx] != 0, :]


for itrial in range(flags.start_trial, flags.start_trial+opt['NTRIAL']):
    K.clear_session()
    mfold = Multifold(version='{}_trial{}'.format(opt['NAME'],itrial),verbose=flags.verbose,config_file=flags.config,plot_folder=flags.plot_folder,weights_folder=flags.weights_folder)
    mfold.mc_gen = mc_gen
    mfold.mc_reco =mc_reco
    mfold.data = data

    thisSeed = itrial*100+123456789
    tf.random.set_seed(thisSeed)
    mfold.Preprocessing(weights_mc_reco=mc_weights_reco,weights_mc=mc_weights,weights_data=data_weights,pass_reco=reco_mask,pass_gen=gen_mask)
    mfold.Unfold(cheat=flags.cheat)
