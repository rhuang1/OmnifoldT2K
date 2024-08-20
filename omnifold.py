import numpy as np
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
import sys, os
import horovod.tensorflow.keras as hvd
import json, yaml
import utils
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss    
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    t_loss = -weights * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * K.log(1 - y_pred))
    return K.mean(t_loss) 


def LoadJson(file_name):
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))


class Multifold():
    def __init__(self,version,config_file='config_omnifold.json',verbose=False,plot_folder="../plots/",weights_folder="../weights/"):
        self.opt = LoadJson(config_file)
        self.plot_folder = plot_folder
        self.niter = self.opt['NITER']
        self.version=version
        self.mc_gen = None
        self.mc_reco = None
        self.data=None
        self.verbose=verbose
                
        self.weights_folder = weights_folder
        if not os.path.exists(self.weights_folder):
            os.makedirs(self.weights_folder)
            
    def Unfold(self, cheat=True):
        self.BATCH_SIZE=self.opt['BATCH_SIZE']
        self.EPOCHS=self.opt['EPOCHS']

        self.weights_pull = np.ones(self.weights_mc.shape[0])
        self.weights_push = np.ones(self.weights_mc.shape[0])

        for i in range(self.niter):
            print("ITERATION: {}".format(i + 1))
            self.CompileModel(float(self.opt['LR']),self.model1)
            self.RunStep1(i, cheat)
            self.CompileModel(float(self.opt['LR']),self.model2)
            self.RunStep2(i)
                
    def RunStep1(self,i,cheat):
        '''Data versus reco MC reweighting'''
        print("RUNNING STEP 1")
        loss = 0.5*(np.mean(self.weights_data * np.log(0.5)) + np.mean(self.weights_push[self.pass_gen]*self.weights_mc_reco*np.log(0.5)))
        print("Loss before step 1 training: %.4f" % (-1*loss))
        if cheat:
            new_weights = self.weights_data / (self.weights_push[self.pass_gen]*self.weights_mc_reco)
            new_weights = np.nan_to_num(new_weights, nan=10)
            new_weights = np.clip(new_weights, 1e-3, 10)
            newWeightIdx = 0
            for idx in range(self.weights_push.shape[0]):
                if self.pass_gen[idx]:
                    self.weights_pull[idx] = self.weights_push[idx] * new_weights[newWeightIdx]
                    newWeightIdx = newWeightIdx + 1
                else:
                    self.weights_pull[idx] = self.weights_push[idx]
            return
        
        self.RunModel(
            np.concatenate([self.mc_reco, self.data]),
            np.concatenate([self.labels_mc, self.labels_data]),
            np.concatenate([self.weights_push[self.pass_gen]*self.weights_mc_reco,self.weights_data]),
            i,self.model1,stepn=1,
        )

        new_weights=self.reweight(self.mc_reco,self.model1)
        newWeightIdx = 0
        meanPull = np.mean(new_weights)
        print("Mean pull: %.2f" % meanPull)
        for idx in range(self.weights_push.shape[0]):
            if self.pass_gen[idx]:
                self.weights_pull[idx] = self.weights_push[idx] * new_weights[newWeightIdx]
                newWeightIdx = newWeightIdx + 1
            else:
                self.weights_pull[idx] = self.weights_push[idx]
        print("Total pulled weights: %d" % newWeightIdx) 
        if self.verbose:            
            print("Plotting the results after step 1")
            print(self.plot_folder)
#            titles = ['Muon Momentum (Normalized)', r'Muon cos $\theta$ (Normalized)', r'Muon $\phi$ (Normalized)']
            titles = ['Muon $P_{x}$ (Normalized)', 'Muon $P_{y}$ (Normalized)', 'Muon $P_{z}$ (Normalized)',
                      'Proton $P_{x}$ (Normalized)', 'Proton $P_{y}$ (Normalized)', 'Proton $P_{z}$ (Normalized)', 'Sample']
            for varIdx in range(7):
                weight_dict = {
                    'mc reco':self.weights_mc_reco*self.weights_push[self.pass_gen],
                    #                'mc reco':self.weights_pull[self.pass_gen]*self.weights_mc[self.pass_gen],
                    'data reco': self.weights_data,
                }

                feed_dict = {
                    'data reco':self.data[:,varIdx],
                    'mc reco':self.mc_reco[:,varIdx],
                }

                fig,ax = utils.HistRoutine(feed_dict,plot_ratio=True,
                                           weights = weight_dict,
                                           binning=utils.binning,
                                           xlabel=titles[varIdx],logy=True,
                                           ylabel='Events',
                                           reference_name='data reco')
                fig.savefig(os.path.join(self.plot_folder, 'Unfolded_Hist_T_step1_{}_{}_Var{}_NoPull.pdf'.format(i,self.opt['NAME'],varIdx)))
                weight_dict = {
                    'mc reco':self.weights_push[self.pass_gen]*self.weights_mc_reco,
                    'mc reco reweighted':self.weights_pull[self.pass_gen]*self.weights_mc_reco,
                    #                'mc reco':self.weights_pull[self.pass_gen]*self.weights_mc[self.pass_gen],
                    'data reco': self.weights_data,
                }
                
                feed_dict = {
                    'data reco':self.data[:,varIdx],
                    'mc reco':self.mc_reco[:,varIdx],
                    'mc reco reweighted': self.mc_reco[:,varIdx]
                }
                
                fig,ax = utils.HistRoutine(feed_dict,plot_ratio=True,
                                           weights = weight_dict,
                                           binning=utils.binning,
                                           xlabel=titles[varIdx],logy=True,
                                           ylabel='Events',
                                           reference_name='data reco')
                fig.savefig(os.path.join(self.plot_folder, 'Unfolded_Hist_T_step1_{}_{}_Var{}.pdf'.format(i,self.opt['NAME'],varIdx)))
                #plt.clf()
                #plt.hist(self.data[:,0],weights=self.weights_data,bins=np.linspace(-1,1,40))
                #plt.hist(self.mc_reco[:,0],weights=self.weights_pull[self.pass_gen]*self.weights_mc_reco,bins=np.linspace(-1,1,40),alpha=0.4)
                #plt.savefig('../plots/Iteration{}.pdf'.format(i))
                #plt.clf()
        

    def RunStep2(self,i):
        '''Gen to Gen reweighing'''        
        print("RUNNING STEP 2")
        loss = 0.5*(np.mean(self.weights_mc * np.log(0.5)) + np.mean(self.weights_mc*self.weights_pull*np.log(0.5)))
        print("Loss before step 1 training: %.4f" % (-1*loss))
        print(np.mean(self.weights_pull))

        self.RunModel(
            np.concatenate((self.mc_gen, self.mc_gen)),
            np.concatenate((np.full_like(self.labels_gen, 0), self.labels_gen)),
            np.concatenate((self.weights_mc, self.weights_mc*self.weights_pull)),
            i,self.model2,stepn=2,
        )


        new_weights=self.reweight(self.mc_gen,self.model2)
        self.weights_push = new_weights


    def RunModel(self,sample,labels,weights,iteration,model,stepn):
        
        mask = np.full_like(sample[:,0], 1)
        NTRAIN,NTEST = self.GetNtrainNtest(np.sum(mask))

        verbose = 1 if hvd.rank() == 0 else 0
        
        callbacks = [
#            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
#            hvd.callbacks.MetricAverageCallback(),
#            hvd.callbacks.LearningRateWarmupCallback(
#                 initial_lr=self.hvd_lr, warmup_epochs=self.opt['NWARMUP'],
#                 verbose=verbose),
#            ReduceLROnPlateau(patience=8, min_lr=1e-7,verbose=verbose),
            EarlyStopping(patience=self.opt['NPATIENCE'],restore_best_weights=True)
        ]
        
        
        if hvd.rank() ==0:
            callbacks.append(
                ModelCheckpoint('{}/{}_iter{}_step{}.h5'.format(
                    self.weights_folder,self.version,iteration,stepn),
                                save_best_only=True,mode='auto',period=1,save_weights_only=False))

        X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = train_test_split(sample, labels, weights,test_size=0.25)
        history = model.fit(X_train_1,np.stack((Y_train_1, w_train_1), axis=1),
                            epochs=self.EPOCHS,
                            steps_per_epoch=int(X_train_1.shape[0]/self.BATCH_SIZE),
                            validation_data=(X_test_1,np.stack((Y_test_1, w_test_1), axis=1)),
                            validation_steps=int(X_test_1.shape[0]/self.BATCH_SIZE),
                            callbacks=callbacks, batch_size=self.BATCH_SIZE,
                            verbose=verbose)
        json.dump(history.history, open('{}/{}_iter{}_step{}_History.json'.format(self.weights_folder,self.version,iteration,stepn), 'w'))




    def Preprocessing(self,weights_mc_reco=None,weights_mc=None,weights_data=None,pass_reco=None,pass_gen=None):
#        self.PrepareWeights(weights_mc,weights_data,pass_reco,pass_gen)
        self.PrepareWeights(weights_mc_reco,weights_mc,weights_data,pass_reco,pass_gen)
        self.PrepareInputs()
        self.PrepareModel(nvars1 = self.mc_reco.shape[1], nvars2 = self.mc_gen.shape[1])

    def PrepareWeights(self,weights_mc_reco,weights_mc,weights_data,pass_reco,pass_gen):
        self.pass_gen = np.copy(pass_gen).astype('bool')
        self.pass_reco = np.copy(pass_reco).astype('bool')
        if pass_reco is None:
            if self.verbose: print("No reco mask provided, making one based on inputs")
            self.not_pass_reco = self.mc_reco[:,0]==-10
        else:
            self.not_pass_reco = pass_reco==0
            #self.mc_reco[self.not_pass_reco]=-10
            
        if pass_gen is None:
            if self.verbose: print("No gen mask provided, making one based on inputs")
            self.not_pass_gen = self.mc_gen[:,0]==-10
        else:
            self.not_pass_gen = pass_gen==0
            #self.mc_gen[self.not_pass_gen]=-10

        if weights_mc_reco is None:
            if self.verbose: print("No MC reco weights provided, making one filled with 1s")
            self.weights_mc_reco = np.ones(self.mc_reco.shape[0])
        else:
            self.weights_mc_reco = np.copy(weights_mc_reco)
        
        if weights_mc is None:
            if self.verbose: print("No MC weights provided, making one filled with 1s")
            self.weights_mc = np.ones(self.mc_reco.shape[0])
        else:
            self.weights_mc = np.copy(weights_mc)

        if weights_data is None:
            if self.verbose: print("No data weights provided, making one filled with 1s")
            self.weights_data = np.ones(self.data.shape[0])
        else:
            self.weights_data = np.copy(weights_data)
            
    def CompileModel(self,lr,model):
#        self.hvd_lr = lr*np.sqrt(hvd.size())
        self.hvd_lr = lr
#        opt = tensorflow.keras.optimizers.Adadelta(learning_rate=self.hvd_lr)
        opt = tensorflow.keras.optimizers.Adam(learning_rate=float(self.opt['LR']))
        opt = hvd.DistributedOptimizer(opt)

        model.compile(loss=weighted_binary_crossentropy,
                      optimizer=opt,metrics=['accuracy'],experimental_run_tf_function=False)


    def PrepareInputs(self):
        self.labels_mc = np.zeros(len(self.mc_reco))
        self.labels_data = np.ones(len(self.data))
        self.labels_gen = np.ones(len(self.mc_gen))

    def PrepareModel(self,nvars1,nvars2):
        inputs1,outputs1 = MLP(nvars1)
        inputs2,outputs2 = MLP(nvars2)
                                   
        self.model1 = Model(inputs=inputs1, outputs=outputs1)
        self.model2 = Model(inputs=inputs2, outputs=outputs2)


    def GetNtrainNtest(self,nevts):
        NTRAIN=int(0.8*nevts)
        NTEST=int(0.2*nevts)                        
        return NTRAIN,NTEST

    def reweight(self,events,model):
        f = np.nan_to_num(model.predict(events, batch_size=10000)[:,:1],posinf=1,neginf=0)
        weights = f / (1. - f)
        weights = weights[:,0]
        return np.squeeze(np.nan_to_num(weights,posinf=1))

    def LoadModel(self,iteration):
        model_name = '{}/{}_iter{}_step2.h5'.format(
            self.weights_folder,self.version,iteration)
        self.model2.load_weights(model_name)



def MLP(nvars):
    ''' Define a simple fully conneted model to be used during unfolding'''
    inputs = Input((nvars, ))
    layer = Dense(200,activation='selu')(inputs)
    layer = Dense(200,activation='selu')(layer)
    layer = Dense(200,activation='selu')(layer)
    outputs = Dense(1,activation='sigmoid')(layer)
    return inputs,outputs

