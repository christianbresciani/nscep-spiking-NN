from matplotlib import pyplot as plt
import torch
import os
import gc
import pickle
import numpy as np
import random
from metrics import bayesianHypothesisTesting
from deepprobhar import main as deepprobhar


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from nsNetworks import SNNetwork, CNNetwork, predict

from deepproblog.dataset import Dataset, DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.utils.stop_condition import Threshold, StopOnPlateau, EpochStop

from problog.logic import Term, Constant
from deepproblog.query import Query
import time


import cProfile

# Set the random seed for reproducibility
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Set constants
BATCH_SIZE = 35
TIME_WINDOW = 1
CSI_PER_SECOND = 30
WINDOW_SIZE = int(TIME_WINDOW * CSI_PER_SECOND)
ACTIONS = ['A', 'B', 'C', 'G', 'H', 'J', 'K'] 
LABELS = ['walk', 'run', 'jump', 'wave', 'clap', 'wipe', 'squat']

TEST_ON_DIFFERENT_SETS = False
if TEST_ON_DIFFERENT_SETS:
    TRAIN_STARTS = range(0, 80*CSI_PER_SECOND-int(TIME_WINDOW*CSI_PER_SECOND), int(TIME_WINDOW*CSI_PER_SECOND/15))
    TEST_STARTS = range(0, 80*CSI_PER_SECOND-int(TIME_WINDOW*CSI_PER_SECOND), int(CSI_PER_SECOND/15))
else:
    TRAIN_STARTS = range(0, 60*CSI_PER_SECOND-int(TIME_WINDOW*CSI_PER_SECOND), int(TIME_WINDOW*CSI_PER_SECOND/15)) # len(TRAIN_STARTS) = 885
    TEST_STARTS = range(60*CSI_PER_SECOND, 80*CSI_PER_SECOND-int(TIME_WINDOW*CSI_PER_SECOND), int(TIME_WINDOW*CSI_PER_SECOND/10)) # len(TEST_STARTS) = 190

# Define the CSI dataset class
class CsiDataset(Dataset):
    def __init__(self, data, subset):

        self.data = data
        self.subset = subset

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx[0].value
        return self.data[0][idx]
    
    def to_query(self, idx):
        sub = {Term("a"): Term("tensor", Term(self.subset, Constant(idx)))}
        return Query(Term("activity", Term("a"), Term(self.data[1][idx])), sub)
    
# A function to load the data from different files and create the Datasets and DataLoaders
def prepare_data():
    train_csi = torch.Tensor()
    train_labels = []
    test_csi = torch.Tensor()
    test_labels = []

    for x, label in zip(ACTIONS, LABELS):
        if os.getcwd().split('/')[-1] == 'SNN':
            os.chdir('..')

        file = f'datasets/mean_dataset_abs/S1a_{x}.pkl'

        # Load CSI data from pickle file
        with open(file, 'rb') as f:
            data = pickle.load(f)       # WARNING This code does not handle exceptions for simplicity exceptions would require keeping track of indices

        # divide the 80s of data in 2310+770 windows of 1s
        train = []
        for start in TRAIN_STARTS:
            train.append(data[start:start+WINDOW_SIZE, ...])
        train = torch.Tensor(np.array(train))

        test = []
        for start in TEST_STARTS:
            test.append(data[start:start+WINDOW_SIZE, ...])
        test = torch.Tensor(np.array(test))
        
        train_csi = torch.cat([train_csi, train], dim=0)
        train_labels += [label] * len(train) 

        test_csi = torch.cat([test_csi, test], dim=0)
        test_labels += [label] * len(test)

        del train, test, data
        gc.collect()

    # Normalize the CSI dataset
    max_val = max([torch.max(train_csi), torch.max(test_csi)])
    train_csi = torch.true_divide(train_csi, max_val)
    gc.collect()
    test_csi = torch.true_divide(test_csi, max_val)
    gc.collect()

    indices = torch.randperm(train_csi.size(0))
    train_csi = train_csi[indices]
    train_labels = [train_labels[i] for i in indices]

    # create Datasets and DataLoaders
    csi_trainset = CsiDataset([train_csi, train_labels], "train")
    csi_testset = CsiDataset([test_csi, test_labels], "test")

    csi_dataloader = DataLoader(csi_trainset, batch_size=BATCH_SIZE)

    return csi_trainset, csi_testset, csi_dataloader


def load_test_data(set=2):
    test_csi = torch.Tensor()
    test_labels = []

    for x, label in zip(ACTIONS, LABELS):
        if os.getcwd().split('/')[-1] == 'SNN':
            os.chdir('..')

        filetest = f'datasets/mean_dataset_abs/S{set}a_{x}.pkl'
        # Load CSI data from pickle file
        with open(filetest, 'rb') as f:
            datatest = pickle.load(f)       # WARNING This code does not handle exceptions for simplicity exceptions would require keeping track of indices

        # divide the 80s of data in 2310 windows of 3s
        csidata = []
        for start in TEST_STARTS:
            csidata.append(datatest[start:start+WINDOW_SIZE, ...])

        csidata = torch.Tensor(np.array(csidata))

        test_csi = torch.cat([test_csi, csidata], dim=0)
        test_labels = test_labels + [label] * len(csidata)

    # Normalize the CSI dataset
    max_val = torch.max(test_csi)
    test_csi = torch.true_divide(test_csi, max_val)

    csi_testset = CsiDataset([test_csi, test_labels], "test")

    return csi_testset


def test_net(model, test, name):
    model.add_tensor_source("test", test)

    # Compute the predictions
    preds, labels, _ = predict(model, test, LABELS)

    # Compute the confusion matrix
    cm = confusion_matrix(labels, preds)

    return cm

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = 0.001

    nn1 = SNNetwork(100, 3, num_steps=10)
    net1 = Network(nn1, "upper_legs_net", batching=True)
    net1.optimizer = torch.optim.AdamW(nn1.parameters(), lr=lr)
    # net1.cuda(device)

    nn2 = SNNetwork(100, 2, num_steps=10)
    net2 = Network(nn2, "lower_legs_net", batching=True)
    net2.optimizer = torch.optim.AdamW(nn2.parameters(), lr=lr)
    # net2.cuda(device)

    nn3 = SNNetwork(100, 4, num_steps=10)
    net3 = Network(nn3, "forearms_net", batching=True)
    net3.optimizer = torch.optim.AdamW(nn3.parameters(), lr=lr)
    # net3.cuda(device)

    modelSnn = Model("SNN/logic.pl", [net1, net2, net3])
    modelSnn.add_tensor_source("train", train)
    modelSnn.add_tensor_source("test", test)
    modelSnn.set_engine(ExactEngine(modelSnn), cache=True)

    trained_model = train_model(
        modelSnn,
        loader,
        StopOnPlateau("Accuracy", warm_up=5, patience=5) | Threshold("Accuracy", 1.0, duration=2) | EpochStop(30),
        log_iter=100,
        profile=0,
    )

    if TEST_ON_DIFFERENT_SETS:
        for set, name in zip([2, 6, 5, 4], ['Sub', 'Env', 'Day', 'Same']):
            # Load the test data
            test_set = load_test_data(set)

            test_net(modelSnn, test_set, f'SNN{name}1s')
            print(f"Tested on {name} dataset")
            del test_set
            gc.collect()
    else:
        return test_net(modelSnn, test, 'SNN1s')

if __name__ == "__main__":
    
    train, test, loader = prepare_data()
    cmSNN = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    cmVAE = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    
    for _ in range(3):
        snn = main()
        vae = deepprobhar()
        cmSNN += snn
        cmVAE += vae
    

    # Save the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cmSNN, display_labels=LABELS)
    cmdisp = disp.plot(cmap="cividis")
    plt.setp(cmdisp.ax_.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(cmdisp.ax_.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    cmdisp.figure_.set_size_inches(12, 12)
    cmdisp.figure_.savefig(f"SNN/results/neuroSymbolic/ConfMatSNN.png", bbox_inches='tight')

        # Save the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cmVAE, display_labels=LABELS)
    cmdisp = disp.plot(cmap="cividis")
    plt.setp(cmdisp.ax_.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(cmdisp.ax_.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    cmdisp.figure_.set_size_inches(12, 12)
    cmdisp.figure_.savefig(f"SNN/results/neuroSymbolic/ConfMatVAE.png", bbox_inches='tight')

    print(f'Accuracy SNN: {np.trace(cmSNN)/np.sum(cmSNN)}')
    print(f'Accuracy CNN: {np.trace(cmVAE)/np.sum(cmVAE)}')

    bayesianHypothesisTesting(cmSNN, cmVAE)

