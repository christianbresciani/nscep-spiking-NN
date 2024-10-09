from matplotlib import pyplot as plt
import torch
import os
import gc
import pickle
import numpy as np
import random

from metrics import bayesianHypothesisTesting, bayesian_hypothesis_testing

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
TIME_WINDOW = 1.5
CSI_PER_SECOND = 30
WINDOW_SIZE = int(TIME_WINDOW * CSI_PER_SECOND)

TEST_ON_DIFFERENT_SETS = False
if TEST_ON_DIFFERENT_SETS:
    TRAIN_STARTS = range(0, 80*CSI_PER_SECOND-int(TIME_WINDOW*CSI_PER_SECOND), int(TIME_WINDOW*CSI_PER_SECOND/15))
    TEST_STARTS = range(0, 80*CSI_PER_SECOND-int(TIME_WINDOW*CSI_PER_SECOND), int(CSI_PER_SECOND/30))
else:
    TRAIN_STARTS = range(0, 60*CSI_PER_SECOND-int(TIME_WINDOW*CSI_PER_SECOND), int(TIME_WINDOW*CSI_PER_SECOND/15))
    TEST_STARTS = range(60*CSI_PER_SECOND, 80*CSI_PER_SECOND-int(TIME_WINDOW*CSI_PER_SECOND), int(TIME_WINDOW*CSI_PER_SECOND/3))

ACTIONS = ['A', 'K'] 
LABELS = ['walk', 'squat']

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

        # divide the 80s of data in 2310 windows of 3s
        train = []
        for start in TRAIN_STARTS:
            train.append(data[start:start+WINDOW_SIZE, ...])
        train = torch.Tensor(np.array(train))

        test = []
        for start in TEST_STARTS:
            test.append(data[start:start+WINDOW_SIZE, ...])
        test = torch.Tensor(np.array(test))
        
        # Split the data into training and test sets [80%, 20%]
        train_csi = torch.cat([train_csi, train], dim=0)
        train_labels += [label] * len(train) 

        test_csi = torch.cat([test_csi, test], dim=0)
        test_labels += [label] * len(test)

        del train, test, data
        gc.collect()

    # Normalize the CSI dataset
    max_val = max([torch.max(train_csi), torch.max(test_csi)])
    train_csi = torch.true_divide(train_csi, max_val)
    test_csi = torch.true_divide(test_csi, max_val)

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
    preds, labels, probabilities = predict(model, test, LABELS)

    # Compute the confusion matrix
    cm = np.zeros((len(LABELS)+1, len(LABELS)+1), dtype=int)
    assert len(preds) == len(labels)
    for true, pred in zip(labels, preds):
        cm[true, pred] += 1

    # Save the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ACTIONS+['None'])
    cmdisp = disp.plot(cmap="cividis")
    cmdisp.figure_.savefig(f"SNN/results/neuroSymbolic/ConfMat{name}.png", bbox_inches='tight')

    return cm, probabilities


def main():
    train, test, loader = prepare_data()


    lr = 0.0007
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    nnul1 = CNNetwork(2, 1)
    netul1 = Network(nnul1, "upper_legs_net_1", batching=True)
    netul1.cuda(device)
    netul1.optimizer = torch.optim.AdamW(nnul1.parameters(), lr=lr)

    nnul2 = CNNetwork(2, 2)
    netul2 = Network(nnul2, "upper_legs_net_2", batching=True)
    netul2.cuda(device)
    netul2.optimizer = torch.optim.AdamW(nnul2.parameters(), lr=lr)

    nnul3 = CNNetwork(2, 3)
    netul3 = Network(nnul3, "upper_legs_net_3", batching=True)
    netul3.cuda(device)
    netul3.optimizer = torch.optim.AdamW(nnul3.parameters(), lr=lr)

    nnua1 = CNNetwork(2, 1)
    netua1 = Network(nnua1, "upper_arms_net_1", batching=True)
    netua1.cuda(device)
    netua1.optimizer = torch.optim.AdamW(nnua1.parameters(), lr=lr)

    nnua2 = CNNetwork(2, 2)
    netua2 = Network(nnua2, "upper_arms_net_2", batching=True)
    netua2.cuda(device)
    netua2.optimizer = torch.optim.AdamW(nnua2.parameters(), lr=lr)

    nnua3 = CNNetwork(2, 3)
    netua3 = Network(nnua3, "upper_arms_net_3", batching=True)
    netua3.cuda(device)
    netua3.optimizer = torch.optim.AdamW(nnua3.parameters(), lr=lr)

    modelCnn = Model("SNN/temporalLogic.pl", [netul1, netul2, netul3, netua1, netua2, netua3])
    modelCnn.add_tensor_source("train", train)
    modelCnn.set_engine(ExactEngine(modelCnn), cache=True)

    trained_model = train_model(
        modelCnn,
        loader,
        StopOnPlateau("Accuracy", warm_up=5, patience=5) | Threshold("Accuracy", 1.0, duration=2) | EpochStop(30),
        log_iter=100,
        profile=0,
    )


    del trained_model, nnua1, nnua2, nnua3, nnul1, nnul2, nnul3, netua1, netua2, netua3, netul1, netul2, netul3
    gc.collect()

    lr = 0.001

    nnul1 = SNNetwork(100, 2, time_frame=1)
    netul1 = Network(nnul1, "upper_legs_net_1", batching=True)
    netul1.optimizer = torch.optim.AdamW(nnul1.parameters(), lr=lr)
    netul1.cuda(device)

    nnul2 = SNNetwork(100, 2, time_frame=2)
    netul2 = Network(nnul2, "upper_legs_net_2", batching=True)
    netul2.optimizer = torch.optim.AdamW(nnul2.parameters(), lr=lr)
    netul2.cuda(device)

    nnul3 = SNNetwork(100, 2, time_frame=3)
    netul3 = Network(nnul3, "upper_legs_net_3", batching=True)
    netul3.optimizer = torch.optim.AdamW(nnul3.parameters(), lr=lr)
    netul3.cuda(device)

    nnua1 = SNNetwork(100, 2, time_frame=1)
    netua1 = Network(nnua1, "upper_arms_net_1", batching=True)
    netua1.optimizer = torch.optim.AdamW(nnua1.parameters(), lr=lr)
    netua1.cuda(device)

    nnua2 = SNNetwork(100, 2, time_frame=2)
    netua2 = Network(nnua2, "upper_arms_net_2", batching=True)
    netua2.optimizer = torch.optim.AdamW(nnua2.parameters(), lr=lr)
    netua2.cuda(device)

    nnua3 = SNNetwork(100, 2, time_frame=3)
    netua3 = Network(nnua3, "upper_arms_net_3", batching=True)
    netua3.optimizer = torch.optim.AdamW(nnua3.parameters(), lr=lr)
    netua3.cuda(device)

    modelSnn = Model("SNN/temporalLogic.pl", [netul1, netul2, netul3, netua1, netua2, netua3])
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

            test_net(modelSnn, test_set, f'SNNtemp{name}1s')
            test_net(modelCnn, test_set, f'CNNtemp{name}1s')
            print(f"Tested on {name} dataset")
            del test_set
            gc.collect()
    else:
        cmCNN, probabilitiesCNN = test_net(modelCnn, test, 'CNNtemp')
        cmSNN, probabilitiesSNN = test_net(modelSnn, test, 'SNNtemp')

        for name in probabilitiesCNN.keys():
            plt.figure()
            x = np.arange(1.5, 20, 0.5) 
            probabilities = {'SNN': probabilitiesSNN[name], 'CNN': probabilitiesCNN[name]}
            for label, y in probabilities.items():
                plt.plot(x, y, label=label)

            plt.xlabel('Time (s)')
            plt.ylabel('Probability')
            plt.title('Probabilities over time')
            plt.ylim(0, 1)
            plt.xlim(0, 20)
            plt.legend()
            plt.savefig(f"SNN/results/neuroSymbolic/Probabilities{name}.png", bbox_inches='tight')
        bayesian_hypothesis_testing(cmSNN, cmCNN)


if __name__ == "__main__":
    main()