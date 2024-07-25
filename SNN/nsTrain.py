import torch
import os
import gc
import pickle
import numpy as np
import random

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

# Set the random seed for reproducibility
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Set constants
BATCH_SIZE = 35
TIME_WINDOW = 3
CSI_PER_SECOND = 30
WINDOW_SIZE = int(TIME_WINDOW * CSI_PER_SECOND)
STARTS = range(0, 80*CSI_PER_SECOND-int(TIME_WINDOW*CSI_PER_SECOND), int(TIME_WINDOW*CSI_PER_SECOND/30))
ACTIONS = ['A', 'B', 'C', 'G', 'H', 'J', 'K'] # 
LABELS = ['walk', 'run', 'jump', 'wave', 'clap', 'wipe', 'squat']

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
        csi = []
        for start in STARTS:
            csi.append(data[start:start+WINDOW_SIZE, ...])
        csi = torch.Tensor(np.array(csi))
        
        # Split the data into training and test sets [80%, 20%]
        sep = int(0.8 * len(csi))
        train_csi = torch.cat([train_csi, csi[:sep]], dim=0)
        train_labels += [label] * sep 

        test_csi = torch.cat([test_csi, csi[sep:]], dim=0)
        test_labels += [label] * (len(csi) - sep)

        del csi, data
        gc.collect()

    # Normalize the CSI dataset
    max_val = max([torch.max(train_csi), torch.max(test_csi)])
    train_csi = torch.true_divide(train_csi, max_val)
    gc.collect()
    test_csi = torch.true_divide(test_csi, max_val)
    gc.collect()

    indices = torch.randperm(train_csi.size(0))

    # create Datasets and DataLoaders
    csi_trainset = CsiDataset([train_csi[indices], [train_labels[i] for i in indices]], "train")
    csi_testset = CsiDataset([test_csi, test_labels], "test")

    csi_dataloader = DataLoader(csi_trainset, batch_size=BATCH_SIZE)

    return csi_trainset, csi_testset, csi_dataloader

train, test, loader = prepare_data()


lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


nn1 = CNNetwork(3)
net1 = Network(nn1, "upper_legs_net", batching=True)
net1.cuda(device)
net1.optimizer = torch.optim.Adam(nn1.parameters(), lr=lr)

nn2 = CNNetwork(2)
net2 = Network(nn2, "lower_legs_net", batching=True)
net2.cuda(device)
net2.optimizer = torch.optim.Adam(nn2.parameters(), lr=lr)

nn3 = CNNetwork(4)
net3 = Network(nn3, "forearms_net", batching=True)
net3.cuda(device)
net3.optimizer = torch.optim.Adam(nn3.parameters(), lr=lr)

model = Model("logic.pl", [net1, net2, net3])
model.add_tensor_source("train", train)
model.add_tensor_source("test", test)
model.set_engine(ExactEngine(model), cache=True)

trained_model = train_model(
    model,
    loader,
    StopOnPlateau("Accuracy", warm_up=5, patience=5) | Threshold("Accuracy", 1.0, duration=2) | EpochStop(10),
    log_iter=100,
    profile=0,
)
trained_model.logger.comment(
    "Accuracy {}".format(get_confusion_matrix(model, test, verbose=1).accuracy())
)
# save the confusion matrix
predictions, labels = predict(model, test)
cm = confusion_matrix(labels, predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ACTIONS)
cmdisp = disp.plot(cmap="cividis")
cmdisp.figure_.savefig(f"SNN/results/neuroSymbolic/ConfMatCNN.png", bbox_inches='tight')






nn1 = SNNetwork(100, 9, 3)
net1 = Network(nn1, "upper_legs_net", batching=True)
net1.optimizer = torch.optim.Adam(nn1.parameters(), lr=lr)

nn2 = SNNetwork(100, 9, 2)
net2 = Network(nn2, "lower_legs_net", batching=True)
net2.optimizer = torch.optim.Adam(nn2.parameters(), lr=lr)

nn3 = SNNetwork(100, 9, 4)
net3 = Network(nn3, "forearms_net", batching=True)
net3.optimizer = torch.optim.Adam(nn3.parameters(), lr=lr)

model = Model("logic.pl", [net1, net2, net3])
model.add_tensor_source("train", train)
model.add_tensor_source("test", test)
model.set_engine(ExactEngine(model), cache=True)

trained_model = train_model(
    model,
    loader,
    StopOnPlateau("Accuracy", warm_up=15, patience=5) | Threshold("Accuracy", 1.0, duration=2) | EpochStop(20),
    log_iter=100,
    profile=0,
)
trained_model.logger.comment(
    "Accuracy {}".format(get_confusion_matrix(model, test, verbose=1).accuracy())
)

# save the confusion matrix
predictions, labels = predict(model, test)
cm = confusion_matrix(labels, predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
cmdisp = disp.plot(cmap="cividis")
cmdisp.figure_.savefig(f"SNN/results/neuroSymbolic/ConfMatSNN.png", bbox_inches='tight')
