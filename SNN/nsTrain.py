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
STARTS = range(0, 80*CSI_PER_SECOND-int(TIME_WINDOW*CSI_PER_SECOND), int(TIME_WINDOW*CSI_PER_SECOND/30))
TEST_STARTS = range(0, 80*CSI_PER_SECOND-int(TIME_WINDOW*CSI_PER_SECOND), CSI_PER_SECOND)
ACTIONS = ['A', 'B', 'C', 'G', 'H', 'J', 'K'] 
LABELS = ['walk', 'run', 'jump', 'wave', 'clap', 'wipe', 'squat']

TEST_ON_DIFFERENT_SETS = False

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
    preds, labels = predict(model, test, LABELS)

    # Compute the confusion matrix
    cm = confusion_matrix(labels, preds)

    # Save the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ACTIONS)
    cmdisp = disp.plot(cmap="cividis")
    cmdisp.figure_.savefig(f"SNN/results/neuroSymbolic/ConfMat{name}.png", bbox_inches='tight')

def main():
    train, test, loader = prepare_data()


    lr = 0.0007
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    nn1 = CNNetwork(3)
    net1 = Network(nn1, "upper_legs_net", batching=True)
    net1.cuda(device)
    net1.optimizer = torch.optim.AdamW(nn1.parameters(), lr=lr)

    nn2 = CNNetwork(2)
    net2 = Network(nn2, "lower_legs_net", batching=True)
    net2.cuda(device)
    net2.optimizer = torch.optim.AdamW(nn2.parameters(), lr=lr)

    nn3 = CNNetwork(4)
    net3 = Network(nn3, "forearms_net", batching=True)
    net3.cuda(device)
    net3.optimizer = torch.optim.AdamW(nn3.parameters(), lr=lr)

    modelCnn = Model("SNN/logic.pl", [net1, net2, net3])
    modelCnn.add_tensor_source("train", train)
    modelCnn.set_engine(ExactEngine(modelCnn), cache=True)

    start_time = time.time()

    trained_model = train_model(
        modelCnn,
        loader,
        StopOnPlateau("Accuracy", warm_up=5, patience=5) | Threshold("Accuracy", 1.0, duration=2) | EpochStop(30),
        log_iter=100,
        profile=0,
    )


    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")
    with(open("SNN/results/training_time.txt", "a")) as f:
        f.write(f"NS-CNN: {training_time} seconds\n")

    del trained_model, nn1, nn2, nn3, net1, net2, net3
    gc.collect()

    lr = 0.001

    nn1 = SNNetwork(100, 3, time_frame=10)
    net1 = Network(nn1, "upper_legs_net", batching=True)
    net1.optimizer = torch.optim.AdamW(nn1.parameters(), lr=lr)
    net1.cuda(device)

    nn2 = SNNetwork(100, 2, time_frame=10)
    net2 = Network(nn2, "lower_legs_net", batching=True)
    net2.optimizer = torch.optim.AdamW(nn2.parameters(), lr=lr)
    net2.cuda(device)

    nn3 = SNNetwork(100, 4, time_frame=10)
    net3 = Network(nn3, "forearms_net", batching=True)
    net3.optimizer = torch.optim.AdamW(nn3.parameters(), lr=lr)
    net3.cuda(device)

    modelSnn = Model("SNN/logic.pl", [net1, net2, net3])
    modelSnn.add_tensor_source("train", train)
    modelSnn.add_tensor_source("test", test)
    modelSnn.set_engine(ExactEngine(modelSnn), cache=True)

    start_time = time.time()
    trained_model = train_model(
        modelSnn,
        loader,
        StopOnPlateau("Accuracy", warm_up=5, patience=5) | Threshold("Accuracy", 1.0, duration=2) | EpochStop(30),
        log_iter=100,
        profile=0,
    )

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")
    with(open("SNN/results/training_time.txt", "a")) as f:
        f.write(f"NS-SNN: {training_time} seconds\n")


    if TEST_ON_DIFFERENT_SETS:
        for set, name in zip([2, 6, 5, 4], ['Sub', 'Env', 'Day', 'Same']):
            # Load the test data
            test_set = load_test_data(set)

            test_net(modelSnn, test_set, f'SNN{name}1s')
            test_net(modelCnn, test_set, f'CNN{name}1s')
            print(f"Tested on {name} dataset")
            del test_set
            gc.collect()
    else:
        test_net(modelCnn, test, 'CNN1s')
        test_net(modelSnn, test, 'SNN1s')

if __name__ == "__main__":
    main()
#     cProfile.run("main()", "profile_output_cnn.prof")