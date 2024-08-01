import gc
import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from networks import SNNetwork, CNNetwork
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


# Set the random seed for reproducibility
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Set constants
BATCH_SIZE = 25
TIME_WINDOW = 3
CSI_PER_SECOND = 30
WINDOW_SIZE = int(TIME_WINDOW * CSI_PER_SECOND)
TRAIN_STARTS = range(0, 80*CSI_PER_SECOND-int(TIME_WINDOW*CSI_PER_SECOND), int(TIME_WINDOW*CSI_PER_SECOND/30))
TEST_STARTS = range(0, 80*CSI_PER_SECOND-int(TIME_WINDOW*CSI_PER_SECOND), CSI_PER_SECOND)
ACTIONS = ['A', 'B', 'C', 'G', 'H', 'J', 'K'] # [Walk, Run, Jump, Wave hands, Clapping, Wiping, Squat]

# Define the CSI dataset class
class CsiDataset(Dataset):
    def __init__(self, data, labels):

        self.csi = data
        self.labels = labels


    def __len__(self):
        return self.csi.shape[0]

    def __getitem__(self, idx):
        return self.csi[idx], self.labels[idx]
    
# A function to load the data from different files and create the Datasets and DataLoaders
def load_train_data():
    train_csi = torch.Tensor()
    train_labels = torch.Tensor()
    val_csi = torch.Tensor()
    val_labels = torch.Tensor()

    for x in ACTIONS:
        if os.getcwd().split('/')[-1] == 'SNN':
            os.chdir('..')

        filetrain = f'datasets/mean_dataset_abs/S1a_{x}.pkl'
        # Load CSI data from pickle file
        with open(filetrain, 'rb') as f:
            datatrain = pickle.load(f)       # WARNING This code does not handle exceptions for simplicity exceptions would require keeping track of indices

        # divide the 80s of data in 2310 windows of 3s
        csidata = []
        for start in TRAIN_STARTS:
            csidata.append(datatrain[start:start+WINDOW_SIZE, ...])

        csidata = torch.Tensor(np.array(csidata))

        # create labels for each window
        activity_label = ACTIONS.index(x)  # Labels depend on file
        labels = torch.nn.functional.one_hot(torch.Tensor(activity_label * np.ones(len(TRAIN_STARTS))).to(torch.int64), len(ACTIONS))

        sep = int(0.90 * len(csidata))
        # Split the data into training and validation sets
        train_csi = torch.cat([train_csi, csidata[:sep]], dim=0)
        train_labels = torch.cat([train_labels, labels[:sep]], dim=0)

        val_csi = torch.cat([val_csi, csidata[sep:]], dim=0)
        val_labels = torch.cat([val_labels, labels[sep:]], dim=0)

        del csidata, labels, datatrain
        gc.collect()

    # Normalize the CSI dataset
    max_val = max([torch.max(train_csi), torch.max(val_csi)])
    train_csi = torch.true_divide(train_csi, max_val)
    val_csi = torch.true_divide(val_csi, max_val)
    train_indices = torch.randperm(train_csi.size(0))
    val_indices = torch.randperm(val_csi.size(0))
    gc.collect()

    # create Datasets and DataLoaders
    csi_trainset = CsiDataset(train_csi[train_indices], train_labels[train_indices])
    csi_valset = CsiDataset(val_csi[val_indices], val_labels[val_indices])

    csi_dataloader = DataLoader(csi_trainset, batch_size=BATCH_SIZE)
    csi_valloader = DataLoader(csi_valset, batch_size=BATCH_SIZE)

    return csi_dataloader, csi_valloader

def test_net(model, test, name):
    # Compute the predictions
    preds, labels = model.predict_data(test)

    # Print the classification report
    print("Classification Report different subject:")
    print(classification_report(labels, preds, target_names=ACTIONS))

    # Compute the confusion matrix
    cm = confusion_matrix(labels, preds)

    # Save the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ACTIONS)
    cmdisp = disp.plot(cmap="cividis")
    cmdisp.figure_.savefig(f"SNN/results/neuralOnly/ConfMat{name}.png", bbox_inches='tight')

def load_test_data(set=2):
    test_csi = torch.Tensor()
    test_labels = torch.Tensor()

    for x in ACTIONS:
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

        # create labels for each window
        activity_label = ACTIONS.index(x)  # Labels depend on file
        labels = torch.nn.functional.one_hot(torch.Tensor(activity_label * np.ones(len(TEST_STARTS))).to(torch.int64), len(ACTIONS))

        test_csi = torch.cat([test_csi, csidata], dim=0)
        test_labels = torch.cat([test_labels, labels], dim=0)

    # Normalize the CSI dataset
    max_val = torch.max(test_csi)
    test_csi = torch.true_divide(test_csi, max_val)

    csi_testset = CsiDataset(test_csi, test_labels)

    return csi_testset










train, val = load_train_data()

# Set the device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

modelSnn = SNNetwork(100, len(ACTIONS), 10, reset_mechanism='subtract', device=device).to(device)
modelSnn.train_net(
    train,
    val,
    20,
    torch.optim.Adam(modelSnn.parameters(), lr=0.0009053778099794136),
    num_epochs_annealing=18,
    patience=5
)


modelCnn = CNNetwork(len(ACTIONS), device=device).to(device)
modelCnn.train_net(
    train,
    val,
    35,
    torch.optim.Adam(modelCnn.parameters(), lr=0.0001),
    num_epochs_annealing=24,
    patience=5
)

del train, val
gc.collect()

# Evaluate the model on different subject or environment or day, and same as dataset
for set, name in zip([2, 6, 5, 4], ['Sub', 'Env', 'Day', 'Same']):
    # Load the test data
    test_set = load_test_data(set)

    test_net(modelSnn, test_set, f'SNN{name}3s')
    test_net(modelCnn, test_set, f'CNN{name}3s')
    print(f"Tested on {name} dataset")
    del test_set
    gc.collect()
