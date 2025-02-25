from matplotlib import pyplot as plt
import torch
import os
import gc
import pickle
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

from networks import SNNetwork, CNNetwork
from metrics import bayesianHypothesisTesting, bayesian_hypothesis_testing

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import optuna
from optuna.storages import JournalStorage, JournalFileStorage

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
STARTS = range(0, 80*CSI_PER_SECOND-int(TIME_WINDOW*CSI_PER_SECOND))
ACTIONS = ['A', 'B', 'C', 'G', 'H', 'J', 'K'] # 
LABELS = ['Walk', 'Run', 'Jump', 'Wave hands', 'Clapping', 'Wiping', 'Squat']

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
def prepare_data():
    train_csi = torch.Tensor()
    train_labels = torch.Tensor()
    val_csi = torch.Tensor()
    val_labels = torch.Tensor()
    test_csi = torch.Tensor()
    test_labels = torch.Tensor()

    for x in ACTIONS:
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

        # create labels for each window
        activity_label = ACTIONS.index(x)  # Labels depend on file
        labels = torch.nn.functional.one_hot(torch.Tensor(activity_label * np.ones(len(STARTS))).to(torch.int64), len(ACTIONS))
        
        # Split the data into training, validation, and test sets [79%, 8%, 13%]
        train_csi = torch.cat([train_csi, csi[:1810]], dim=0)
        train_labels = torch.cat([train_labels, labels[:1810]], dim=0)

        val_csi = torch.cat([val_csi, csi[1810:2010]], dim=0)
        val_labels = torch.cat([val_labels, labels[1810:2010]], dim=0)

        test_csi = torch.cat([test_csi, csi[2010:]], dim=0)
        test_labels = torch.cat([test_labels, labels[2010:]], dim=0)

        del labels, csi, data
        gc.collect()

    # Normalize the CSI dataset
    max_val = max([torch.max(train_csi), torch.max(val_csi), torch.max(test_csi)])
    train_csi = torch.true_divide(train_csi, max_val)
    gc.collect()
    val_csi = torch.true_divide(val_csi, max_val)
    gc.collect()
    test_csi = torch.true_divide(test_csi, max_val)
    gc.collect()

    indices = torch.randperm(train_csi.size(0))

    # create Datasets and DataLoaders
    csi_trainset = CsiDataset(train_csi[indices], train_labels[indices])
    csi_valset = CsiDataset(val_csi, val_labels)
    csi_testset = CsiDataset(test_csi, test_labels)

    csi_dataloader = DataLoader(csi_trainset, batch_size=BATCH_SIZE)
    csi_valloader = DataLoader(csi_valset, batch_size=BATCH_SIZE)

    return csi_dataloader, csi_valloader, csi_testset 

train, val, test = prepare_data()

# Set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define a function to perform hyperparameter tuning
def parameter_tuning(trials=10):
    # Define the objective function for Optuna
    patience = 5
    def objective(trial : optuna.Trial):
        # Define the search space for hyperparameters
        step = trial.suggest_categorical('step', [2, 3, 5, 6, 10, 15])
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        epochs = trial.suggest_int('epochs', 20, 40, step=5)
        hidden_dim = trial.suggest_int('hidden_dim', 50, 500, step=50)
        epoch_annealing = trial.suggest_int('epoch_annealing', 12, 30)
        reset_mechanism = trial.suggest_categorical('reset_mechanism', ['zero', 'subtract'])

        # Set up the model
        model = SNNetwork(hidden_dim, len(ACTIONS), step, reset_mechanism=reset_mechanism, device=device).to(device)
        # Train the model
        _, _, _, val_loss = model.train_net(
            train,
            val,
            epochs,
            torch.optim.Adam(model.parameters(), lr=lr),
            num_epochs_annealing=epoch_annealing,
            patience=patience           # Early stopping patience, works after half of the epochs
        )

        # Report the metrics to Optuna
        trial.report(val_loss, step=epochs)
        # trial.report(-val_acc, step=epochs)

        if trial.should_prune():
            raise optuna.TrialPruned()
        
        torch.cuda.empty_cache()
        
        # Return the metrics for optimization
        return val_loss


    study_name = "snn-study-1s"
    study_path = "SNN/optuna.log"
    storage_name = JournalStorage(JournalFileStorage(study_path))
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.SuccessiveHalvingPruner()
    )

    # Create an Optuna study
    # optuna.logging.set_verbosity(optuna.logging.DEBUG)
    study.optimize(objective, n_trials=trials, gc_after_trial=True)

    return study
    


# Train the model with hyperparameter tuning
# study = parameter_tuning(trials=5)

# # Get the best hyperparameters and metrics
# best_params = study.best_params
# best_val_loss = study.best_value
# # best_val_accuracy = study.best_trial.user_attrs['best_val_accuracy']

# # Print the results
# print("Best Hyperparameters:")
# print(best_params)
# print("Best Validation Loss:", best_val_loss)
# # print("Best Validation Accuracy:", best_val_accuracy)

# # Train the model with the best hyperparameters
# best_lr = best_params['lr']
# best_epochs = best_params['epochs']
# best_hidden_dim = best_params['hidden_dim']
# best_epoch_annealing = best_params['epoch_annealing']
# best_reset_mechanism = best_params['reset_mechanism']
# best_step = best_params['step']

# model = SNNetwork(best_hidden_dim, len(ACTIONS), best_step, reset_mechanism=best_reset_mechanism, device=device).to(device)

# model.train_net(
#     train,
#     val,
#     best_epochs,
#     torch.optim.Adam(model.parameters(), lr=best_lr),
#     num_epochs_annealing=best_epoch_annealing,
#     patience=5
# )


def main():
    
    model = SNNetwork(50, len(ACTIONS), 5, reset_mechanism='subtract', device=device).to(device)
    res = model.train_net(
        train,
        val,
        35,
        torch.optim.Adam(model.parameters(), lr=0.006540246981917088),
        num_epochs_annealing=30,
        patience=5
    )



    # Evaluate the model on test set
    # Compute the predictions on the test set
    preds, labels = model.predict_data(test)

    # Print the classification report
    print("Classification Report:")
    print(classification_report(labels, preds, target_names=LABELS))

    # Compute the confusion matrix
    cmSNN = confusion_matrix(labels, preds)

    # train convolotional model
    model = CNNetwork(len(ACTIONS), device=device, last_cannel=65).to(device)
    model.train_net(
        train,
        val,
        35,
        torch.optim.Adam(model.parameters(), lr=0.0003062326975389536),
        num_epochs_annealing=27,
        patience=5
    )

    # Evaluate the model on test set
    # Compute the predictions on the test set
    preds, labels = model.predict_data(test)

    # Print the classification report
    print("Classification Report:")
    print(classification_report(labels, preds, target_names=LABELS))

    # Compute the confusion matrix
    cmCNN = confusion_matrix(labels, preds)

    return cmSNN, cmCNN

def optunaTrain():
    # Define the objective function for Optuna
    patience = 5
    def objective(trial : optuna.Trial):
        # Define the search space for hyperparameters
        lr = trial.suggest_float('lr', 1e-7, 1e-2, log=True)
        epochs = trial.suggest_int('epochs', 15, 40, step=5)
        last_channel = trial.suggest_int('last_channel', 25, 90, step=5)
        epoch_annealing = trial.suggest_int('epoch_annealing', 12, 30)

        while True:
            # Set up the model
            model = CNNetwork(len(ACTIONS), device=device, last_cannel=last_channel).to(device)
            # Train the model
            _, _, _, val_loss = model.train_net(
                train,
                val,
                epochs,
                torch.optim.Adam(model.parameters(), lr=lr),
                num_epochs_annealing=epoch_annealing,
                patience=patience           # Early stopping patience, works after half of the epochs
            )
            
            if val_loss < 0.8: break

        # Report the metrics to Optuna
        trial.report(val_loss, step=epochs)
        # trial.report(-val_acc, step=epochs)

        if trial.should_prune():
            raise optuna.TrialPruned()
        
        torch.cuda.empty_cache()
        
        # Return the metrics for optimization
        return val_loss


    study_name = "cnn-study-1s"
    study_path = "SNN/optuna.log"
    storage_name = JournalStorage(JournalFileStorage(study_path))
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.SuccessiveHalvingPruner()
    )

    # Create an Optuna study
    # optuna.logging.set_verbosity(optuna.logging.DEBUG)
    study.optimize(objective, n_trials=40, gc_after_trial=True)

    return study


if __name__ == "__main__":
    # optunaTrain()
    # parameter_tuning(40)
    # quit()
    cmSNN = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    cmCNN = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for _ in range(10):
        snn, cnn = main()
        cmSNN += snn
        cmCNN += cnn


    # Print the confusion matrix
    print("Confusion Matrix:")
    disp = ConfusionMatrixDisplay(confusion_matrix=cmSNN, display_labels=LABELS)
    cmdisp = disp.plot(cmap="cividis")
    plt.setp(cmdisp.ax_.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(cmdisp.ax_.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    cmdisp.figure_.set_size_inches(12, 12)
    cmdisp.figure_.savefig("SNN/results/neuralOnly/ConMatSNN1s.png", bbox_inches='tight')


    # Print the confusion matrix
    print("Confusion Matrix:")
    disp = ConfusionMatrixDisplay(confusion_matrix=cmCNN, display_labels=LABELS)
    cmdisp = disp.plot(cmap="cividis")
    plt.setp(cmdisp.ax_.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(cmdisp.ax_.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    cmdisp.figure_.set_size_inches(12, 12)
    cmdisp.figure_.savefig("SNN/results/neuralOnly/ConfMatCnn1s.png", bbox_inches='tight')

    print(f'Accuracy SNN: {np.trace(cmSNN)/np.sum(cmSNN)}')
    print(f'Accuracy CNN: {np.trace(cmCNN)/np.sum(cmCNN)}')

    # bayesianHypothesisTesting(cmSNN, cmCNN)

    bayesian_hypothesis_testing(cmSNN, cmCNN)
