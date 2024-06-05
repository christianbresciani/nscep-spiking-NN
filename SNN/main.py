import torch
import os
import pickle
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

from network import Net
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import optuna



# Set the random seed for reproducibility
random.seed(42)

# Set constants
BATCH_SIZE = 25
TIME_WINDOW = 3
CSI_PER_SECOND = 30
WINDOW_SIZE = TIME_WINDOW * CSI_PER_SECOND
STARTS = range(0, (80-TIME_WINDOW)*CSI_PER_SECOND)
SAMPLES = 80*CSI_PER_SECOND
ACTIONS = ['A', 'B', 'C', 'G', 'H', 'J', 'K']

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
def prepare_data(antennas=4, antenna_select=0):
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
        if antennas == 1:
            data = data[range(SAMPLES), ..., int(antenna_select)]
        # data = np.round(np.abs(data))

        # divide the 80s of data in 78 windows of 3s
        csi = []
        for start in STARTS:
            csi.append(data[start:start+WINDOW_SIZE, ...])
        csi = torch.Tensor(np.array(csi))

        # create labels for each window
        activity_label = ACTIONS.index(x)  # Labels depend on file
        labels = torch.Tensor(activity_label * np.ones(len(STARTS)))

        # Shuffle the data
        indices = torch.randperm(csi.size(0))
        shuffled_data = csi[indices]
        
        # Split the data into training, validation, and test sets
        train_csi = torch.cat([train_csi, shuffled_data[:1810]], dim=0)
        train_labels = torch.cat([train_labels, labels[:1810]], dim=0)

        val_csi = torch.cat([val_csi, shuffled_data[1810:2010]], dim=0)
        val_labels = torch.cat([val_labels, labels[1810:2010]], dim=0)

        test_csi = torch.cat([test_csi, shuffled_data[2010:]], dim=0)
        test_labels = torch.cat([test_labels, labels[2010:]], dim=0)

    # Normalize the CSI dataset
    max_val = max([torch.max(train_csi), torch.max(val_csi), torch.max(test_csi)])
    train_csi = torch.true_divide(train_csi, max_val)
    val_csi = torch.true_divide(val_csi, max_val)
    test_csi = torch.true_divide(test_csi, max_val)

    # create Datasets and DataLoaders
    csi_trainset = CsiDataset(train_csi, train_labels)
    csi_valset = CsiDataset(val_csi, val_labels)
    csi_testset = CsiDataset(test_csi, test_labels)

    csi_dataloader = DataLoader(csi_trainset, batch_size=BATCH_SIZE)
    csi_valloader = DataLoader(csi_valset, batch_size=BATCH_SIZE)

    return csi_dataloader, csi_valloader, csi_testset 

train, val, test = prepare_data()

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def parameter_tuning():
    # Define the objective function for Optuna
    def objective(trial : optuna.Trial):
        # Define the search space for hyperparameters
        lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
        epochs = trial.suggest_int('epochs', 10, 50, step=5)
        hidden_dim = trial.suggest_int('hidden_dim', 50, 150, step=10)
        epoch_annealing = trial.suggest_int('epoch_annealing', 15, 30)
        patience = trial.suggest_int('patience', 3, 10)

        # Set up the model
        model = Net(2048*4, hidden_dim=hidden_dim, output_dim=len(ACTIONS), num_steps=WINDOW_SIZE, device=device).to(device)

        # Train the model
        _, _, _, _, val_loss, val_acc = model.train_net(
            train,
            val,
            epochs,
            torch.optim.Adam(model.parameters(), lr=lr),
            torch.cuda.amp.GradScaler(),
            num_epochs_annealing=epoch_annealing,
            patience=patience           # Early stopping patience, works after half of the epochs
        )

        # Report the metrics to Optuna
        trial.report(val_loss, step=epochs)
        trial.report(val_acc, step=epochs)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Return the metrics for optimization
        return val_loss

    # Create an Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)

    # Get the best hyperparameters and metrics
    best_params = study.best_params
    best_val_loss = study.best_value
    best_val_accuracy = study.best_trial.user_attrs['best_val_accuracy']

    # Print the results
    print("Best Hyperparameters:")
    print(best_params)
    print("Best Validation Loss:", best_val_loss)
    print("Best Validation Accuracy:", best_val_accuracy)

    # Train the model with the best hyperparameters
    best_lr = best_params['lr']
    best_epochs = best_params['epochs']
    best_hidden_dim = best_params['hidden_dim']
    best_epoch_annealing = best_params['epoch_annealing']
    best_patience = best_params['patience']

    model = Net(2048, hidden_dim=best_hidden_dim, output_dim=len(ACTIONS), num_steps=WINDOW_SIZE, device=device).to(device)

    model.train_net(
        train,
        val,
        best_epochs,
        torch.optim.Adam(model.parameters(), lr=best_lr),
        torch.cuda.amp.GradScaler(),
        num_epochs_annealing=best_epoch_annealing,
        patience=best_patience
    )

    return model



# Train the model with hyperparameter tuning
# model = parameter_tuning()

model = Net(2048, hidden_dim=100, output_dim=len(ACTIONS), num_steps=WINDOW_SIZE, device=device).to(device)
model.train_net(
    train,
    val,
    50,
    torch.optim.Adam(model.parameters(), lr=0.0001),
    torch.cuda.amp.GradScaler(),
    num_epochs_annealing=22,
    patience=5
)

# Evaluate the model on test set
# Compute the predictions on the test set
test_preds = model.predict_data(test)

# Print the classification report
print("Classification Report:")
print(classification_report(test.labels, test_preds, target_names=ACTIONS))

# Compute the confusion matrix
cm = confusion_matrix(test.labels, test_preds)

# Print the confusion matrix
print("Confusion Matrix:")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ACTIONS)
cmdisp = disp.plot(cmap="cividis")
cmdisp.figure_.savefig("SNN/CM.png", bbox_inches='tight')