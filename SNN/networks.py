import gc
import pickle
import torch.nn as nn
import snntorch as snn
import torch
from tqdm.auto import tqdm
import copy
import numpy as np

class Net(nn.Module):
   def __init__(self, device = None):
       super().__init__()
       self.device = device

   def train_net(self, train_data, val_data, epochs, optimizer, num_epochs_annealing=22, patience=np.inf):

      val_losses = []
      accuracies = []
      best_val_loss = np.Inf
      best_ep = 0
      # best_model = copy.deepcopy(self.state_dict())

      for ep in tqdm(range(epochs)):
         self.train()

         for batch in train_data:
            optimizer.zero_grad()

            data, labels = batch


            logits = self(data.to(self.device))
            labels = labels.float().to(self.device)
            loss = torch.mean(self.loss(labels, logits, ep, num_epochs_annealing))

            # Backpropagation
            loss.backward()
            optimizer.step()

         # Evaluation
         val_loss = self.eval_model(val_data, ep, num_epochs_annealing)
         print(f"Epoch {ep}| val loss: {np.round(val_loss.cpu(),4)}")#, accuracy: {np.round(accuracy,4)}")

         # accuracies.append(accuracy)
         val_losses.append(val_loss)

         if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(self.state_dict())
            best_ep = ep
         
         if ep - best_ep >= patience: # Early stopping
            print(f"Early stopping at epoch {ep}")
            break
         
         gc.collect()

      self.load_state_dict(best_model)
      pickle.dump(best_model, open(f'bestmodel{self.__class__}.pkl', 'wb'))

      return accuracies, val_losses, best_ep, best_val_loss#, accuracies[best_ep]

   def eval_model(self, val_data, ep, num_epochs_annealing):
      self.eval()

      val_loss = []

      with torch.no_grad():
         for batch in val_data:
            data, labels = [b.to(self.device) for b in batch]

            logits = self(data)
            labels = labels.float()
            loss = self.loss(labels, logits, ep, num_epochs_annealing)

            val_loss += loss

               # acc = accuracy_score(logits.round().cpu(), labels.cpu())

      return torch.mean(torch.Tensor(val_loss).view(-1).to(self.device))#, float(acc)

   def predict_data(self, test_data):
      self.eval()

      test_preds = []
      test_labels = []

      with torch.no_grad():
         for batch in tqdm(test_data, total=len(test_data)):
            data, labels = [b.to(self.device) for b in batch]

            logits = self(data)

            test_preds.append(logits.round().cpu().numpy().argmax())
            test_labels.append(labels.cpu().numpy().argmax())

      return test_preds, test_labels
   


class CustomLoss(nn.Module):
    def __init__(self, num_outputs, device):
        super(CustomLoss, self).__init__()
        self.num_outputs = num_outputs
        self.device = device

    def KL(self, alpha):
        beta = torch.ones((1, self.num_outputs), dtype=torch.float32, device=self.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)

        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)

        return torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni

    def forward(self, target, pred, ep, num_epochs_annealing):
        alpha = pred + 1
        S = torch.sum(alpha, dim=1, keepdims=True)
        m = alpha / S

        # A + B minimizes the sum of squared loss
        A = torch.sum((target - m)**2, dim=1, keepdims=True)
        B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdims=True)

        # lambda_t parameter
        ll = min(1.0, float(ep) / float(num_epochs_annealing))

        alp = pred * (1 - target) + 1
        C = ll * self.KL(alp)

        return A + B + C
    

class SNNetwork(Net):
   def __init__(self, hidden_dim, output_dim, num_steps, reset_mechanism='zero', device=None):
      super().__init__(device=device)
      assert reset_mechanism in ['zero', 'subtract'], "reset_mechanism must be either 'zero' or 'subtract'"

      self.num_steps = num_steps

      # initialize layers
      self.antennas_fuse = nn.Conv2d(4, 1, (1,1))
      self.hidden_linear1 = nn.Linear(2048, hidden_dim)
      self.hidden_linear2 = nn.Linear(hidden_dim, hidden_dim//2)
      self.snn1 = snn.Leaky(.95, reset_mechanism=reset_mechanism, learn_beta=True, learn_threshold=True, reset_delay=False) #
      self.snn2 = snn.Leaky(.95, reset_mechanism=reset_mechanism, learn_beta=True, learn_threshold=True, reset_delay=False) #

      # self.time_fuse = nn.Conv1d(num_steps, 1, 1)
      self.output_linear = nn.Linear(hidden_dim//2, output_dim)
      self.softmax = nn.Softmax(dim=1)

      self.loss = CustomLoss(num_outputs=output_dim, device=device)

   def forward(self, x):
      mem = self.snn1.init_leaky()
      mem2 = self.snn2.init_leaky()

      if len(x.shape) < 4: x = x.unsqueeze(0)

      x = x.permute(0, 3, 1, 2) # [batch, antennas, time, freq]
      x = self.antennas_fuse(x) # fuse the 4 antennas into 1 channel averaging  the values
      x = x.squeeze(1)

      if x.shape[1] != self.num_steps:
         x = x.view(x.shape[0], -1, self.num_steps, x.shape[2]) # [batch, time/steps, steps, freq]

      x = self.hidden_linear1(x)
      for step in range(self.num_steps):
         sn_input = x[:,:,step,:]
         spk, mem = self.snn1(sn_input, mem)

      hidden = self.hidden_linear2(spk)

      for step in range(x.shape[1]):
         spk2, mem2 = self.snn2(hidden[:,step,:], mem2)

      out = self.output_linear(spk2)

      return self.softmax(out)
   

class CNNetwork(Net):
   def __init__(self, output_dim, device=None):
      super().__init__(device=device)

      # fuse antennas
      self.antennas_fuse = nn.Conv2d(4, 1, (1,1))

      # initialize layers
      self.hidden_conv = nn.Conv2d(1, 4, (5,5), stride=(3,2)) # (3,5), stride=(2,2) 90 -> 32 -> 16 -> 6 -> 3 
      self.maxpool = nn.MaxPool2d((2,5)) # (2,5)
      self.hidden_conv2 = nn.Conv2d(4, 16, (5,11), stride=(2,5)) # (3,11), stride=(2,5)
      self.maxpool2 = nn.MaxPool2d((2,2)) # (1,2)
      self.hidden_conv3 = nn.Conv2d(16, 25, (3,5), stride=(2,2)) # (3,5), stride=(2,2)
      
      # self.hidden_conv = nn.Conv2d(4, 32, (5,8), stride=(5,8), padding='valid') # [batch, 4, 30, 2048] -> [batch, 32, 6, 256]
      # self.hidden_conv2 = nn.Conv2d(32, 32, (3,8), stride=(3,8), padding='valid') # [batch, 32, 6, 256] -> [batch, 32, 2, 32]
      # self.hidden_conv3 = nn.Conv2d(32, 32, (2,4), stride=(2,4), padding='valid') # [batch, 32, 2, 32] -> [batch, 32, 1, 8]
      
      self.flatten = nn.Flatten()
      self.output_linear = nn.Linear(200, output_dim)
      self.softmax = nn.Softmax(dim=1)
      self.loss = CustomLoss(num_outputs=output_dim, device=device)

   def forward(self, x):
      if len(x.shape) < 4: x = x.unsqueeze(0)

      x = x.permute(0, 3, 1, 2) # [batch, antennas, time, freq]
      x = self.antennas_fuse(x)
      x = nn.functional.relu(self.hidden_conv(x))
      x = self.maxpool(x)
      x = nn.functional.relu(self.hidden_conv2(x)) 
      x = self.maxpool2(x)
      x = nn.functional.relu(self.hidden_conv3(x))
      x = self.output_linear(self.flatten(x))

      return self.softmax(x)



class SNNetwork2(Net):
   def __init__(self, hidden_dim, output_dim, num_steps, reset_mechanism='zero', device=None):
      super().__init__(device=device)
      assert reset_mechanism in ['zero', 'subtract'], "reset_mechanism must be either 'zero' or 'subtract'"

      self.num_steps = num_steps

      # initialize layers
      self.antennas_fuse = nn.Conv2d(4, 1, (1,1))
      self.hidden_linear1 = nn.Linear(2048, hidden_dim)
      self.hidden_linear2 = nn.Linear(hidden_dim, 1)
      self.snn1 = snn.Leaky(.95, reset_mechanism=reset_mechanism, learn_beta=True, learn_threshold=True, reset_delay=False)
      self.snn2 = snn.Leaky(.95, reset_mechanism=reset_mechanism, learn_beta=True, learn_threshold=True, reset_delay=False)

      # self.loss = CustomLoss(num_outputs=output_dim, device=device)

   def forward(self, x):
      mem = self.snn1.init_leaky()
      mem2 = self.snn2.init_leaky()

      if len(x.shape) < 4: x = x.unsqueeze(0)

      x = x.permute(0, 3, 1, 2)
      x = self.antennas_fuse(x)
      x = x.squeeze(1)

      if x.shape[1] != self.num_steps:
         x = x.view(x.shape[0], 2, -1, self.num_steps, x.shape[2])

      x = self.hidden_linear1(x)
      spks = torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4], device=self.device) # [batch, 2, time/steps, steps, hdim]
      for step in range(self.num_steps):
         sn_input = x[:,:,:,step,:]
         spk, mem = self.snn1(sn_input, mem)

         spks[:,:,:,step,:] = spk
      
      hidden = self.hidden_linear2(spks.sum(dim=3))
      hidden = hidden.squeeze(3) # [batch, 2, time/steps]
      spks = torch.zeros(hidden.shape[0], hidden.shape[1], hidden.shape[2], device=self.device)

      for step in range(hidden.shape[2]):
         spk2, mem2 = self.snn2(hidden[:,:,step], mem2)

         spks[:,:,step] = spk2

      return spks