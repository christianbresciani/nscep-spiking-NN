import gc
import torch.nn as nn
import snntorch as snn
import torch
from tqdm.auto import tqdm
from torch.cuda.amp import autocast
import copy
import numpy as np
from sklearn.metrics import accuracy_score

class Net(nn.Module):
   def __init__(self, input_dim, hidden_dim, output_dim, num_steps, reset_mechanism='zero', device=None):
      super().__init__()
      assert reset_mechanism in ['zero', 'subtract'], "reset_mechanism must be either 'zero' or 'subtract'"

      self.hidden_dim = hidden_dim
      self.num_outputs = output_dim
      self.num_steps = num_steps
      self.device = device

      # initialize layers
      self.antennas_fuse = nn.Conv2d(4, 1, (1,1))
      self.hidden_linear = nn.Linear(input_dim, hidden_dim)
      self.snn1 = snn.Leaky(.95, reset_mechanism=reset_mechanism) #, learn_beta=True, learn_threshold=True, reset_delay=False
      self.snn2 = snn.Leaky(.95, reset_mechanism=reset_mechanism) #, learn_beta=True, learn_threshold=True, reset_delay=False

      # self.time_fuse = nn.Conv1d(num_steps, 1, 1)
      self.output_linear = nn.Linear(hidden_dim, output_dim)
      self.softmax = nn.Softmax(dim=1)

   def forward(self, x):
      mem = self.snn1.init_leaky()
      mem2 = self.snn2.init_leaky()

      # spk_rec = []  # Record the output trace of spikes

      if len(x.shape) < 4: x = x.unsqueeze(0)

      x = x.permute(0, 3, 1, 2) # [batch, antennas, time, freq]
      x = self.antennas_fuse(x) # fuse the 4 antennas into 1 channel
      x = x.squeeze(1)

      if x.shape[1] != self.num_steps:
         x = x.view(x.shape[0], -1, self.num_steps, x.shape[2]) # [batch, time/steps, steps, freq]

      for step in range(self.num_steps):
         sn_input = x[:,:,step,:]
         # sn_input = self.antennas_fuse(input.reshape(input.shape[0], input.shape[-1], input.shape[1], -1)) # fuse the 4 antennas into 1 channel
         
         # sn_input = self.hidden_linear(sn_input)

         spk, mem = self.snn1(sn_input, mem)

         # spk = self.hidden_linear(spk)
         # spk, mem2 = self.snn2(spk, mem2)

         # spk_rec.append(spk)
         # mem1_rec.append(mem1)

      # spk_rec = torch.stack(spk_rec, dim=2)
      # fused_spikes = self.time_fuse(spk_rec)
      # out = self.output_linear(fused_spikes.squeeze(1))

      # hidden = spk_rec[-1]
      hidden = self.hidden_linear(spk)

      # spk2_rec = []
      for step in range(hidden.shape[1]):
         spk2, mem2 = self.snn2(hidden[:,step,:], mem2)
         # spk2_rec.append(spk2)

      out = self.output_linear(spk2)

      return self.softmax(out)

   def train_net(self, train_data, val_data, epochs, optimizer, scaler, num_epochs_annealing=22, patience=np.inf):

      train_losses = []
      val_losses = []
      accuracies = []
      best_val_loss = np.Inf
      best_ep = 0
      # best_model = copy.deepcopy(self.state_dict())

      for ep in tqdm(range(epochs)):
         self.train()
         ep_loss = 0.0

         for batch in train_data:
            optimizer.zero_grad()

            data, labels = [b.to(self.device) for b in batch]

            with autocast():
               logits = self(data)
               labels = labels.float()
               loss = self.mse_loss(labels, logits, ep, num_epochs_annealing)

            ep_loss += loss

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

         train_losses.append(ep_loss/len(train_data))

         # Evaluation
         val_loss = self.eval_model(val_data, ep, num_epochs_annealing)
         print(f"Epoch {ep}| val loss: {np.round(val_loss.cpu(),4)}")#, accuracy: {np.round(accuracy,4)}")

         # accuracies.append(accuracy)
         val_losses.append(val_loss)

         if val_loss < best_val_loss:
            best_val_loss = val_loss
            # best_model = copy.deepcopy(self.state_dict())
            best_ep = ep
         
         if ep - best_ep >= patience: # Early stopping
            print(f"Early stopping at epoch {ep}")
            break
         
         gc.collect()

      # torch.save(best_model, f"./SNN/best_model.pth")

      return train_losses, accuracies, val_losses, best_ep, best_val_loss#, accuracies[best_ep]

   def KL(self, alpha):
      beta = torch.ones((1, self.num_outputs), dtype=torch.float32, device=self.device)
      S_alpha = torch.sum(alpha, dim=1, keepdim=True)
      S_beta = torch.sum(beta, dim=1, keepdim=True)

      lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
      lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

      dg0 = torch.digamma(S_alpha)
      dg1 = torch.digamma(alpha)

      return torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni

   # Loss function considering the expected squared error and the KL divergence
   def mse_loss(self, target, pred, ep, num_epochs_annealing):
      alpha = pred + 1
      S = torch.sum(alpha, dim=1, keepdims=True)
      m = alpha / S

      # A + B minimises the sum of squared loss, see discussion in EDL paper for the derivation
      A = torch.sum((target-m)**2, dim=1, keepdims=True)
      B = torch.sum(alpha*(S-alpha)/(S*S*(S+1)), dim=1, keepdims=True)

      # the lambda_t parameter, in this case min{1, t/10} with t the number of epochs
      ll = min(1.0, float(ep/float(num_epochs_annealing)))

      alp = pred*(1-target) + 1 
      C =  ll * self.KL(alp)

      return torch.mean(A + B + C) # mean over batch

   def eval_model(self, val_data, ep, num_epochs_annealing):
      self.eval()

      val_loss = 0.0

      with torch.no_grad():
         for batch in val_data:
               data, labels = [b.to(self.device) for b in batch]

               with autocast():
                  logits = self(data)
                  labels = labels.float()
                  loss = self.mse_loss(labels, logits, ep, num_epochs_annealing)

                  val_loss += loss

                  # acc = accuracy_score(logits.round().cpu(), labels.cpu())

      return val_loss/len(val_data)#, float(acc)

   def predict_data(self, test_data):
      self.eval()

      test_preds = []
      test_labels = []

      with torch.no_grad():
         for batch in tqdm(test_data, total=len(test_data)):
               data, labels = [b.to(self.device) for b in batch]

               with autocast():
                  logits = self(data)

               test_preds.append(logits.round().cpu().numpy().argmax())
               test_labels.append(labels.cpu().numpy().argmax())

      return test_preds, test_labels