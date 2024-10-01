from typing import List
import torch
import torch.nn as nn

import snntorch as snn

WINDOWS = [[0, 45], [0,15], [15,30], [30, 45]]

class CNNetwork(nn.Module):
   def __init__(self, num_classes=2, time_frame=0):
      super(CNNetwork, self).__init__()
      self.final = nn.Softmax(1)
      self.num_classes = num_classes
      self.time_frame = time_frame

      if time_frame == 0:
         # Define the CNN layers and the classifier the comments are the values for 1s input
         self.conv = nn.Sequential( # 4, 30, 2048
            nn.Conv2d(4, 1, (1,1)), # 1, 30, 2048
            nn.Conv2d(1, 4, (3,5), stride=(2,2)), # 4, 14, 1022
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,5)), # 4, 7, 204
            nn.Conv2d(4, 6, (3,11), stride=(2,5)), # 6, 3, 39
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,2)), # 6, 3, 19
            nn.Conv2d(6, 8, (3,5), stride=(2,2)), # 8, 1, 8
         )
      
      else:
         self.conv = nn.Sequential( # 4, 15, 2048
            nn.Conv2d(4, 1, (1,1)), # 1, 15, 2048
            nn.Conv2d(1, 4, (2,5), stride=(1,2)), # 4, 14, 1022
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,5)), # 4, 7, 204
            nn.Conv2d(4, 6, (3,11), stride=(2,5)), # 6, 3, 39
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,2)), # 6, 3, 19
            nn.Conv2d(6, 8, (3,5), stride=(2,2)), # 8, 1, 8
         )

      self.classifier = nn.Sequential(            
         nn.Linear(64, self.num_classes),
         self.final,
      )

   def forward(self, x: torch.Tensor) -> torch.Tensor:
      x = x[:, WINDOWS[self.time_frame][0]:WINDOWS[self.time_frame][1], :, :]
      x = x.permute(0, 3, 1, 2)
      x = self.conv(x)
      x = x.flatten(1)
      x = self.classifier(x)
      return x
    


class SNNetwork(nn.Module):
   def __init__(self, hidden_dim, num_classes, num_steps=15, time_frame=0):
      super(SNNetwork, self).__init__()

      self.num_classes = num_classes
      self.time_frame = time_frame
      if time_frame != 0:
         self.num_steps = 5
      else:
         self.num_steps = num_steps

      # initialize layers
      self.antennas_fuse = nn.Conv2d(4, 1, (1,1))
      self.hidden_linear1 = nn.Linear(2048, hidden_dim)
      self.snn1 = snn.Leaky(.95, reset_mechanism='subtract', learn_beta=True, learn_threshold=True, reset_delay=False)
      self.hidden_linear2 = nn.Linear(hidden_dim, num_classes)
      self.snn2 = snn.Leaky(.95, reset_mechanism='subtract', learn_beta=True, learn_threshold=True, reset_delay=False)

      # self.loss = CustomLoss(num_outputs=output_dim, device=device)

   def forward(self, x):
      mem = self.snn1.init_leaky()
      mem2 = self.snn2.init_leaky()

      if len(x.shape) < 4: x = x.unsqueeze(0)

      x = x[:, WINDOWS[self.time_frame][0]:WINDOWS[self.time_frame][1], :, :]

      x = x.permute(0, 3, 1, 2)
      x = self.antennas_fuse(x)
      x = x.squeeze(1)

      if x.shape[1] != self.num_steps:
         x = x.view(x.shape[0], -1, self.num_steps, x.shape[2])

      x = self.hidden_linear1(x)
      spks = torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device) # [batch, time/steps, steps, hdim]
      for step in range(self.num_steps):
         sn_input = x[:,:,step,:]
         spk, mem = self.snn1(sn_input, mem)

         spks[:,:,step,:] = spk
      
      hidden = self.hidden_linear2(spks.sum(dim=2))
      spks = torch.zeros(hidden.shape[0], hidden.shape[1], self.num_classes, device=hidden.device)

      for step in range(hidden.shape[1]):
         spk2, mem2 = self.snn2(hidden[:,step,:], mem2)

         spks[:,step,:] = spk2

      # return nn.functional.softmax(spks[-1], dim=1)
      return nn.functional.softmax(spks.sum(dim=1), dim=1)

from deepproblog.dataset import Dataset
from deepproblog.model import Model

def predict(model: Model, dataset: Dataset, labels: List[str]):
   model.eval()
   predictions = []
   actuals = []
   probabilities = {label: [] for label in labels}
   for query in dataset.to_queries():
      test_query = query.variable_output()
      answer = model.solve([test_query])[0]
      actual = str(query.output_values()[0])
      max_ans = max(answer.result, key=lambda x: answer.result[x])
      predicted = str(max_ans.args[query.output_ind[0]])
      try:
         probabilities[predicted].append(answer.result[max_ans].item())
      except:
         probabilities[predicted].append(answer.result[max_ans])
         continue
      predictions.append(labels.index(predicted))
      actuals.append(labels.index(actual))
   return predictions, actuals, probabilities



