# -*- coding: utf-8 -*-
"""
DNN model for dataset provided by Shashank
"""

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import config as cfg

data = pd.read_csv('trainData.data', delimiter = ' ', header = None)

TEST_data = pd.read_csv('testData1.data', delimiter = ' ', header = None)

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

X_TEST = TEST_data.iloc[:, :-1]
Y_TEST = TEST_data.iloc[:, -1]

# Standardizing data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_TEST = scaler.transform(X_TEST)
print(X)
#scaler.fit_transform(X)

X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, Y, train_size=cfg.train_size, shuffle=True)

print(X_train_raw)

X_train = torch.tensor(X_train_raw, dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1,1)
X_valid = torch.tensor(X_val_raw, dtype=torch.float32)
y_valid = torch.tensor(y_val.to_numpy(), dtype=torch.float32).reshape(-1, 1)

#for test data
X_TEST = torch.tensor(X_TEST, dtype=torch.float32)
Y_TEST = torch.tensor(Y_TEST.to_numpy(), dtype=torch.float32).reshape(-1, 1)

batch_size = cfg.batch_size

train_loader = DataLoader(list(zip(X_train,y_train)), shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(list(zip(X_valid,y_valid)), shuffle=True, batch_size=batch_size)
TEST_loader = DataLoader(list(zip(X_TEST,Y_TEST)), shuffle=True, batch_size=batch_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
      super().__init__()
      self.stack = nn.Sequential(
          nn.Linear(cfg.input_features, cfg.layer_neurons[0]),
          nn.ReLU(),
          nn.Linear(cfg.layer_neurons[0], cfg.layer_neurons[1]),
          nn.ReLU(),
          nn.Linear(cfg.layer_neurons[1],cfg.output_features))
    def forward(self, x):
      logits = self.stack(x)
      return logits

#self.layer1 = nn.Linear(input_size, hidden_size)
#self.layer2 = nn.Linear(hidden_size, hidden_size)
#self.layer3 = nn.Linear(hidden_size, num_classes)
#self.activation_function = activation_function

 #   def forward(self, x):
 #       x = self.activation_function(self.layer1(x))
 #       x = self.activation_function(self.layer2(x))
  #      x = self.layer3(x)
   #     return x

model = NeuralNetwork()
#print(list(model.parameters()))
#pred = model(X_train)
#print(pred)
loss_fn = nn.MSELoss()
#loss = loss_fn(pred, y_train)
optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
# Utility function to train the model

def trdnn(model, criterion, optimizer, train_loader):
    # Repeat for given number of epochs
    num_batches = len(train_loader)
    optimizer.zero_grad()
    train_loss = 0.0
    for xb,yb in train_loader:
            # 1. Generate predictions
        pred = model(xb)
            #print(pred)
            # 2. Calculate loss
        loss = criterion(pred, yb)
            #print(loss)
            # 3. Compute gradients
        loss.backward()
            # 4. Update parameters using gradients
        optimizer.step()
            # 5. Reset the gradients to zero
        optimizer.zero_grad()
        train_loss += loss.item()
    train_loss = train_loss/num_batches
    return train_loss
    #trainingloss_history.append(train_loss)
        # Print the progress
    #if (epoch+1) % 10 == 0:
        #print('Epoch [{}/{}], Loss: {:.7f}'.format(epoch+1, num_epochs, loss.item()))

def testDNN(model, criterion, train_loader):
    # Repeat for given number of epochs
    num_batches = len(train_loader)
    test_loss = 0.0
    with torch.no_grad():
      for xb,yb in train_loader:
        # 1. Generate predictions
        pred = model(xb)
        #print(pred)
        # 2. Calculate loss
        loss = criterion(pred, yb)
        #print(loss)
        test_loss += loss.item()
      test_loss = test_loss/num_batches
    return test_loss

num_epochs = cfg.epochs
trainingloss_history = []
validloss_history = []
testloss_history = []
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_loss = trdnn(model, loss_fn, optimizer, train_loader)
    trainingloss_history.append(train_loss)
    valid_loss = testDNN(model, loss_fn, valid_loader)
    validloss_history.append(valid_loss)


test_loss = testDNN(model, loss_fn, TEST_loader)
testloss_history.append(test_loss)
print("Done!")

print(trainingloss_history)
plt.semilogy(trainingloss_history, label = 'training_data')
print(validloss_history)
plt.semilogy(validloss_history, label = 'validation_data')
print(testloss_history)
plt.semilogy(testloss_history,  label = 'test_data')
plt.legend()
plt.savefig('training_vs_validation.png')

#predictions = model(X_TEST)
#print(predictions)
#pred = predictions.detach().numpy()
#Y_TEST = Y_TEST.detach().numpy()
#print(Y_TEST)

#plt.scatter(np.arange(1,501), pred[:500], label ='predictions', linewidth = 1)
#plt.scatter(np.arange(1, 501), Y_TEST[:500], label = 'testData')
#plt.legend()

