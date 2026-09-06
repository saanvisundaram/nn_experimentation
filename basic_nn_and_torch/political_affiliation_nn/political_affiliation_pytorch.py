import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class AllData(Dataset):
    def __init__(self, X, y):
        self.features = np.loadtxt(X, usecols=[0,1,2,3,4,5],
    delimiter=",", comments="#", dtype=np.float32)
        self.labels = np.loadtxt(y, usecols=6,
    delimiter=",", comments="#", dtype=np.int64)

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]

# Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, num_outputs)
        )

    def forward(self, x):
        return self.layers(x)


## Load data I haven't done this yet

torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3

for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):

        logits = model(features)

        loss = F.cross_entropy(logits, labels) # Loss function

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### LOGGING
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train/Val Loss: {loss:.2f}")

    model.eval()
    # Optional model evaluation
