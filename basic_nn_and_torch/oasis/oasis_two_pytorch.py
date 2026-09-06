import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class AllData(Dataset):
    @staticmethod	
    def vec_to_onehot(data_y, n):
        # convert ordinal (0,1,2 . .) to one-hot
        rows = len(data_y)
        cols = n
        result = np.zeros((rows,cols), dtype=np.float32)
        for i in range(rows):
            k = data_y[i]   # 0,1,2 . .
            result[i][k] = 1.0;  # [ 0.0  1.0  0.0]
        return result;

    def __init__(self, in_file):
        x = np.loadtxt(in_file, usecols=[2,3,4,5, 6],
                             delimiter=",", comments="#", dtype=np.float32)
        y = np.loadtxt(in_file, usecols=[0],
                             delimiter=",", comments="#", dtype=np.float32)

        # self.labels = np.zeros((len(y), 1), dtype=np.float32)
        # for i in range(len(y)): 
        #    if y[i] == 0:
        #        self.labels[i] = [0, 1]
        #    else:
        #        self.labels[i] = [1, 0]
        
        self.features = x
        # self.labels =  AllData.vec_to_onehot(y, 2)
        self.labels = y.reshape(len(y), 1)
        # input("initialized, press enter to continue")
        # print(self.labels)

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        # print(one_x, one_y)
        # input("--> ")
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]

# Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 10),
            nn.ReLU(),
            # nn.Linear(5, 3),
            # nn.ReLU(),
            nn.Linear(10, num_outputs)
        )

    def forward(self, x):
        return self.layers(x)
    
train_file = "oasis_train.txt"
test_file = "oasis_test.txt"

train_ds = AllData(train_file)
test_ds = AllData(test_file)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


def compute_accuracy(model, dataloader):

    model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):

        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        target_index = torch.argmax(labels, dim=1)

        # print(labels)
        # print("Pred", predictions)
        compare = target_index == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()

def compute_accuracy_bce(model, dataloader):

    model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):

        with torch.no_grad():
            logits = model(features)

        predictions = (logits >= 0.0).float()
        # target_index = torch.argmax(labels, dim=1)
        # input(f"Logits: {logits}, Pred: {predictions}, Labels: {labels}")


        # print(labels)
        # print("Pred", predictions)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()

torch.manual_seed(123)
model = NeuralNetwork(num_inputs=5, num_outputs=1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

num_epochs = 50
print(model)
print(model.parameters())
input("-->")

for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):

        logits = model(features)

        # loss = F.cross_entropy(logits, labels) 
        loss = F.binary_cross_entropy_with_logits(logits, labels) # Loss function

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### LOGGING
        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train/Val Loss: {loss:.2f}")

# model.eval()
# Optional model evaluation
print("computing accuracy on test set")
# print(compute_accuracy(model, test_loader))
print(compute_accuracy_bce(model, test_loader))