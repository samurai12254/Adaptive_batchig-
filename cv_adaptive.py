import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

batch_size = 2
learning_rate = 0.001
num_epochs = 15
num_classes = 10

device = torch.device("cuda")

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train(model, loader, criterion, optimizer, device, losses):
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # loader.step([x for x in model.parameters()])
        optimizer.step()
        losses.append(loss.item())

        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    return test_loss / len(loader), accuracy

losses = []
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device, losses)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    # losses.append(train_loss);
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

torch.save(model.state_dict(), "cifar10_resnet18.pth")
np.save('losses_good.npy', np.array(losses))

import scipy

losses_bad = np.load('losses_bad.npy')
losses_optim = np.load('losses_optim.npy')
losses_bad_sgd = np.load('losses_SGD.npy')
losses_optim_sgd = np.load('losses_optim_SGD.npy')

iters = np.arange(len(losses_bad))
iters_optim = np.arange(len(losses_optim))
iters_bad_sgd = np.arange(len(losses_bad_sgd))
iters_optim_sgd = np.arange(len(losses_optim_sgd))

plt.plot(iters, scipy.signal.medfilt(losses_bad, 513), label='Loss Adam')
plt.plot(iters_optim, scipy.signal.medfilt(losses_optim, 513), label='Loss with optimizer Adam')
plt.plot(iters_bad_sgd, scipy.signal.medfilt(losses_bad_sgd, 513), label='Loss SGD')
plt.plot(iters_optim_sgd, scipy.signal.medfilt(losses_optim_sgd, 513), label='Loss with optimizer SGD')

plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.show()