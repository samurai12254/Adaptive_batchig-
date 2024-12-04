import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target

X = X[y != 2]
y = y[y != 2]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

class SVM(nn.Module):
    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        return self.linear(x)

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, outputs, targets):
        targets = 2 * targets - 1
        return torch.mean(torch.clamp(1 - targets * outputs.squeeze(), min=0))
def train_model(optimizer_name, optimizer_fn):
    model = SVM(input_dim=X_train.shape[1])
    criterion = HingeLoss()
    optimizer = optimizer_fn(model.parameters())

    losses = []
    for epoch in range(20):
        total_loss = 0
        for batch in train_loader:
            batch_X, batch_y = batch

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        losses.append(total_loss)
        print(f"{optimizer_name} - Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    return model, losses

sgd_model, sgd_losses = train_model("SGD", lambda params: optim.SGD(params, lr=0.1, weight_decay=0.01))
adam_model, adam_losses = train_model("Adam", lambda params: optim.Adam(params, lr=0.01))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), sgd_losses, label='SGD', marker='o')
plt.plot(range(1, 21), adam_losses, label='Adam', marker='x')
plt.title('Loss Curve Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

def plot_decision_boundary(model, title):
    with torch.no_grad():
        w = model.linear.weight[0].numpy()
        b = 0
        x0, x1 = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 200),
                             np.linspace(X[:, 1].min(), X[:, 1].max(), 200))
        grid = np.c_[x0.ravel(), x1.ravel(), np.zeros_like(x0.ravel()), np.zeros_like(x0.ravel())]
        grid = torch.tensor(grid, dtype=torch.float32)
        decision_values = model(grid).reshape(x0.shape)

        plt.figure(figsize=(10, 6))
        plt.contourf(x0, x1, decision_values.numpy(),
                     levels=np.linspace(decision_values.min(), decision_values.max(), 50),
                     cmap='coolwarm', alpha=0.8)
        plt.colorbar(label='Decision Value')
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True)
        plt.show()
plot_decision_boundary(sgd_model, "SGD - Decision Boundary")
plot_decision_boundary(adam_model, "Adam - Decision Boundary")
