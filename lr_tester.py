import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 2: Define a learning rate scheduler using OneCycleLR
def get_optimizer_and_scheduler(model, max_lr, num_epochs, momentum=0.9, weight_decay=1e-5):
    optimizer = optim.SGD(model.parameters(), lr=max_lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=5e-6,
        epochs=num_epochs,
        steps_per_epoch=1,
        div_factor=25,
        final_div_factor=10000,
        pct_start = 0.1
    )
    return optimizer, scheduler

# Step 3: Run "empty" epochs to upate the learning rate at each step
def run_empty_epochs(model, dataloader, optimizer, scheduler, num_epochs):
    lr_hist = []
    for epoch in range(num_epochs):
        # for inputs, targets in dataloader:
        #     outputs = model(inputs)
        #     loss = torch.mean(outputs)
        #     optimizer.zero_grad()
        #     loss.backward()
        optimizer.step()
        lr_hist.append(scheduler.get_last_lr())
        scheduler.step()
    return lr_hist

# Step 4: Plot the learning rate along the training process
def plot_learning_rate(lr_hist):
    print(f'{(lr_hist[0]) = },\n'
          f'{(lr_hist[-1]) = },\n'
          f'{max(lr_hist) = },\n'
          f'{min(lr_hist) = },\n'
          f'{np.mean(lr_hist) = }')
    plt.plot(lr_hist)
    # plt.yscale('log')
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.title('OneCycleLR Learning Rate Schedule')
    plt.show()

# Example usage:
if __name__ == '__main__':
    # Dummy dataset and dataloader
    dummy_data = torch.randn(100, 10)
    dummy_labels = torch.randn(100, 1)
    dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
    dummy_dataloader = torch.utils.data.DataLoader(dummy_dataset, batch_size=10)

    # Initialize the model and other components
    model = SimpleNN()
    max_lr = 0.1  # Set your desired max learning rate
    # total_steps = 1000  # Set the total number of training steps
    num_epochs = 300
    optimizer, scheduler = get_optimizer_and_scheduler(model, max_lr,num_epochs)

    # Run empty epochs to update the learning rate at each step
    lr_hist = run_empty_epochs(model, dummy_dataloader, optimizer, scheduler, num_epochs)

    # Plot the learning rate schedule
    plot_learning_rate(lr_hist)
