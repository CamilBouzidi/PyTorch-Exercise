import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Hyperparameters
learning_rate = 1e-3
batch_size = 512
epochs = 1000

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*32*32, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class NeuralNetworkManager:
    def __init__(self, device):
        self.device = device
    
    def training_loop(self, dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        # training mode
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)  # Move data to the correct device

            # prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (batch % 100 == 0):
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:7F} [{current:>5d}/{size:>5d}]")

    def testing_loop(self, dataloader, model, loss_fn):
        # evaluation mode
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        # Evaluate model (no gradients mode for memory usage purposes)
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)  # Move data to the correct device
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        test_loss /= num_batches
        correct /= size
        print(f"Test error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        
    # Script to run and optimize the model:
    def run_and_optimize(self):
        # exit if GPU not used
        if not torch.cuda.is_available():
            print("cuda not available")
            print(f"{torch.cuda.get_device_name(0)}")
            return
        
        print("cude available")
        print(f"{torch.cuda.get_device_name(0)}")
        # Generate training and test data
        training_data = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )

        testing_data = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )

        # use the dataloader class
        training_dataloader = DataLoader(training_data, batch_size=512)
        testing_dataloader = DataLoader(testing_data, batch_size=512)

        # Create the model using the selected device.
        model = NeuralNetwork().to(self.device)
        print(model)

        # Initialize the loss function
        loss_fn = nn.CrossEntropyLoss()

        # Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.training_loop(training_dataloader, model, loss_fn, optimizer)
            self.testing_loop(testing_dataloader, model, loss_fn)
        print("Done!")

        # Save the model
        torch.save(model.state_dict(), 'model_weights.pth')