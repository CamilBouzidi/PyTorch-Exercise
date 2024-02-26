import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F

# Here, we implement a model that has different layers and optimizer.
# It's a Convolutional Neural Network, we thus expect a better accuracy than the linear Neural Network
# implemented in model.py

# Hyperparameters
learning_rate = 1e-3
batch_size = 4
epochs = 50

class ConvNeuralNetwork(nn.Module):
    def __init__(self, version = 1):
        super().__init__()
        self.layers_version = version
        # Each image has 3 color channels
        if self.layers_version == 1:
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
        else:
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(128 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 10)
            self.dropout = nn.Dropout(0.2)
            self.batch_norm = nn.BatchNorm2d(32)

    def forward(self, x):
        if self.layers_version == 1:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = self.pool(F.relu(self.batch_norm(self.conv1(x))))
            x = self.pool(F.relu(self.batch_norm(self.conv2(x))))
            x = self.pool(F.relu(self.batch_norm(self.conv3(x))))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.dropout(F.relu(self.fc2(x)))
            x = self.fc3(x)
        return x


class ConvNeuralNetworkManager:
    def __init__(self, device):
        self.device = device
    
    def training_loop(self, dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        # training mode
        model.train()

        # let's iterate over the dataset twice
        for epoch in range(2):
            # keep track of the running loss
            running_loss = 0.0
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)  # Move data to the correct device

                # zero parameter gradients
                optimizer.zero_grad()
                
                # prediction and loss
                pred = model(X)
                loss = loss_fn(pred, y)

                # backpropagation without zeroing the parameter gradients
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if batch % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {batch + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
        print(f"Finished training across {epoch+1} epochs.")
    

    def validation_loop(self, dataloader, model, loss_fn):
        # evaluation mode
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        val_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)  # Move data to the correct device
                pred = model(X)
                val_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        val_loss /= num_batches
        correct /= size
        print(f"Validation error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")

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

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
        
        # Generate training, validation and test data
        training_data = datasets.CIFAR10(
            root='./data', 
            train=True,
            download=True,
            transform=transform
        )

        testing_data = datasets.CIFAR10(
            root='./data', 
            train=False,
            download=True,
            transform=transform
        )

        # Use a 80/20 split for training/validation
        training_data_size = int( 0.8 * len(training_data))
        validation_data_size = len(training_data) - training_data_size
        training_data, validation_data = random_split(training_data, [training_data_size, validation_data_size])

        # use the dataloader for training, validation and testing data
        training_dataloader = DataLoader(
            training_data,
            batch_size=batch_size,
            shuffle=True,
        )

        validation_dataloader = DataLoader(
            validation_data,
            batch_size=batch_size,
            shuffle=True,
        )

        testing_dataloader = DataLoader(
            testing_data,
            batch_size=batch_size,
            shuffle=False,
        )

        classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # Create the model using the selected device.
        model = ConvNeuralNetwork().to(self.device)
        print(model)

        # Initialize the loss function
        loss_fn = nn.CrossEntropyLoss()

        # Optimizer with momentum
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.training_loop(training_dataloader, model, loss_fn, optimizer)
            self.validation_loop(validation_dataloader, model, loss_fn)
            self.testing_loop(testing_dataloader, model, loss_fn)
        print("Done!")

        # Save the model
        torch.save(model.state_dict(), f"conv_model_v{model.layers_version}_weights.pth")