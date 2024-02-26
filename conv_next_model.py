from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models
import torchvision.transforms as transforms
import torch.nn.functional as F
import lightning.pytorch as pyLight

# Here, we use Lightning to implement ConvNeXt_Small(https://pytorch.org/vision/main/models/generated/torchvision.models.convnext_small.html)
# Hyperparameters
learning_rate = 1e-3
batch_size = 4
epochs = 50

class ConvNextModel(pyLight.LightningModule):
    def __init__(self, num_classes = 10) -> None:
        super().__init__()
        # Use the widespread IMAGENET1K_V1
        self.model = models.convnext_small(weights="IMAGENET1K_V1")
        # Use a linear classifier
        self.model.classifier = nn.Linear(in_features=6 * 6 * 256, out_features=num_classes)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss(outputs, labels)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

class ConvNextManager():
    def __init__(self, device):
        self.device = device

    def run_and_optimize(self):
        # gpu management is done by Lightning

        #use same transforms as my own conv_model
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

        # Initialize the model
        model = ConvNextModel()

        # Initialize the Trainer, will use the GPU
        trainer = pyLight.Trainer(max_epochs=epochs, accelerator="gpu")

        trainer.fit(model, training_dataloader, validation_dataloader)
        trainer.test(dataloaders=testing_dataloader)
        torch.save(model.state_dict(), f'convnext_model_weights.pth')