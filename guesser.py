import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from model import NeuralNetwork
import torchvision.datasets as datasets

# Load the trained model
model = NeuralNetwork()
state_dict = torch.load('C:/GitHub/PyTorch-Exercise/model_weights.pth')
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Create the input
# Load the image
img = Image.open('C:/GitHub/PyTorch-Exercise/input2.jpg')

# Resize the image to 32x32 pixels
img = img.resize((32, 32))

# Convert the image to a PyTorch tensor
transform = transforms.Compose([
    transforms.ToTensor(),
])
input_tensor = transform(img)
input_tensor = input_tensor.view(1,-1)
print(input_tensor.size())

# Pass the input tensor to the model
output = model(input_tensor)

# Get the predicted class
_, predicted = torch.max(output.data, 1)

# Load the CIFAR-10 dataset
cifar10_dataset = datasets.CIFAR10(root='./data', train=True, download=True)

# Get the list of class labels
class_labels = cifar10_dataset.classes

# Now you can get the label of the predicted class
predicted_label = class_labels[predicted.item()]

print(predicted_label)