import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from model import NeuralNetwork, NeuralNetworkManager
import torchvision.datasets as datasets

# Current accuracy 48.3%
# Hyperparameters
# learning_rate = 1e-3
# batch_size = 512
# epochs = 1000
retrain_model = False

device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
            )
print(f"Using {device} device")



# Optional: Retrain the model
if retrain_model:
    manager = NeuralNetworkManager(device)
    manager.run_and_optimize()


# Load the trained model
model = NeuralNetwork()
state_dict = torch.load('C:/GitHub/PyTorch-Exercise/model_weights.pth')
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Create the input
# Load the image
img = Image.open('C:/GitHub/PyTorch-Exercise/horse.jpg')

# Resize the image to 32x32 pixels
img = img.resize((32, 32))

# Convert the image to a PyTorch tensor
transform = transforms.Compose([
    transforms.ToTensor(),
])

# unsqueeze needed because of batch size 1
input_tensor = transform(img).unsqueeze(0)
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