import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from model import NeuralNetwork, NeuralNetworkManager
from conv_model import ConvNeuralNetwork, ConvNeuralNetworkManager
import torchvision.datasets as datasets

retrain_model = True
use_convNN = True

device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
            )
print(f"Using {device} device")


def use_linear_model():
    # Current accuracy with model.py: 48.3%
    # Hyperparameters
    # learning_rate = 1e-3
    # batch_size = 512
    # epochs = 1000

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
    img = Image.open('C:/GitHub/PyTorch-Exercise/input3.jpg')

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

def use_conv_model():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 4
    if retrain_model:
        manager = ConvNeuralNetworkManager(device)
        manager.run_and_optimize()
    
    # Load the trained model
    model = ConvNeuralNetwork()
    state_dict = torch.load('C:/GitHub/PyTorch-Exercise/conv_model_weights.pth')
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()

    # Create the input
    # Load the image
    img = Image.open('C:/GitHub/PyTorch-Exercise/input3.jpg')

    # Resize the image to 32x32 pixels
    img = img.resize((32, 32))
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


if use_convNN:
    use_conv_model()
else:
    use_linear_model()
    