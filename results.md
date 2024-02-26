With learning_rate = 1e-3, batch_size = 4, epochs = 50:
    Conv Model 1:
        Layers:
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        Results:
            Run 1:
                Validation error:
                    Accuracy: 43.7%, Avg loss: 1.581223
                Test error:
                    Accuracy: 43.4%, Avg loss: 1.590385
            Run 2 (Additional Transforms):
                Validation error: 
                    Accuracy: 39.8%, Avg loss: 1.669266
                Test error:
                    Accuracy: 39.4%, Avg loss: 1.678332