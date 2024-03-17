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
    Conv Model 2:
        Layers:
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
            self.bn4 = nn.BatchNorm2d(256)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 10)
            self.dropout = nn.Dropout(0.2)
            self.batch_norm = nn.BatchNorm2d(32)
        Results:
            Run 1:
                Validation error: 
                    Accuracy: 65.3%, Avg loss: 0.992993

                Test error: 
                    Accuracy: 65.8%, Avg loss: 0.993094
                Final: 
                    Correctly guessed!