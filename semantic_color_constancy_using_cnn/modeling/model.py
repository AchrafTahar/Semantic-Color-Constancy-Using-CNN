import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from tqdm import tqdm

# Import the model
class TrueColorNet(nn.Module):
    def __init__(self):
        super(TrueColorNet, self).__init__()
        
        # Conv layers
        self.conv1 = nn.Conv2d(4, 96, kernel_size=11, stride=4, padding=0)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.fc7 = nn.Linear(4096, 2048)
        self.fc8 = nn.Linear(2048, 1024)
        self.fc9 = nn.Linear(1024, 4)  # r, g, b, gamma
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Normalization layers
        self.norm = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0)
        
    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.norm(x)
        x = torch.max_pool2d(x, kernel_size=3, stride=2)
        
        # Conv2
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.norm(x)
        x = torch.max_pool2d(x, kernel_size=3, stride=2)
        
        # Conv3-5
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.max_pool2d(x, kernel_size=3, stride=2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected with ReLU and dropout
        x = torch.relu(self.fc6(x))
        x = self.dropout(x)
        x = torch.relu(self.fc7(x))
        x = self.dropout(x)
        x = torch.relu(self.fc8(x))
        x = self.dropout(x)
        x = self.fc9(x)
        
        return x

# Configure logger
logger.add("model_test.log", format="{time} {level} {message}", level="DEBUG")

# Create a dummy dataset
def create_dummy_data(num_samples=100, input_size=(4, 224, 224)):
    logger.info(f"Creating dummy dataset with {num_samples} samples and input size {input_size}...")
    X = torch.randn(num_samples, *input_size)  # Random input data
    y = torch.randn(num_samples, 4)  # Random targets (r, g, b, gamma)
    logger.success("Dummy dataset created.")
    return X, y

# Test the model
def test_model():
    logger.info("Starting model test...")
    try:
        # Initialize model
        model = TrueColorNet()
        logger.info("Model initialized successfully.")

        # Dummy data
        X, y = create_dummy_data()
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        logger.info("Loss and optimizer defined.")

        # Training loop
        model.train()
        epochs = 2
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}...")
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training")):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                logger.debug(f"Batch {i + 1}: Loss = {loss.item():.4f}")

            logger.info(f"Epoch {epoch + 1} completed. Average loss: {running_loss / len(dataloader):.4f}")

        logger.success("Model test completed successfully.")
    except Exception as e:
        logger.exception(f"An error occurred during model testing: {e}")

# Run the test
if __name__ == "__main__":
    test_model()
