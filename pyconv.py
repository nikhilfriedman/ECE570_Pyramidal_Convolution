import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

class PyramidalConv(nn.Module):
    def __init__(self, in_channels, out_channels, scales=[1, 2, 4]):
        super(PyramidalConv, self).__init__()
        self.scales = scales
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) for _ in scales
        ])

    def forward(self, x):
        feature_pyramids = []
        for scale, conv in zip(self.scales, self.convs):
            scaled_x = F.interpolate(x, scale_factor=1/scale, mode='bilinear', align_corners=False)
            scaled_feature = conv(scaled_x)
            upscaled_feature = F.interpolate(scaled_feature, size=x.shape[2:], mode='bilinear', align_corners=False)
            feature_pyramids.append(upscaled_feature)
        return torch.cat(feature_pyramids, dim=1)

class SimplePyramidalCNN(nn.Module):
    def __init__(self):
        super(SimplePyramidalCNN, self).__init__()
        self.pyramid_conv1 = PyramidalConv(3, 16, scales=[1, 2, 4])
        self.conv2 = nn.Conv2d(16 * 3, 32, kernel_size=3, padding=1)  # 16 * 3 channels from pyramidal conv
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pyramid_conv1(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 99:
            print(f'Epoch {epoch+1}, Batch {batch_idx+1}: Loss = {running_loss / 100:.4f}')
            running_loss = 0.0

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss = {test_loss:.4f}, Accuracy = {accuracy:.2f}%')
    return accuracy

def main():
    # Hyperparameters and configurations
    batch_size = 64
    epochs = 10
    learning_rate = 0.001

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CIFAR-10 dataset and DataLoader
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize model, criterion, and optimizer
    model = SimplePyramidalCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and testing loop
    best_accuracy = 0
    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, criterion, epoch)
        accuracy = test(model, device, test_loader, criterion)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

    print("Training complete. Best accuracy:", best_accuracy)

if __name__ == "__main__":
    main()