import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
if __name__ == "__main__":
        
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')

    # Define dataset loading function
    def load_data():
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor()
        ])

        train_dataset = datasets.ImageFolder(root='dataset/train', transform=transform)
        test_dataset = datasets.ImageFolder(root='dataset/test', transform=transform)

        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=25, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=25, shuffle=False, num_workers=2)

        return train_loader, val_loader, test_loader

    # Load dataset
    train_loader, val_loader, test_loader = load_data()

    # Define model
    class VGGModel(nn.Module):
        def __init__(self):
            super(VGGModel, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * 6 * 6, 4096),
                nn.Dropout(0.2),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.Dropout(0.2),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 7)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = VGGModel().to(device)
    print(model)

    # Define loss function and optimizer
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0.0001)
    scaler = GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5, verbose=True)

    # Training loop
    def train_model(model, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            model.train()
            train_loss, train_correct = 0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                train_correct += (outputs.argmax(dim=1) == labels).sum().item()

            val_loss, val_correct = 0, 0
            model.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    val_correct += (outputs.argmax(dim=1) == labels).sum().item()

            print(f"Epoch {epoch+1}/{epochs}, "
                f"Train Loss: {train_loss / len(train_loader.dataset):.4f}, "
                f"Train Accuracy: {train_correct / len(train_loader.dataset):.4f}, "
                f"Val Loss: {val_loss / len(val_loader.dataset):.4f}, "
                f"Val Accuracy: {val_correct / len(val_loader.dataset):.4f}")

    train_model(model, train_loader, val_loader, epochs=10)

    # Evaluate and plot confusion matrix
    def evaluate_model(model, test_loader):
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(outputs.argmax(dim=1).cpu().numpy())

        cm = confusion_matrix(y_true, y_pred)
        cmd = ConfusionMatrixDisplay(cm, display_labels=['Angry', 'Disgusted', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised'])
        cmd.plot()
        plt.show()

    evaluate_model(model, test_loader)
