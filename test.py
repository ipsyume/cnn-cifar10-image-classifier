import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

TRAIN_MODEL = True

# Transform: convert images to tensors + normalize
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# Load training and test datasets
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True,
    download=True, transform=transform_train
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True
)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False,
    download=True, transform=transform_test
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False
)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch.nn as nn
import torch.nn.functional as F

class BetterCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
import torch.optim as optim

net = BetterCNN()
if not TRAIN_MODEL:
    net.load_state_dict(torch.load("better_cnn_cifar10.pth"))
    print("Loaded saved model")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.5
)


# Train for 25 epochs (start small)
if TRAIN_MODEL:
    for epoch in range(25):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 200 == 199:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

        scheduler.step()

    print("Finished Training")
    torch.save(net.state_dict(), "better_cnn_cifar10.pth")
    print("Model saved as better_cnn_cifar10.pth")

net.eval()

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in testloader:
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
first_conv = net.conv_block1[0]

# Take one test image
image, label = testset[0]
image = image.unsqueeze(0)  # add batch dimension

with torch.no_grad():
    feature_maps = first_conv(image)

# Plot first 8 feature maps
fig, axes = plt.subplots(1, 8, figsize=(20, 5))
for i in range(8):
    axes[i].imshow(feature_maps[0, i].cpu(), cmap="gray")
    axes[i].axis("off")

plt.suptitle("First Conv Layer Feature Maps")
plt.show()
net.eval()

def show_predictions(model, dataset, classes, num_images=6):
    plt.figure(figsize=(15, 5))

    for i in range(num_images):
        image, label = dataset[i]

        with torch.no_grad():
            output = model(image.unsqueeze(0))
            _, predicted = torch.max(output, 1)

        # Unnormalize image
        img = image.permute(1, 2, 0).cpu()
        img = img * 0.5 + 0.5

        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title(
            f"Predicted: {classes[predicted.item()]}\n"
            f"Actual: {classes[label]}"
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("sample_predictions.png")
    plt.show()


show_predictions(net, testset, classes, num_images=6)