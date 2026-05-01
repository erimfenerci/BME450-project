import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import os
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, "train")
test_path  = os.path.join(script_dir, "test")

training_data = datasets.ImageFolder(root=train_path, transform=transform)
test_data     = datasets.ImageFolder(root=test_path,  transform=transform)

categories  = training_data.classes
num_classes = len(categories)

print("=" * 50)
print("DATASET LOADED!")
print("=" * 50)
print(f"Classes: {categories}")
print(f"Number of classes: {num_classes}")
print(f"Training images: {len(training_data)}")
print(f"Test images: {len(test_data)}")
print()

sample_num = 0
print('Inputs sample - image size:', training_data[sample_num][0].shape)
print('Label:', training_data[sample_num][1], '\n')

ima = training_data[sample_num][0]
print('Inputs sample - min,max,mean,std:', ima.min().item(), ima.max().item(), ima.mean().item(), ima.std().item())
ima = (ima - ima.mean()) / ima.std()
print('Inputs sample normalized - min,max,mean,std:', ima.min().item(), ima.max().item(), ima.mean().item(), ima.std().item())
iman = ima.permute(1, 2, 0)
plt.imshow(iman)
plt.show()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 8 * 8, 512)
        self.fc2   = nn.Linear(512, 256)
        self.fc3   = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        output = self.fc3(x)
        return output


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return 100 * correct / size


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100 * correct


model = Net()

batch_size = 32
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader  = DataLoader(test_data,     batch_size=batch_size, shuffle=False)

loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

epochs = 10
train_accuracies = []
test_accuracies  = []

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_acc = train_loop(train_dataloader, model, loss_fn, optimizer)
    test_acc  = test_loop(test_dataloader, model, loss_fn)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
print("Done!")

plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), train_accuracies, 'b-o', label='Train Accuracy', markersize=4)
plt.plot(range(1, epochs+1), test_accuracies,  'r-o', label='Test Accuracy',  markersize=4)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Epoch')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


model.eval()
num_per_class = 10
num_show = num_per_class * num_classes
fig, axes = plt.subplots(num_classes, num_per_class, figsize=(20, 8))

with torch.no_grad():
    for class_idx in range(num_classes):
        class_images = [i for i, (_, label) in enumerate(test_data) if label == class_idx]
        for col_idx, img_idx in enumerate(class_images[:num_per_class]):
            img, true_label = test_data[img_idx]
            output = model(img.unsqueeze(0))
            pred_label = torch.argmax(output).item()

            img_display = img.permute(1, 2, 0) * 0.5 + 0.5
            img_display = img_display.clamp(0, 1)

            correct = pred_label == true_label
            color = 'green' if correct else 'red'

            axes[class_idx][col_idx].imshow(img_display, cmap='gray')
            axes[class_idx][col_idx].set_title(
                f"True: {categories[true_label]}\nPred: {categories[pred_label]}\n{'✓' if correct else '✗'}",
                color=color, fontsize=7
            )
            axes[class_idx][col_idx].axis('off')

plt.suptitle("Test Results - Green = Correct, Red = Wrong", fontsize=12)
plt.tight_layout()
plt.show()