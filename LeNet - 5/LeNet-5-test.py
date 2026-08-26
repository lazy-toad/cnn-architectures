import torch
import torch.nn as nn
import torch.optim as optim
from LeNet_5 import LeNet5
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(batch_size=128, data_dir="./data"):

    transform = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # clear gradients from the last step
        outputs = model(images)
        loss = criterion(outputs, labels)  # CrossEntropyLoss applies softmax internally
        loss.backward()
        optimizer.step()  # update every weight/bias using those gradients

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()  # no gradients needed during evaluation. saves memory/compute
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)  # pick the class with highest logit
        correct += (predictions == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(batch_size=128)

    model = LeNet5(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch:2d}/{num_epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"test loss: {test_loss:.4f} | "
            f"test acc: {test_acc * 100:.2f}%"
        )

    torch.save(model.state_dict(), "lenet5_mnist.pt")
    print("Saved trained weights to lenet5_mnist.pt")


if __name__ == "__main__":
    main()
