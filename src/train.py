import argparse
import json
import os
import torch
import torchvision.transforms as transforms
import sys

from sklearn.model_selection import train_test_split
from torchvision.models import resnet18, resnet34, resnet50
from torch import nn
import torch.optim as optim

from dataset import CustomFashionMNISTDataset

# Set path to import dataset
sys.path.append("/home/oralph/repos/fashion-mnist")

from utils import mnist_reader

CLASSES = ("T-shirt/top", "Trouser", "Pullover", "Dress",
           "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot")

MODEL_ARCHITECTURES = {
    "resnet18": resnet18(weights=None),
    "resnet34": resnet34(weights=None),
    "resnet50": resnet50(weights=None)
}

IMAGE_SHAPE = (28, 28)

TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ]
)

COMPUTE_TYPES = {
    "gpu": "cuda:0",
    "cpu": "cpu"
}


def set_device(device):
    return torch.device(
        COMPUTE_TYPES[device] if torch.cuda.is_available()
        else COMPUTE_TYPES["cpu"]
    )


def load_fashion_mnist_data(data_path):
    X, y = mnist_reader.load_mnist(data_path, kind="train")
    X_test, y_test = mnist_reader.load_mnist(data_path, kind="t10k")

    X_copy, y_copy = X.copy(), y.copy()
    X_test_copy, y_test_copy = X_test.copy(), y_test.copy()

    return X_copy, y_copy, X_test_copy, y_test_copy


def set_dataset(images, labels):
    return CustomFashionMNISTDataset(
        images=images,
        labels=labels,
        image_shape=IMAGE_SHAPE,
        image_transform=TRANSFORM
    )


def set_dataloader(dataset, batch_size, shuffle):
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


def set_model(model_architecture):
    model = MODEL_ARCHITECTURES[model_architecture].to(device)

    # Convert model to grayscale
    model.conv1 = torch.nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    ).to(device)

    # Update the fully connected layer based on the number of classes
    # in the dataset
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES)).to(device)

    return model


def train_model(
        model,
        optimizer,
        criterion,
        epochs,
        train_dataloader,
        val_dataloader
):
    for epoch in range(epochs):  # loop over the dataset multiple times

        # begin training
        print(f"Training - Epoch {epoch + 1}")
        model.train()

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 batches
                print(
                    f"[Epoch: {epoch + 1}, Batch: {i + 1:3d}]\
                      loss: {running_loss / 100:.3f}"
                )
                running_loss = 0.0

        print(f"Validation - Epoch {epoch + 1}")
        test_model(model, val_dataloader)
        calculate_accuracy_per_class(model, val_dataloader)

    print("Finished Training")


def save_model(model_path):
    torch.save(model.state_dict(), model_path)


def test_model(model, dataloader):
    # begin model evaluation
    model.eval()

    correct = 0
    total = 0
    # since we're not training, we don't need to \
    # calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Accuracy of the network on the dataset: \
          {100 * correct // total} %"
    )


def calculate_accuracy_per_class(model, dataloader):
    # begin model evaluation
    model.eval()

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in CLASSES}
    total_pred = {classname: 0 for classname in CLASSES}

    # again no gradients needed
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[CLASSES[label]] += 1
                total_pred[CLASSES[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser for model architecture \
            and config file for remaining arguments"
    )

    parser.add_argument(
        "--config_file",
        required=True,
        help="Location of config file to parse reamining arguments"
    )
    parser.add_argument(
        "--model_architecture",
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50"],
        help="Model architecture for model training (default='resnet18')"
    )
    parser.add_argument(
        "--compute_type",
        default="gpu",
        choices=["gpu", "cpu"],
        help="Train model on 'gpu' or 'cpu' (default='gpu')"
    )
    args = parser.parse_args()

    f = open(args.config_file)

    contents = json.load(f)

    batch_size = int(contents["params"]["batch_size"])
    epochs = int(contents["params"]["epochs"])
    model_architecture = args.model_architecture
    data_path = contents["paths"]["data_path"]
    model_path = contents["paths"]["model_path"]

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    device = set_device(args.compute_type)
    print("device:", device)

    X, y, X_test, y_test = load_fashion_mnist_data(data_path)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=0
    )

    torch.manual_seed(0)
    train_dataset = set_dataset(images=X, labels=y)
    train_dataloader = set_dataloader(train_dataset, batch_size, shuffle=True)

    val_dataset = set_dataset(images=X_val, labels=y_val)
    val_dataloader = set_dataloader(val_dataset, batch_size, shuffle=False)

    test_dataset = set_dataset(images=X_test, labels=y_test)
    test_dataloader = set_dataloader(test_dataset, batch_size, shuffle=False)

    model = set_model(model_architecture)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_model(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )

    save_model(model_path)

    model_to_test = set_model(model_architecture)
    model_to_test.load_state_dict(torch.load(model_path))

    print("Testing")
    test_model(model=model_to_test, dataloader=test_dataloader)
    calculate_accuracy_per_class(
        model=model_to_test,
        dataloader=test_dataloader
    )
