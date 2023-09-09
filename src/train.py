import argparse
import json
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet34, resnet50
from torch import nn
import torch.optim as optim

CLASSES = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress',
           'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

MODEL_ARCHITECTURES = {
    "resnet18": resnet18(weights=None),
    "resnet34": resnet34(weights=None),
    "resnet50": resnet50(weights=None)
}

def get_transform():
    return transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))])


def set_dataset(data_path, train, download):
    transform = get_transform()
    return torchvision.datasets.FashionMNIST(root=data_path, train=train,
                                        download=download, transform=transform)
    

def set_dataloader(dataset, batch_size, shuffle, num_workers, drop_last):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, 
                                          shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)

def set_model(model_architecture):
    model = MODEL_ARCHITECTURES[model_architecture]

    # Convert model to grayscale
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Update the fully connected layer based on the number of classes in the dataset
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))

    return model

def train_model(epochs):
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')


def save_model(model_path):
    torch.save(model.state_dict(), model_path)


def test_model():
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model_to_test(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

def calculate_accuracy_per_class():
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in CLASSES}
    total_pred = {classname: 0 for classname in CLASSES}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model_to_test(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[CLASSES[label]] += 1
                total_pred[CLASSES[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for model architecture and config file for remaining arguments")

    parser.add_argument("--config_file",
                        required=True,
                        help="Location of config file to parse reamining arguments")
    parser.add_argument("--model_architecture",
                        default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50"],
                        help="Select model architecture for model training (dsefault='resnet18')")
    args = parser.parse_args()
 
    f = open(args.config_file)
    
    contents = json.load(f)

    batch_size = int(contents["params"]["batch_size"])
    epochs = int(contents["params"]["epochs"])
    model_architecture = args.model_architecture
    data_path = contents["paths"]["data_path"]
    model_path = contents["paths"]["model_path"]

    torch.manual_seed(0)
    trainset = set_dataset(data_path, train=True, download=True)
    trainloader = set_dataloader(trainset, batch_size, 
                                            shuffle=True, num_workers=2, drop_last=True)

    testset = set_dataset(data_path, train=False, download=True)
    testloader = set_dataloader(testset, batch_size,
                                            shuffle=False, num_workers=2, drop_last=True)

    model = set_model(model_architecture)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_model(epochs)

    save_model(model_path)

    model_to_test = set_model(model_architecture)
    model_to_test.load_state_dict(torch.load(model_path))

    test_model()
    calculate_accuracy_per_class()



