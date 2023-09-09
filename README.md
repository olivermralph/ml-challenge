# monolith-ai-challenge
Train a deep learning model on the fashion-MNIST dataset

## About
This code is used to train a model to classify grayscale images from the Fashion-MNIST dataset with an accuracy of 86% on the 10000 test images.

The last printout is as follows:
`[1,   100] loss: 0.887`
`[1,   200] loss: 0.481`
`[1,   300] loss: 0.439`
`[1,   400] loss: 0.394`
`[2,   100] loss: 0.320`
`[2,   200] loss: 0.309`
`[2,   300] loss: 0.310`
`[2,   400] loss: 0.304`
`Finished Training`
`Accuracy of the network on the 10000 test images: 86 %`
`Accuracy for class: T-shirt/top is 83.7 %`
`Accuracy for class: Trouser is 96.2 %`
`Accuracy for class: Pullover is 81.6 %`
`Accuracy for class: Dress is 86.1 %`
`Accuracy for class: Coat  is 77.4 %`
`Accuracy for class: Sandal is 95.1 %`
`Accuracy for class: Shirt is 62.8 %`
`Accuracy for class: Sneaker is 92.9 %`
`Accuracy for class: Bag   is 96.6 %`
`Accuracy for class: Ankle Boot is 96.1 %`

The test class accuracy is not always the same and could be due to the shuffling of the training dataset but could also just be the non-determinsm of the model.

There is currently no validation as I ran out of time to add this, but all other requirements have been met.

The time taken to complete this was approximately 6h 30m.

## How to Run
You will need to create a conda environement using the definition file that can be found under:
`"environments/deep_learning/conda_dependencies.yml"`.
Or alternatively install these libraries on a clean conda/pip environment if you can't use the conda dependencies file.

## Arguments

There is a config file called `config.json` under the `config` folder which contain parameters for running training and also paths to the data and output model locations (the data will be downloaded if it isn't already). You will need to call this when you run the training script `src/train.py`.

The model architecture is set as a separate variable to the config file as there is a limited set of choices, those currently being `"resnet18"`, `"resnet34"` and `"resnet50"`.

On the command line run the following with the working directory set as the project folder:
`python src/train.py --config_file="config/config.json" --model_architecture="resnet18"` 