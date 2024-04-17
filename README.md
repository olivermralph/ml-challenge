# ml-challenge
Train a deep learning model on the fashion-MNIST dataset

## About
This code is used to train a model to classify grayscale images from the Fashion-MNIST dataset with an accuracy of 86% on the 10000 test images.

The best model that I have trained so far had the following parameters:  
`"batch_size": 128`  
`"epochs": 10`  
`"model_architecture": "resnet18"`  
`"compute_type": "gpu"`  

The printout is as follows:  
`device: cuda:0`  
`[1,   100] loss: 0.971`  
`[1,   200] loss: 0.523`  
`[1,   300] loss: 0.466`  
`[1,   400] loss: 0.435`  
`[2,   100] loss: 0.355`  
`[2,   200] loss: 0.341`  
`[2,   300] loss: 0.350`  
`[2,   400] loss: 0.339`  
`[3,   100] loss: 0.282`  
`[3,   200] loss: 0.274`  
`[3,   300] loss: 0.283`  
`[3,   400] loss: 0.281`  
`[4,   100] loss: 0.232`  
`[4,   200] loss: 0.236`  
`[4,   300] loss: 0.240`  
`[4,   400] loss: 0.238`  
`[5,   100] loss: 0.192`  
`[5,   200] loss: 0.198`  
`[5,   300] loss: 0.205`  
`[5,   400] loss: 0.214`  
`[6,   100] loss: 0.165`  
`[6,   200] loss: 0.174`  
`[6,   300] loss: 0.181`  
`[6,   400] loss: 0.181`  
`[7,   100] loss: 0.140`  
`[7,   200] loss: 0.144`  
`[7,   300] loss: 0.145`  
`[7,   400] loss: 0.159`  
`[8,   100] loss: 0.117`  
`[8,   200] loss: 0.124`  
`[8,   300] loss: 0.135`  
`[8,   400] loss: 0.142`  
`[9,   100] loss: 0.105`  
`[9,   200] loss: 0.106`  
`[9,   300] loss: 0.116`  
`[9,   400] loss: 0.126`  
`[10,   100] loss: 0.097`  
`[10,   200] loss: 0.090`  
`[10,   300] loss: 0.101`  
`[10,   400] loss: 0.112`  
`Finished Training`  
`Accuracy of the network on the 10000 test images: 88 %`  
`Accuracy for class: T-shirt/top is 81.8 %`  
`Accuracy for class: Trouser is 97.4 %`  
`Accuracy for class: Pullover is 80.1 %`  
`Accuracy for class: Dress is 87.8 %`  
`Accuracy for class: Coat  is 83.8 %`  
`Accuracy for class: Sandal is 96.1 %`  
`Accuracy for class: Shirt is 68.1 %`  
`Accuracy for class: Sneaker is 93.9 %`  
`Accuracy for class: Bag   is 97.4 %`  
`Accuracy for class: Ankle Boot is 95.0 %`  

The test class accuracy is not always the same and could be due to the shuffling of the training dataset but could also just be the non-determinsm of the model.

There is currently no validation as I ran out of time to add this, but all other requirements have been met.

The time taken to complete this was approximately 6h 30m.

## How to Run
You will need to create a conda environement using the definition file that can be found under:
`"environments/deep_learning/conda_dependencies.yml"`.
Or alternatively install these libraries on a clean conda/pip environment if you can't use the conda dependencies file.

## Arguments

There is a config file called `config.json` under the `config` folder which contain parameters for running training and also paths to the data and output model locations (the data will be downloaded if it isn't already). You will need to call this when you run the training script `src/train.py`.

The model architecture is set as a variable in the command line, rather than in the config file, as there is a limited set of choices, those currently being `"resnet18"`, `"resnet34"` and `"resnet50"`.
The compute type can be set to `"gpu"` (code will print `cuda:0` if available) or `"cpu"` (code will print `cpu`) in the command line too.

On the command line, run the following with the working directory set as the project folder:  
`python src/train.py --config_file="config/config.json" --model_architecture="resnet18" --compute_type="gpu"`