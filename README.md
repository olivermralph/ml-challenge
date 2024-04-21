# ml-challenge
Train a deep learning model on the fashion-MNIST dataset.  
The repository that contains the fashion-mnist dataset required to run this code is subject to the following coptyright licence:  
The MIT License (MIT) Copyright © 2017 Zalando SE, https://tech.zalando.com

## About
This code is used to train a model to classify grayscale images from the Fashion-MNIST dataset with an accuracy of 86% on the 10000 test images.

Here is an example of the output of a model being trained from the python script `train.py`:  
`"batch_size": 128`  
`"epochs": 10`  
`"model_architecture": "resnet34"`  
`"compute_type": "gpu"`  

`The printout is as follows:`   
`device: cuda:0`  
`Training - Epoch 1`  
`[1,   100] loss: 0.948`  
`[1,   200] loss: 0.520`  
`[1,   300] loss: 0.470`  
`[1,   400] loss: 0.430`  
`Validation - Epoch 1`  
`Accuracy of the network on the dataset: 87 %`  
`Accuracy for class: T-shirt/top is 84.4 %`  
`Accuracy for class: Trouser is 96.6 %`  
`Accuracy for class: Pullover is 76.6 %`  
`Accuracy for class: Dress is 89.6 %`  
`Accuracy for class: Coat  is 83.6 %`  
`Accuracy for class: Sandal is 93.5 %`  
`Accuracy for class: Shirt is 64.0 %`  
`Accuracy for class: Sneaker is 90.0 %`  
`Accuracy for class: Bag   is 97.8 %`  
`Accuracy for class: Ankle Boot is 97.0 %`  
`Training - Epoch 2`  
`[2,   100] loss: 0.357`  
`[2,   200] loss: 0.341`  
`[2,   300] loss: 0.341`  
`[2,   400] loss: 0.341`  
`Validation - Epoch 2`  
`Accuracy of the network on the dataset: 89 %`  
`Accuracy for class: T-shirt/top is 86.8 %`  
`Accuracy for class: Trouser is 97.7 %`  
`Accuracy for class: Pullover is 80.1 %`  
`Accuracy for class: Dress is 88.3 %`  
`Accuracy for class: Coat  is 93.4 %`  
`Accuracy for class: Sandal is 98.0 %`  
`Accuracy for class: Shirt is 65.3 %`  
`Accuracy for class: Sneaker is 94.8 %`  
`Accuracy for class: Bag   is 98.2 %`  
`Accuracy for class: Ankle Boot is 96.3 %`  
`Finished Training`  
`Testing`  
`Accuracy of the network on the dataset: 86 %`  
`Accuracy for class: T-shirt/top is 83.7 %`  
`Accuracy for class: Trouser is 97.0 %`  
`Accuracy for class: Pullover is 74.1 %`  
`Accuracy for class: Dress is 83.7 %`  
`Accuracy for class: Coat  is 88.7 %`  
`Accuracy for class: Sandal is 96.5 %`  
`Accuracy for class: Shirt is 55.2 %`  
`Accuracy for class: Sneaker is 94.2 %`  
`Accuracy for class: Bag   is 96.9 %`  
`Accuracy for class: Ankle Boot is 94.4 %`  

The test class accuracy is not always the same and could be due to the shuffling of the training dataset but could also just be the non-determinsm of the model.

## How to Run
You will need to create a conda environement using the definition file that can be found under:
`"environments/deep_learning/conda_dependencies.yml"`.
Or alternatively install these libraries on a clean conda/pip environment if you can't use the conda dependencies file.

You will then need to clone the following repository to access the Fashion-MNIST dataset:  
https://github.com/zalandoresearch/fashion-mnist  
This should be adjacent to this repository.

In order to access the data from the cloned repo, you need to make sure that the path that is appended to the system (cell 4 in model_train.ipynb and line 15 in train.py) is that of the fashion-mnist repo location,  
e.g. `'~/repos/fashion-mnist'`

### Arguments

There is a config file called `config.json` under the `config` folder which contain parameters for running training and also paths to the data and output model locations. You will need to call this when you run the training script `src/train.py`.

An example of the config file looks like this:

`{`  
`    "params":{`  
`        "batch_size": 128,`  
`        "epochs": 2`  
`    },`  
`    "paths":`  
`    {`  
`        "data_path": "~/repos/fashion-mnist/data/fashion",`  
`        "model_path": "./fashion_mnist_model.pth"`  
`    }`  
`}`  

The batch size is applied to training, validation and test.

The `data_path` is the path required to use the dataset, `fashion-mnist/data/fashion` should remain the same regardless of where the data is stored on your machine.

The model architecture is set as a variable in the command line, rather than in the config file, as there is a limited set of choices, those currently being `"resnet18"`, `"resnet34"` and `"resnet50"`.
The compute type can be set to `"gpu"` (code will print `cuda:0` if available) or `"cpu"` (code will print `cpu`) in the command line too.

On the command line, run the following with the working directory set as the project folder:  
`python src/train.py --config_file="config/config.json" --model_architecture="resnet18" --compute_type="gpu"`