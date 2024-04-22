# ml-challenge
Train a deep learning model on the fashion-MNIST dataset.  
The repository that contains the fashion-mnist dataset required to run this code is subject to the following coptyright licence:  
The MIT License (MIT) Copyright © 2017 Zalando SE, https://tech.zalando.com

## About
This code is used to train a model to classify grayscale images from the Fashion-MNIST dataset with an accuracy of 86% on the 10000 test images. The train/val dataset split is `0.33` (train dataset: 40200 images, validation dataset: 19800 images). This parameter is not currently configurable.

Here is an example of the output of a model being trained from the python script `src/train.py`:  
`"batch_size": 128`  
`"epochs": 2`  
`"model_architecture": "resnet34"`  
`"compute_type": "gpu"`  

The printout is as follows:  
`device: cuda:0`  
`Training - Epoch 1`  
`[epoch: 1, batch: 100], loss: 0.964`  
`[epoch: 1, batch: 200], loss: 0.519`  
`[epoch: 1, batch: 300], loss: 0.473`  
`[epoch: 1, batch: 400], loss: 0.425`  
`Validation - Epoch 1`  
`Accuracy of the network on the dataset: 87 %`  
`Accuracy for class: T-shirt/top is 87.6 %`  
`Accuracy for class: Trouser is 96.6 %`  
`Accuracy for class: Pullover is 75.5 %`  
`Accuracy for class: Dress is 91.4 %`  
`Accuracy for class: Coat  is 82.9 %`  
`Accuracy for class: Sandal is 94.4 %`  
`Accuracy for class: Shirt is 57.4 %`  
`Accuracy for class: Sneaker is 91.4 %`  
`Accuracy for class: Bag   is 98.0 %`  
`Accuracy for class: Ankle Boot is 97.0 %`  
`Training - Epoch 2`  
`[epoch: 2, batch: 100], loss: 0.356`  
`[epoch: 2, batch: 200], loss: 0.344`  
`[epoch: 2, batch: 300], loss: 0.344`  
`[epoch: 2, batch: 400], loss: 0.338`  
`Validation - Epoch 2`  
`Accuracy of the network on the dataset: 89 %`  
`Accuracy for class: T-shirt/top is 85.8 %`  
`Accuracy for class: Trouser is 97.8 %`  
`Accuracy for class: Pullover is 80.1 %`  
`Accuracy for class: Dress is 87.4 %`  
`Accuracy for class: Coat  is 92.7 %`  
`Accuracy for class: Sandal is 98.3 %`  
`Accuracy for class: Shirt is 66.5 %`  
`Accuracy for class: Sneaker is 94.7 %`  
`Accuracy for class: Bag   is 97.7 %`  
`Accuracy for class: Ankle Boot is 96.4 %`  
`Finished Training`  
`Testing`  
`Accuracy of the network on the dataset: 86 %`  
`Accuracy for class: T-shirt/top is 82.1 %`  
`Accuracy for class: Trouser is 96.5 %`  
`Accuracy for class: Pullover is 73.3 %`  
`Accuracy for class: Dress is 85.2 %`  
`Accuracy for class: Coat  is 89.0 %`  
`Accuracy for class: Sandal is 96.7 %`  
`Accuracy for class: Shirt is 56.7 %`  
`Accuracy for class: Sneaker is 94.6 %`  
`Accuracy for class: Bag   is 95.5 %`  
`Accuracy for class: Ankle Boot is 96.0 %`  

The test class accuracy is not always the same and could be due to the shuffling of the training dataset but could also just be the non-determinsm of the model.

## How to Run
### Environment Setup
You will need a python environment using the requirements file that can be found under:
`"environments/ml_venv/requirements.txt"`.
Instructions on how to create a python virtual environment can be found here:  
https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/  
Or alternatively install these libraries on a clean python/conda environment if you can't use the requirements file.

You will then need to clone the following repository to access the Fashion-MNIST dataset:  
https://github.com/zalandoresearch/fashion-mnist  
The local repository location should be adjacent to this repository, but is not essential.

In order to access the data from the cloned repo, you need to make sure that the path that is appended to the system (cell 4 in model_train.ipynb and line 15 in train.py) is that of the fashion-mnist repo location,  
e.g. `'/home/<user>/repos/fashion-mnist'`

### Running the Notebook

To run the notebook `notebooks/model_train.ipynb`, you will need to set up a kernel with the environment created earlier, this can be done using the following command:

`ipython kernel install --user --name=<kernel_name>`

The name of the kernel can be different to the evironment name, but it makes sense to keep them the same.  
You will then be able to select the kernel in your chosen IDE.

### Running the Python Script

There is a config file called `config.json` under the `config` folder which contain parameters for running training and also paths to the data and output model locations. You will need to call this in the run command when you run the training script `src/train.py`.

An example of the config file looks like this:

`{`  
`    "params":`  
`    {`  
`        "batch_size": 128,`  
`        "epochs": 2`  
`    },`  
`    "paths":`  
`    {`  
`        "data_path": "/home/jbloggs/repos/fashion-mnist/data/fashion",`  
`        "model_path": "/home/jbloggs/repos/ml-challenge/models/fashion_mnist_model.pth"`  
`    }`  
`}`  

The batch size is applied to training, validation and test.

The `data_path` is the path required to use the dataset. `fashion-mnist/data/fashion` should remain the same regardless of where the data is stored on your machine.

The model architecture is set as a variable in the command line, rather than in the config file, as there is a limited set of choices, those currently being `"resnet18"`, `"resnet34"` and `"resnet50"`.
The compute type can be set to `"gpu"` (code will print `cuda:0` if available and `cpu` if unavailable) or `"cpu"` (code will print `cpu`) in the command line too.

On the command line, run the following with the working directory set as the project folder:  
`python src/train.py --config_file="config/config.json" [--model_architecture="resnet18" --compute_type="gpu"]`