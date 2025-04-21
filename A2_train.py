import argparse
import numpy as np
import torchmetrics
from A2_conv import ConvolutionalNN
import torch
import torchvision as tv
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
import os





parser = argparse.ArgumentParser(description='CNN Training Configuration')

parser.add_argument('-wandb_project', '--wandb_project', type=str, default='myprojectname', help='Project name used to track experiments in Weights & Biases dashboard')
parser.add_argument('-wandb_entity', '--wandb_entity', type=str, default='myname', help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
parser.add_argument('-wandb_sweepid', '--wandb_sweepid', type=str, default=None, help='Wandb Sweep Id to log in sweep runs the Weights & Biases dashboard.')
parser.add_argument('-dataset', '--dataset', type=str, default='inaturalist_12K', choices=["inaturalist_12K"], help='Dataset choices: ["inaturalist_12K"]')

parser.add_argument('-epochs', '--epochs', type=int, default=10, help='Number of epochs to train the model.')
parser.add_argument('-batch_size', '--batch_size', type=int, default=32, help='Batch size for training the model.')
parser.add_argument('-activation_function', '--activation_function', type=str, default='ReLU', choices=['ReLU', 'Tanh', 'GELU', 'Mish','LeakyReLU'], help='Activation function choices: ["ReLU", "Tanh", "GELU", "Mish","LeakyReLU"]')
parser.add_argument('-filter_size', '--filter_size', type=int, default=5, help='Filter size for all convolutional layers.')
parser.add_argument('-filter_change_ratio', '--filter_change_ratio', type=float, default=1.3, help='Filter change ratio for convolutional layers, if 2 then number of filters doubles or if 0.5 filters halves supsequent layer.')
parser.add_argument('-stride', '--stride', type=int, default=1, help='Stride for all convolutional layers.')

parser.add_argument('-padding', '--padding', type=bool, default=True, help='Enable padding or not.')

parser.add_argument('-size_dense', '--size_dense', type=int, default=50, help='Size of the dense layer.')

parser.add_argument('-data_augmentation', '--data_augmentation', type=bool, default=True, help='Enable data augmentation.')
parser.add_argument('-batch_normalization', '--batch_normalization', type=bool, default=True, help='Enable batch normalization.')
parser.add_argument('-dropout_ratio', '--dropout_ratio', type=float, default=0.3, help='Dropout ratio for the model.')
parser.add_argument('-convolutional_layers', '--convolutional_layers', type=int, default=5, help='Number of convolutional layers in the model.')


args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

image_transform = [
    tv.transforms.ToTensor(),
    tv.transforms.Resize((300, 300)),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

if args.data_augmentation:
    image_transform.extend([
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomRotation(20),
    ])

dataset = tv.datasets.ImageFolder(root='inaturalist_12K/train', transform=tv.transforms.Compose(image_transform))
train_data, val_data = torch.utils.data.random_split(dataset, [0.8, 0.2])
input_data_dimension = train_data[0][0].shape
padding = 0
if args.padding:
    padding = int(args.filter_size/2)

# Prepare the model layers
model_layers = torch.nn.Sequential()
activation_function = getattr(torch.nn, args.activation_function)
num_filters = 3 # RGB channels from the input image

for i in range(args.convolutional_layers):
    model_layers.extend([
        torch.nn.Conv2d(in_channels=num_filters, out_channels=int(num_filters * args.filter_change_ratio + args.filter_change_ratio),
                        kernel_size=args.filter_size, stride=args.stride, padding=padding),
        activation_function(),
    ])
    num_filters = int(num_filters * args.filter_change_ratio + args.filter_change_ratio)
    model_layers.extend([torch.nn.MaxPool2d(kernel_size=args.filter_size, stride=args.stride, padding=padding)])

model_layers.extend([torch.nn.Flatten(1, -1)])
if args.dropout_ratio > 0:
    model_layers.extend([torch.nn.Dropout(args.dropout_ratio)])
    
num_features_cnn = np.prod(list(model_layers(torch.rand(1, *input_data_dimension)).shape))
if args.batch_normalization:
    model_layers.extend([torch.nn.BatchNorm1d(num_features_cnn)])
model_layers.extend([torch.nn.Linear(num_features_cnn, args.size_dense)])
model_layers.extend([torch.nn.Linear(args.size_dense, 10)])

print(model_layers)

optimizer_function = torch.optim.NAdam
optimizer_params = {}
accuracy_function = torchmetrics.Accuracy(task="multiclass", num_classes=10)
loss_function = torch.nn.CrossEntropyLoss()
model = ConvolutionalNN(model=model_layers, loss_function=loss_function, accuracy_function=accuracy_function, 
                        optimizer_function=optimizer_function, optimizer_params=optimizer_params)
wandb_logger = WandbLogger(project="Assignment 2", reinit=True)
# log gradients and model topology
wandb_logger.watch(model)
trainer  = pl.Trainer(log_every_n_steps=5, max_epochs=args.epochs, logger=wandb_logger)
train_dataloaders = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size)
val_dataloaders = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size)
trainer.fit( model=model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)
