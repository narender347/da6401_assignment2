There are two main files in this folder:

convnetwork.py
train.py
convnetwork.py
This is the class implementation of LightningModule so that it is flexible to create differnt types of CNN or other type of models. We can choose  to decide model, loss_function, accuracy_function, optimizer_function, optimizer_params.

model = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
    torch.nn.Linear(128, 10)
)
optimizer_function = torch.optim.Adam
optimizer_params = {}
accuracy_function = torchmetrics.Accuracy(task="multiclass", num_classes=10)
loss_function = torch.nn.CrossEntropyLoss()

model = ConvolutionalNN(model=model, loss_function=loss_function, accuracy_function=accuracy_function, 
optimizer_function=optimizer_function, optimizer_params=optimizer_params)
train.py
This is a command line executable code for easy designing of architecture, this file is also getting used to preform wandb sweeps. The command line arguments are as follows:

train.py [-h] [-wandb_project WANDB_PROJECT] [-wandb_entity WANDB_ENTITY] [-wandb_sweepid WANDB_SWEEPID] [-dataset {inaturalist_12K}] [-epochs EPOCHS]
                [-batch_size BATCH_SIZE] [-activation_function {ReLU,Tanh,GELU,Mish,LeakyReLU}] [-filter_size FILTER_SIZE]
                [-filter_change_ratio FILTER_CHANGE_RATIO] [-stride STRIDE] [-padding PADDING] [-size_dense SIZE_DENSE]
                [-data_augmentation DATA_AUGMENTATION] [-batch_normalization BATCH_NORMALIZATION] [-dropout_ratio DROPOUT_RATIO]
                [-convolutional_layers CONVOLUTIONAL_LAYERS]
