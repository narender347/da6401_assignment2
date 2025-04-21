**PART-A**

modular architecture that makes it easy to:
Swap out CNNs or even other neural architectures Customize training from the command line Use Weights & Biases (wandb) for experiment tracking and sweeps.

A2_conv.py
here we are essentially building a plug-and-play LightningModule:

Users define model = nn.Sequential(...) or custom modules
Define custom loss_function, accuracy_function, optimizer_function, and optimizer_params,Pass them all into ConvolutionalNN for a reusable training logic.This makes A2_conv.py a flexible training backend.

A2_train.py
This is experiment driver with CLI support. You can do things like:

bash
Copy
Edit
python A2_train.py --dataset inaturalist_12K --epochs 50 --batch_size 64 --activation_function ReLU --filter_size 3
And use it for wandb sweeps:

bash
Copy
Edit
python A2_train.py --wandb_project myproject --wandb_entity myname --wandb_sweepid sweep123,Which is good for reproducibility and scalability.


 Building a flexible CNN generator based on args like filter_size, activation_function, convolutional_layers, Adding validation/test metrics and logging them to wandb, Packaging the CLI for more model types (like ResNet, MLPs, etc.)
Computing the number of parameters or FLOPs per model, Automatically saving best models via ModelCheckpoint.
