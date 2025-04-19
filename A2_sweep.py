from A2_main import main
import wandb

import sys

wandb_sweep_id = sys.argv[1]
project = sys.argv[2]
wandb_id = sys.argv[3]

wandb.login(key = wandb_id, verify = True)

def train(config_init=None):
  # Initialize a new wandb run
    with wandb.init(config=config_init) as run:

        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        run_name = str(config).replace("': '", ' ').replace("'", '')
        print(run_name)
        run.name = run_name
       
        params = ['data_augmentation', 'batch_normalization', 'filter_size', 'filter_change_ratio', 'stride',
                   'padding', 'activation_function', 'dropout_ratio', 'convolutional_layers']

        main([f'--{i}={config[i]}' for i in params])


wandb.agent(wandb_sweep_id, train, project=project)
            