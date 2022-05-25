# simple test script to make sure that everything is workign or easy debugging: 

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

import wandb
from py_scripts.dataset_params import *
from py_scripts.combine_params import *

model_style = ModelStyles.SDM
dataset = DataSet.SPLIT_MNIST

load_path = None

extras = dict(
    num_workers=0, 
    epochs_to_train_for = 50,
    epochs_per_dataset = 10,
    k_min=1, 
    num_binary_activations_for_gaba_switch = 100000,
)

if load_path:
    print("LOADING IN A MODEL!!!")

model_params, model, data_module = get_params_net_dataloader(model_style, dataset, load_from_checkpoint=load_path, **extras)

wandb_logger = WandbLogger(project="Foundational-SDM", entity="kreiman-sdm", save_dir="wandb_Logger/")
model_params.logger = wandb_logger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("using cuda", device)
    gpu = [0]
else: 
    print("using cpu", device)
    gpu=None

# SETUP TRAINER
if model_params.load_from_checkpoint and model_params.load_existing_optimizer_state:
    fit_load_state = load_path
else: 
    fit_load_state = None

callbacks = []
if model_params.investigate_cont_learning: 
    #import ipdb
    #ipdb.set_trace()
    num_checkpoints_to_keep = -1 # means all of them. 
    model_checkpoint_obj = pl.callbacks.ModelCheckpoint(
        every_n_epochs = model_params.checkpoint_every_n_epochs,
        save_top_k = num_checkpoints_to_keep,
    )
    callbacks.append(model_checkpoint_obj)
    checkpoint_callback = True 
else: 
    checkpoint_callback = False

temp_trainer = pl.Trainer(
        #precision=16, 
        
        logger=model_params.logger,
        max_epochs=model_params.epochs_to_train_for,
        check_val_every_n_epoch=1,
        num_sanity_val_steps = False,
        enable_progress_bar = True,
        gpus=gpu, 
        #gradient_clip_val = model_params.gradient_clip,
        callbacks = callbacks,
        checkpoint_callback=checkpoint_callback, # dont save these test models. 
        reload_dataloaders_every_n_epochs=model_params.epochs_per_dataset, 
        #limit_train_batches=2,
        #profiler="simple" # if on then need to set epochs_to_train_for to a v low score.
        )
temp_trainer.fit(model, data_module)#, ckpt_path=False)#fit_load_state)
wandb.finish()
