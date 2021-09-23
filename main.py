# torch based module
from pytorch_lightning.accelerators import accelerator
import torch

# base python module
import argparse
import wandb
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import os
import shutil

# custom module
from LightingModule import model_dict

MODEL_CHOICES = list(model_dict.keys())


def main():
    
    seed_everything()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str, 
                        choices=["cifar10", "stl10", "imagenet"])
    parser.add_argument("--model", choices=MODEL_CHOICES)
    script_args, _ = parser.parse_known_args()
    Model = model_dict[script_args.model]
    parser = Model.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.log_dir = None
    
    base_log_dir = f"./log/{args.model}"
    os.makedirs(base_log_dir, exist_ok=True)
    
    # this makes 2 directories if i use multiple gpus
    if not args.log_dir:
        v_num = 0
        while os.path.exists(f"{base_log_dir}/{v_num}"):
            v_num += 1
        args.log_dir = f"{base_log_dir}/{v_num}"
        
        os.makedirs(args.log_dir)
    
    # save file to log directory
    model_file_path = f"LightingModule/{args.model}.py"
    shutil.copy(model_file_path, args.log_dir+f"/{args.model}.py")
    
    wandb_logger = WandbLogger(
        project=f"{args.model}",
        name=f"{os.environ['servername']}-{os.environ['dockername']}-v{v_num}-{args.dataset}" + \
              "-split{args.class_split}",
        log_model="all",
        save_dir=args.log_dir,
        log_model=False
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.log_dir,
        save_weights_only=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    model = Model(**vars(args))
    # trainer = Trainer.from_argparse_args(args)
    trainer = Trainer.from_argparse_args(args, logger=wandb_logger, 
                                         callbacks=[checkpoint_callback, lr_monitor])
    # wandb_logger.watch(model)
    wandb_logger.log_hyperparams(args)
    trainer.fit(model)
    
    return model
    

if __name__ == "__main__":
    
    model = main()
    
    