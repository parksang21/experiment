# torch based module
from pytorch_lightning.accelerators import accelerator
import torch

# base python module
import argparse
import wandb
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
import os

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
    
    log_dir = f"./log/{args.model}"
    os.makedirs(log_dir, exist_ok=True)
    
    

    wandb_logger = WandbLogger(
        project=f"{args.model}",
        name=f"{os.environ['servername']}-{os.environ['dockername']}",
        log_model="all",
        save_dir=log_dir,
    )
    
    model = Model(**vars(args))
    # trainer = Trainer.from_argparse_args(args)
    trainer = Trainer.from_argparse_args(args, logger=wandb_logger)
    wandb_logger.watch(model)
    wandb_logger.log_hyperparams(args)
    trainer.fit(model)
    
    return model
    

if __name__ == "__main__":
    
    model = main()
    
    