from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from training.trainer import Trainer
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, default='default', help='Name of the config file to use (without .yaml)')
args = parser.parse_args()

config_name = args.config_name
with initialize(version_base=None, config_path="training/config"):
    cfg = compose(config_name=config_name)

trainer = Trainer(**cfg)
trainer.run()
m=1
