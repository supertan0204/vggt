from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from training.trainer import Trainer


with initialize(version_base=None, config_path="training/config"):
    cfg = compose(config_name="default")      # loads default.yaml

trainer = Trainer(**cfg)
trainer.run()
import pdb;pdb.set_trace()
m=1
