import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
	config = OmegaConf.to_yaml(cfg)
	print(config)
	print(type(config), type(cfg))

if __name__ == "__main__":
	train()