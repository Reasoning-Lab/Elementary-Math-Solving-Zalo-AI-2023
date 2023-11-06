import random
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run_experiment(cfg):
	configs = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
	# print(configs)
	wandb.login(key=cfg.wandb.WANDB_API_KEY)
	run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, config=configs)
	return wandb

if __name__ == "__main__":
    run_experiment()
    # Simulate training
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset
        # log metrics to wandb
        wandb.log({"acc": acc, "loss": loss})
    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()