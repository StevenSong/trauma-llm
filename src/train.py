import os

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
from wandb.util import generate_id

# fmt: off
# isort: off
from model import LightningClassifierModel
from data import LightningTraumaData
# isort: on
# fmt: on


class StrictWandbLogger(WandbLogger):
    def __init__(self, *, project: str, name: str, version: str, save_dir: str):
        version = version + "-" + generate_id()
        save_dir = os.path.join(save_dir, name, version)
        super().__init__(project=project, name=name, version=version, save_dir=save_dir)
        if os.path.exists(self.save_dir):
            raise FileExistsError(
                "\033[91mREAD THIS ERROR MSG: \033[0m"
                f"Experiment already exists at {self.save_dir}."
                " This logger uses some custom logic to put all logs,"
                " checkpoints, and configs related to an experiment"
                " under one directory. Please delete or rename to retry."
            )
        os.makedirs(self.save_dir)


def run():
    cli = LightningCLI()


if __name__ == "__main__":
    run()
