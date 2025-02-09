import os

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from wandb.util import generate_id


@rank_zero_only
def wrapper(save_dir):
    os.makedirs(save_dir)


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

        wrapper(self.save_dir)
