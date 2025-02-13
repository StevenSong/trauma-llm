import os

import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from wandb.util import generate_id


@rank_zero_only
def makedirs_wrapper(save_dir, save_link):
    os.makedirs(save_dir)
    if os.path.exists(save_link):
        os.remove(save_link)
    rel_dir = save_dir
    rel_dir = rel_dir.replace(os.path.dirname(rel_dir), "")
    rel_dir = rel_dir.lstrip(os.path.sep)
    os.symlink(rel_dir, save_link)


class StrictWandbLogger(WandbLogger):
    def __init__(self, *, project: str, name: str, version: str, save_dir: str):
        self.best_link = os.path.join(save_dir, name, version + ".ckpt")
        save_link = os.path.join(save_dir, name, version)

        # make version unique
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
        makedirs_wrapper(self.save_dir, save_link)

    def after_save_checkpoint(self, checkpoint_callback):
        best_model_path = checkpoint_callback.best_model_path
        best_model_path = best_model_path.replace(os.path.dirname(self.best_link), "")
        best_model_path = best_model_path.lstrip(os.path.sep)
        if os.path.exists(self.best_link):
            os.remove(self.best_link)
        os.symlink(
            best_model_path,
            self.best_link,
        )


class PredictionWriter(BasePredictionWriter):
    def __init__(self):
        super().__init__("epoch")

    def write_on_epoch_end(
        self,
        trainer,
        pl_module,
        predictions,
        batch_indices,
    ):
        preds, trues, mrns = [], [], []
        for _preds, _trues, _mrns in predictions:
            preds.append(_preds)
            trues.append(_trues)
            mrns.append(_mrns)
        to_save = {
            "preds": torch.concat(preds),
            "trues": torch.concat(trues),
            "mrns": torch.concat(mrns),
        }
        torch.save(
            to_save,
            os.path.join(trainer.log_dir, f"predictions_rank_{trainer.global_rank}.pt"),
        )
