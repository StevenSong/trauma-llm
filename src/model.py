from typing import Literal

import torch
from lightning import LightningModule
from torch import nn
from torch.nn.functional import cross_entropy, softmax

BACKBONE_TYPE = Literal["mamba"]


def extract_causal_embedding(backbone_output, batch):
    batch_size = len(batch["input_ids"])
    last_token_idxs = batch["attention_mask"].sum(axis=1) - 1
    emb = backbone_output.last_hidden_state[torch.arange(batch_size), last_token_idxs]
    return emb


class Classifier(nn.Module):
    def __init__(
        self,
        *,  # enforce kwargs
        backbone_name: str,
        backbone_type: BACKBONE_TYPE,
        backbone_dim: int,
        n_classes: int,
    ):
        super().__init__()
        if backbone_type == "mamba":
            from transformers import MambaModel

            self.backbone = MambaModel.from_pretrained(backbone_name)
            self.extract_emb_fn = extract_causal_embedding
        else:
            raise NotImplementedError(f"Unsupported backbone type: {backbone_type}")
        self.cls = nn.Linear(backbone_dim, n_classes)

    def forward(self, **batch):
        out = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        emb = self.extract_emb_fn(out, batch)
        logits = self.cls(emb)
        return logits, softmax(logits, dim=1)


class LightningClassifierModel(LightningModule):
    def __init__(
        self,
        backbone_name: str,
        backbone_type: BACKBONE_TYPE,
        backbone_dim: int,
        n_classes: int,
        cls_target: str,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = Classifier(
            backbone_name=self.hparams.backbone_name,
            backbone_type=self.hparams.backbone_type,
            backbone_dim=self.hparams.backbone_dim,
            n_classes=self.hparams.n_classes,
        )

    def _common_step(self, batch, batch_idx):
        batch_size = batch[list(batch.keys())[0]].shape[0]
        logits, _ = self.model(**batch)
        loss = cross_entropy(logits, batch[self.hparams.cls_target])
        return loss, batch_size

    def training_step(self, batch, batch_idx):
        loss, batch_size = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, batch_size = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, batch_size=batch_size, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, batch_size = self._common_step(batch, batch_idx)
        self.log("test_loss", loss, batch_size=batch_size, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        _, preds = self.model(**batch)
        mrns = batch["mrn"]
        trues = batch[self.hparams.cls_target]
        return preds, trues, mrns
