import numpy as np
import torch
from lightning import LightningModule
from torch import nn
from torch.nn.functional import cross_entropy, softmax

# fmt: off
# isort: off
from mlp import MLP
# isort: on
# fmt: on


def classification_loss(logits, hits):
    targets = torch.as_tensor(
        [1 if hit == "YES" else 0 for hit in hits],
        device=logits.device,
    )
    loss = cross_entropy(logits, targets)
    return loss


class ClassifierCRISPR(nn.Module):
    def __init__(
        self,
        *,  # enforce kwargs
        input_dim: int = 3072 * 4,
        reduction: int = 2,
        layers: int = 4,
    ):
        super().__init__()
        self.mlp = MLP(
            input_dim=input_dim,
            reduction_factor=reduction,
            n_hidden=layers,
            output_dim=2,
        )

    def forward(
        self,
        *,
        method_emb,
        cell_emb,
        phenotype_emb,
        gene_emb,
        **kwargs,
    ):
        emb = torch.concat([method_emb, cell_emb, phenotype_emb, gene_emb], dim=1)
        emb = emb.float()
        logits = self.mlp(emb)
        return logits, softmax(logits, dim=1)


class LightningClassifierModel(LightningModule):
    def __init__(
        self,
        input_dim: int = 9216,
        reduction: int = 3,
        layers: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.classifier = ClassifierCRISPR(
            input_dim=self.hparams.input_dim,
            reduction=self.hparams.reduction,
            layers=self.hparams.layers,
        )

    def _common_step(self, batch, batch_idx):
        batch_size = batch[list(batch.keys())[0]].shape[0]
        logits, _ = self.classifier(**batch)
        loss = classification_loss(logits, batch["hit"])
        return loss, batch_size

    def validation_step(self, batch, batch_idx):
        loss, batch_size = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, batch_size=batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        loss, batch_size = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, batch_size=batch_size)
        return loss
