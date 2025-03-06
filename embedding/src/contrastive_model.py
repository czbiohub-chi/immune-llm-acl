import numpy as np
import torch
from lightning import LightningModule
from torch import nn
from torch.nn.functional import cross_entropy

# fmt: off
# isort: off
from mlp import MLP
# isort: on
# fmt: on


def contrastive_loss(logits_per_gene, logits_per_exp):
    batch_idxs = torch.arange(logits_per_gene.shape[0], device=logits_per_gene.device)
    loss_gene = cross_entropy(logits_per_gene, batch_idxs)
    loss_exp = cross_entropy(logits_per_exp, batch_idxs)
    loss = (loss_gene + loss_exp) / 2
    return loss


class ContrastiveCRISPR(nn.Module):
    def __init__(
        self,
        *,  # enforce kwargs
        exp_input: int = 3072 * 3,
        exp_reduction: int = 3,
        exp_layers: int = 2,
        gene_input: int = 3072,
        gene_reduction: int = 2,
        gene_layers: int = 2,
        shared_dim: int = 512,
    ):
        super().__init__()
        self.exp_proj = MLP(
            input_dim=exp_input,
            reduction_factor=exp_reduction,
            n_hidden=exp_layers,
            output_dim=shared_dim,
        )
        self.gene_proj = MLP(
            input_dim=gene_input,
            reduction_factor=gene_reduction,
            n_hidden=gene_layers,
            output_dim=shared_dim,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self,
        *,
        gene_emb,
        method_emb,
        cell_emb,
        phenotype_emb,
        **kwargs,
    ):
        exp_emb = torch.concat([method_emb, cell_emb, phenotype_emb], dim=1)

        gene_out = self.gene_proj(gene_emb.float())
        exp_out = self.exp_proj(exp_emb.float())

        # normalized features
        gene_norm = gene_out / gene_out.norm(dim=1, keepdim=True)
        exp_norm = exp_out / exp_out.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_gene = logit_scale * gene_norm @ exp_norm.T
        logits_per_exp = logits_per_gene.T

        # shape = [global_batch_size, global_batch_size]
        return logits_per_gene, logits_per_exp


class LightningContrastiveModel(LightningModule):
    def __init__(
        self,
        exp_input: int = 9216,
        exp_reduction: int = 3,
        exp_layers: int = 2,
        gene_input: int = 3072,
        gene_reduction: int = 2,
        gene_layers: int = 2,
        shared_dim: int = 512,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.contraster = ContrastiveCRISPR(
            exp_input=self.hparams.exp_input,
            exp_reduction=self.hparams.exp_reduction,
            exp_layers=self.hparams.exp_layers,
            gene_input=self.hparams.gene_input,
            gene_reduction=self.hparams.gene_reduction,
            gene_layers=self.hparams.gene_layers,
            shared_dim=self.hparams.shared_dim,
        )

    def _common_step(self, batch, batch_idx):
        batch_size = batch[list(batch.keys())[0]].shape[0]
        logits_per_gene, logits_per_exp = self.contraster(**batch)
        loss = contrastive_loss(logits_per_gene, logits_per_exp)
        return loss, batch_size

    def validation_step(self, batch, batch_idx):
        loss, batch_size = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, batch_size=batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        loss, batch_size = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, batch_size=batch_size)
        return loss
