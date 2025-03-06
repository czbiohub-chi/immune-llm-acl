import os

from lightning import LightningDataModule
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, default_collate
from wandb.util import generate_id

# fmt: off
# isort: off
from classifier_model import LightningClassifierModel
from contrastive_model import LightningContrastiveModel
from screen_dataset import VALIDATION_SCREENS, ScreenDataset
# isort: on
# fmt: on


class Data(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        use_summarized: bool,
        hits_only: bool,
        batch_size: int,
        num_workers: int,
        validation_screens: list[int] = VALIDATION_SCREENS,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str):
        if stage == "fit":
            self.train_ds = ScreenDataset(
                data_dir=self.hparams.data_dir,
                use_summarized=self.hparams.use_summarized,
                hits_only=self.hparams.hits_only,
                blacklist=self.hparams.validation_screens,
            )
            self.val_ds = ScreenDataset(
                data_dir=self.hparams.data_dir,
                use_summarized=self.hparams.use_summarized,
                hits_only=self.hparams.hits_only,
                whitelist=self.hparams.validation_screens,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            batch_size=self.hparams.batch_size,
            collate_fn=default_collate,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            batch_size=self.hparams.batch_size,
            collate_fn=default_collate,
            num_workers=self.hparams.num_workers,
        )


class StrictWandbLogger(WandbLogger):
    def __init__(self, *args, **kwargs):
        assert "version" not in kwargs or kwargs["version"] is None
        assert "id" not in kwargs or kwargs["id"] is None
        kwargs["id"] = generate_id()
        kwargs["save_dir"] = os.path.join(
            kwargs["save_dir"], kwargs["name"], kwargs["id"]
        )
        super().__init__(*args, **kwargs)
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
    cli = LightningCLI(datamodule_class=Data)


if __name__ == "__main__":
    run()
