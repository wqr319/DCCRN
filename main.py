import torch
import pytorch_lightning as pl
from data import get_loader
from build import Plmodule,build_trainer

train_loader = get_loader()
trainer = build_trainer()
plmodule = Plmodule()
trainer.fit(plmodule,train_dataloaders=train_loader)
