# !pip install pytorch-forecasting
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
import pandas as pd
import pytorch_forecasting
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
# next(iter(train_dataloader))
from pytorch_lightning import loggers as pl_loggers
tensorboard = pl_loggers.TensorBoardLogger('./')

from load_dataset import read_csv_neurosity_dataset

train_dataloader, val_dataloader, training = read_csv_neurosity_dataset("combined_dataset.csv")

# define trainer with early stopping
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",
    gradient_clip_val=0.1,
    limit_train_batches=30,
    #callbacks=[lr_logger, early_stop_callback],
    callbacks=[lr_logger],
    logger=tensorboard
)

from create_model import create_tft_model

model = create_tft_model(training)

# find optimal learning rate (set limit_train_batches to 1.0 and log_interval = -1)
#res = Tuner(trainer).lr_find(
#    tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
#)

#print(f"suggested learning rate: {res.suggestion()}")
# fig = res.plot(show=True, suggest=True)
# fig.show()

trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
