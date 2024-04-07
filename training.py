# !pip install pytorch-forecasting
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss, RMSE
import pandas as pd
import pytorch_forecasting
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
# next(iter(train_dataloader))
from pytorch_lightning import loggers as pl_loggers
tensorboard = pl_loggers.TensorBoardLogger('./')
import logging
# configure logging at the root level of Lightning
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

from load_dataset import read_csv_neurosity_dataset


#train_dataloader, val_dataloader, training = read_csv_neurosity_dataset("combined_dataset.csv")
train_dataloader, val_dataloader, training = read_csv_neurosity_dataset("combined_dataset.pkl")

from lightning.pytorch.callbacks import Callback

class PrintCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
        baseline_predictions = pl_module.predict(val_dataloader, return_y=True)
        mse=RMSE()(baseline_predictions.output, baseline_predictions.y).mean()
        print(mse)
        pl_module.log({"val_mse": mse})

# define trainer with early stopping
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",
    #accelerator="gpu",
    gradient_clip_val=0.1,
    limit_train_batches=30,
    #callbacks=[lr_logger, early_stop_callback],
    callbacks=[lr_logger, PrintCallback()],
    logger=tensorboard,
    #devices=[1,3,4],
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

trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
