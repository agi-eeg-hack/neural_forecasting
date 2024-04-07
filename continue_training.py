# !pip install pytorch-forecasting
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss, RMSE
import pandas as pd
import pytorch_forecasting
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import torch
# next(iter(train_dataloader))
from pytorch_lightning import loggers as pl_loggers
tensorboard = pl_loggers.TensorBoardLogger('./')
import logging
# configure logging at the root level of Lightning
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

import argparse
parser = argparse.ArgumentParser(description="Model training script")
parser.add_argument("--checkpoint_path", type=str, default="checkpoints/",
                            help="Directory where the model checkpoints will be saved")

args = parser.parse_args()

from load_dataset import read_csv_neurosity_dataset

train_dataloader, val_dataloader, training = read_csv_neurosity_dataset("combined_dataset_finetune.pkl", include_finetune=False)

from lightning.pytorch.callbacks import Callback

class PrintCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
        #baseline_predictions = pl_module.predict(val_dataloader, return_y=True)
        mse = 0
        n = 0
        for x,y in val_dataloader:
            baseline_predictions = pl_module({k: (v.to(pl_module.device) if not isinstance(v,list) else [vv.to(pl_module.device) for vv in v]) for k,v in x.items()})
            ypred = torch.concatenate(baseline_predictions['prediction'][:8], dim=1)[:,:,3]
            y = torch.concatenate(y[0][:8], dim=1).to(pl_module.device)
            #import pdb; pdb.set_trace()
            mse += ((ypred - y)**2).mean()
            n+=val_dataloader.batch_size

        mse /= n
        mse = mse
        #mse=RMSE()(baseline_predictions.output, baseline_predictions.y).mean()
        print(mse)
        pl_module.log("val_mse", mse)
        #print(trainer.logged_metrics)

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
    #callbacks=[lr_logger, PrintCallback()],
    callbacks=[lr_logger],
    logger=tensorboard,
    #devices=[3,4],
)

model = TemporalFusionTransformer.load_from_checkpoint(args.checkpoint_path)

# find optimal learning rate (set limit_train_batches to 1.0 and log_interval = -1)
#res = Tuner(trainer).lr_find(
#    tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
#)

#print(f"suggested learning rate: {res.suggestion()}")
# fig = res.plot(show=True, suggest=True)
# fig.show()

trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
