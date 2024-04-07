# !pip install pytorch-forecasting
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
import pandas as pd
import pytorch_forecasting
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, DeepAR
from pytorch_forecasting.metrics import MAE, SMAPE, MultivariateNormalDistributionLoss
from pytorch_lightning import loggers as pl_loggers
tensorboard = pl_loggers.TensorBoardLogger('./')
from pytorch_forecasting.metrics.base_metrics import MultiLoss

from load_dataset import train_dataloader, val_dataloader, training

# next(iter(train_dataloader))

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

# create the model
net = DeepAR.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=512,
    log_interval=2,
    reduce_on_plateau_patience=4,
    rnn_layers=3,
    loss=MultiLoss([MultivariateNormalDistributionLoss(rank=30) for _ in range(8)]),
    optimizer="Adam"
)
print(f"Number of parameters in network: {net.size()/1e3:.1f}k")

res = Tuner(trainer).lr_find(
    net,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    min_lr=1e-5,
    max_lr=1e0,
    early_stop_threshold=100,
)
print(f"suggested learning rate: {res.suggestion()}")

trainer.fit(net, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
