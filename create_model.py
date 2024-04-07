# !pip install pytorch-forecasting
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss, MultiLoss, CrossEntropy, RMSE
from torch import nn
import pandas as pd
import pytorch_forecasting
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
# next(iter(train_dataloader))
from pytorch_lightning import loggers as pl_loggers
tensorboard = pl_loggers.TensorBoardLogger('./')

def create_tft_model(training):
    # create the model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=1024,
        attention_head_size=16,
        dropout=0.0,
        hidden_continuous_size=512,
        output_size=[7]*8+[6]*2,
        #loss=MultiLoss([QuantileLoss() for _ in range(8)]+[CrossEntropy()]*2, weights=[1.0]*8+[0.0]*2),
        #loss=MultiLoss([QuantileLoss() for _ in range(8)]+[CrossEntropy()]*2, weights=[0.0]*8+[1.0]*2),
        loss=MultiLoss([QuantileLoss() for _ in range(8)]+[CrossEntropy()]*2, weights=[0.5]*8+[1.0]*2),
        log_interval=2,
        #logging_metrics=nn.ModuleList([MultiLoss([RMSE()]*8+[RMSE()]*2, weights=[1.0]*8+[0.0]*2)]),
        reduce_on_plateau_patience=4
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
    return tft

def create_deepvar_model(training):
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
    return net

def create_tft_prob_model(training):
    # create the model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=512,
        attention_head_size=8,
        dropout=0.0,
        hidden_continuous_size=256,
        # output_size=[1,1,1,1,1,1,1,1],
        #loss=QuantileLoss(),
        loss=MultiLoss([MultivariateNormalDistributionLoss(rank=30) for _ in range(8)]),
        log_interval=2,
        reduce_on_plateau_patience=4
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
    return tft
