# !pip install pytorch-forecasting
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
import pandas as pd
import pytorch_forecasting
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
data = pd.read_csv("combined_dataset.csv")

#%%

max_encoder_length = 36
max_prediction_length = 6
N = int(len(data)*0.8)

data['index'] = data.index

data.sample(frac=1)

training = TimeSeriesDataSet(
    data,
    time_idx= "index",
    target=["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"],
    group_ids=["session_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"],
    time_varying_known_reals=["timestamp"],
)

# create validation and training dataset
validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training.index.time.max() + 1, stop_randomization=True)
training = TimeSeriesDataSet.from_dataset(training, data.head(N), min_prediction_idx=0, stop_randomization=True)
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)
