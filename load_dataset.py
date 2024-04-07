import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet


def read_csv_neurosity_dataset(file, include_finetune=True):
    #data = pd.read_csv(file, nrows=10000)
    #data = pd.read_pickle(file)#, nrows=1000)
    data = pd.read_pickle(file)
    if include_finetune:
        data2 = pd.read_pickle("combined_dataset_finetune.pkl")
        data2["session_id"] = data2["session_id"].astype(str)
        data = pd.concat([data, data2], ignore_index=True)
        #import pdb;pdb.set_trace()

    max_encoder_length = 257
    max_prediction_length = 1
    N = int(len(data) * 1.0)

    data["index"] = range(len(data))

    data["left_hand"] = data["left_hand"].astype(str)
    data["right_hand"] = data["right_hand"].astype(str)
    data["showing"] = data["showing"].astype(str)
    data["doing"] = data["doing"].astype(str)

    #groups = [group for _, group in data.groupby('session_id')]
    #np.random.shuffle(groups)
    #data = pd.concat(groups).reset_index(drop=True)

    training = TimeSeriesDataSet(
        data,
        time_idx="index",
        target=[
            "CP3",
            "C3",
            "F5",
            "PO3",
            "PO4",
            "F6",
            "C4",
            "CP4",
            "left_hand",
            "right_hand",
        ],
        group_ids=["session_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"],
        time_varying_unknown_categoricals=["left_hand", "right_hand"],
        time_varying_known_reals=["timestamp"],
        time_varying_known_categoricals=["showing", "doing"],
    )
    print("Created dataset")

    # create validation and training dataset
    validation = TimeSeriesDataSet.from_dataset(
        training,
        data,
        min_prediction_idx=training.index.time.max() + 1,
        stop_randomization=True,
    )
    training = TimeSeriesDataSet.from_dataset(
        training, data.head(N), min_prediction_idx=0, stop_randomization=True
    )
    batch_size = 128
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=2
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=2
    )

    return train_dataloader, val_dataloader, training
