import os
import sys
import torch
import logging
import optuna
from optuna.trial import TrialState
from copy import deepcopy 
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)

original_sys_path = sys.path.copy()
sys.path.insert(0, "/Users/florian/Documents/github/study/IoT/IoT/main/")
from dataprep import create_DataLoader 
from foo import LSTM
sys.path = original_sys_path

# Set paths
BASE_PATH = os.getenv('BASE_PATH', "/Users/florian/Documents/github/study/IoT/IoT/")
sys.path.insert(0, os.path.join(BASE_PATH, "main"))

# Set parameters
FILEPATH = os.path.join(BASE_PATH, "data/aggregated_data/agg_hourly.parquet")
FILEPATH_STUDY = os.path.join(BASE_PATH, "models/train/LSTM_trials.csv")
window_size = 50
train_test_split_ratio = 0.8
batch_size = 64
features = ["tmp", "CO2", "hum", "VOC"]
target = "tmp"
scaling = True
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', 10))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.0001))

# Objective function
def objective(trial):
    """
    Objective function for the Optuna optimization. Trains an LSTM model with the given hyperparameters.

    args:   trial: optuna.trial.Trial
    returns: float
    """
    hidden_size = trial.suggest_categorical('hidden_size', [50, 100, 150])
    num_layers = trial.suggest_categorical('num_layers', [1, 2, 3, 4])
    activation = trial.suggest_categorical('activation', ['relu', 'sigmoid', 'tanh'])
    if num_layers == 1:
        dropout = 0
    else:    
        dropout = trial.suggest_categorical('dropout', [0, 0.5, 1])

    model = LSTM(input_size= data["train"].x.shape[2], hidden_size= hidden_size, num_layers= num_layers, output_size= 1, dropout= dropout, activation= activation)
    optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE)
    model.train()

    for epoch in range(NUM_EPOCHS):
        loop = tqdm(enumerate(data["train"].loader), total=len(data["train"].loader), leave=True)
        for batch_idx, (features, target) in loop:
            optimizer.zero_grad(set_to_none= True)
            output = model(features)
            loss = torch.nn.MSELoss()(output, target)
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
            loop.set_postfix(loss=loss.item(), lr= LEARNING_RATE)
            loop.update()

        trial.report(loss.item(), epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return loss.item()



if __name__ == "__main__":

    data = create_DataLoader(filepath= FILEPATH, window_size= window_size, train_ratio= train_test_split_ratio, batch_size= batch_size, features= features, target= target, scaling= scaling)
    study = optuna.create_study(direction="maximize", study_name= "LSTM")
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logging.info("Study statistics: ")
    logging.info("  Number of finished trials: {}".format(len(study.trials)))
    logging.info("  Number of pruned trials: {}".format(len(pruned_trials)))
    logging.info("  Number of complete trials: {}".format(len(complete_trials)))

    logging.info("Best trial:")
    trial = study.best_trial

    logging.info("  Value: {}".format(trial.value))

    logging.info("  Params: ")
    for key, value in trial.params.items():
        logging.info("    {}: {}".format(key, value))

    trials = study.trials_dataframe()
    trials.to_csv(FILEPATH_STUDY)

    