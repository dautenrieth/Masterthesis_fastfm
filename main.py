"""
The main module/function is used to structure the program flow.
From here the other program parts like the data generation are called.
And the training and visualization is executed.
"""

import numpy as np
from ogb.linkproppred import Evaluator
from pathlib import Path
from configparser import ConfigParser, ExtendedInterpolation
import time
from sklearn.metrics import mean_squared_error
import scipy.sparse as ssp
from scipy.sparse import save_npz, load_npz
from fastFM import als, mcmc, sgd
import sys, os
from data_generation import create_neg_samples
import utils as ut
from sklearn import metrics
from data_generation import get_data
from logger import logging_setup, save_pred
from tqdm import tqdm
from matplotlib import pyplot as plt
import statistics as stats

logger = logging_setup(__name__)

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("config.ini")

# Select active parts
sgd_part = config["METHODS"].getboolean("sgd")
mcmc_part = config["METHODS"].getboolean("mcmc")
als_part = config["METHODS"].getboolean("als")

vis_folder = config["FOLDERNAMES"]["visualizations_folder"]


def main():
    """
    Run Experiments with defined parameters.
    Computed files are stored in data directory.
    Saves results in Excel file and generates visualizations
    """
    logger.info("Program started")
    new_neg_samples = config["RUNS"].getboolean("del_neg_edges")
    runs = config["RUNS"].getint("number")
    n_iter = config["RUNS"].getint("iter")
    # Select parameters
    rank = config["RUNS"].getint("rank")
    seed = 42
    step_size = 1

    hits_overall1 = []
    hits_overall2 = []
    hits_overall3 = []
    first_run = True

    for a in range(1, runs + 1):
        if new_neg_samples:
            ut.delete_precomp_files()

        ## Get or Generate Data for Training
        # Type should be train, valid or test
        X_train, y_train, groups = get_data(typ=f"train")
        # X_valid, y_valid = get_daata(name=f'valid_{d_name}')
        X_test, y_test, _ = get_data(typ=f"test")

        start_time = time.time()

        # Initalize methods
        if als_part:
            fm = als.FMRegression(n_iter=0, rank=rank, random_state=seed)
            # initalize coefs
            fm.fit(X_train, y_train)
        # if sgd_part:
        #    fm2 = sgd.FMRegression(n_iter=0,rank=rank, random_state=seed)
        #    fm2.fit(X_train, y_train)
        if mcmc_part:
            fm3 = mcmc.FMRegression(n_iter=0, rank=rank, random_state=seed)
            fm3.fit_predict(X_train, y_train, X_test)

        rmse_train1 = []
        rmse_test1 = []
        rmse_train2 = []
        rmse_test2 = []
        rmse_train3 = []
        rmse_test3 = []
        pred1 = []
        pred2 = []
        pred3 = []
        hits1 = []
        hits2 = []
        hits3 = []
        for i in tqdm(range(1, int(n_iter / step_size) + 1), desc="Training Step"):
            if als_part:
                fm.fit(X_train, y_train, n_more_iter=step_size)
                y_pred = fm.predict(X_test)
                pred1.append(evaluate([y_pred], y_test))
                rmse_train1.append(
                    np.sqrt(mean_squared_error(fm.predict(X_train), y_train))
                )
                rmse_test1.append(
                    np.sqrt(mean_squared_error(fm.predict(X_test), y_test))
                )
                hits1.append(evaluate([y_pred], y_test))
            if sgd_part:
                fm2 = sgd.FMRegression(n_iter=n_iter, rank=8, random_state=seed)
                fm2.fit(X_train, y_train)
                y_pred2 = fm2.predict(X_test)
                pred2.append(evaluate([y_pred2], y_test))
                a = fm2.predict(X_train)
                rmse_train2.append(
                    np.sqrt(mean_squared_error(fm2.predict(X_train), y_train))
                )
                rmse_test2.append(
                    np.sqrt(mean_squared_error(fm2.predict(X_test), y_test))
                )
                hits2.append(evaluate([y_pred2], y_test))
            if mcmc_part:
                y_pred3 = fm3.fit_predict(
                    X_train, y_train, X_test, n_more_iter=step_size
                )
                pred3.append(evaluate([y_pred3], y_test))
                rmse_train3.append(
                    np.sqrt(mean_squared_error(fm3.predict(X_train), y_train))
                )
                rmse_test3.append(
                    np.sqrt(mean_squared_error(fm3.predict(X_test), y_test))
                )
                hits3.append(evaluate([y_pred3], y_test))

        hits_matrix = []
        rmse_matrix = []
        labels = []
        x = np.arange(1, n_iter + 1) * step_size
        if als_part:
            hits_overall1.append(hits1[-1])
            if first_run:
                visualize_single(x, hits1, rmse_train1, rmse_test1, "ALS")
                hits_matrix.append(hits1)
                rmse_matrix.append(rmse_train1)
                labels.append(f"ALS rank:{rank} n_iter:{n_iter}")
        if sgd_part:
            hits_overall2.append(hits2[-1])
            if first_run:
                visualize_single(x, hits2, rmse_train2, rmse_test2, "SGD")
                hits_matrix.append(hits2)
                rmse_matrix.append(rmse_train2)
                labels.append(f"SGD rank:{rank} n_iter:{n_iter}")
        if mcmc_part:
            hits_overall3.append(hits3[-1])
            if first_run:
                visualize_single(x, hits3, rmse_train3, rmse_test3, "MCMC")
                hits_matrix.append(hits3)
                rmse_matrix.append(rmse_train3)
                labels.append(f"MCMC rank:{rank} n_iter:{n_iter}")

        if first_run:
            visualize_n(x, hits_matrix, rmse_matrix, labels)

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(
            f"Method FMRegressor took {elapsed_time} to run for {n_iter} iterations"
        )

        logger.info(f"Finished run {a}")
        first_run = False

    if als_part:
        command = f"ALS rank:{rank} n_iter:{n_iter} runs: {runs}"
        save_in_excel(hits_overall1, command)
    if sgd_part:
        command = f"SGD rank:{rank} n_iter:{n_iter} runs: {runs}"
        save_in_excel(hits_overall1, command)

    if mcmc_part:
        command = f"SGD rank:{rank} n_iter:{n_iter} runs: {runs}"
        save_in_excel(hits_overall1, command)

    logger.info("Programm finished")
    return


def save_in_excel(hits: list, command: str) -> None:
    """
    Calculates metrics before storing it in excel file
    """
    average = stats.mean(hits)
    if len(hits) > 1:
        stddev = stats.stdev(hits)
    else:
        stddev = 0
    rand = ut.positive_in_top_k_prob()
    save_pred(command, average, stddev, rand)
    return


## Myfm doesnt support step wise fitting.
## Therefore visualization of process is not possible
def visualize_single(x, hits, rmse, rmse_test, method):
    plt.clf()
    parts = ut.active_abbrs()
    d_name = config["STANDARD"]["graph_name"]
    n_iter = max(x)

    fig, ax1 = plt.subplots()

    with plt.style.context("fivethirtyeight"):
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Hits@")
        line1 = ax1.plot(x, hits, label="Hits@")
        ax1.tick_params(axis="y")

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.set_ylabel("RMSE")  # we already handled the x-label with ax1
        line2 = ax2.plot(x, rmse, label="RMSE train", ls="--")
        line3 = ax2.plot(x, rmse_test, label="RMSE test", ls=":")
        ax2.tick_params(axis="y")

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    plt.title(f"{method} {d_name}")
    plt.tight_layout()
    # Create a Path object for the directory
    directory = Path(f"{vis_folder}")
    # If the directory does not exist, create it
    if not directory.exists():
        directory.mkdir(parents=True)
    # Use Path to create the filename
    filename = Path(f"rmse_hits_{d_name}_{parts}_{n_iter}.png")
    # Combine the directory path and filename to get the full file path
    file_path = Path(directory, filename)
    plt.savefig(f"{file_path}")
    # plt.show()
    return


def visualize_n(x, hits, rmse, labels):
    plt.clf()

    parts = ut.active_abbrs()
    d_name = config["STANDARD"]["graph_name"]
    n_iter = max(x)
    fig, ax1 = plt.subplots()

    with plt.style.context("fivethirtyeight"):
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Hits@")

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        lines = []
        for i in range(len(labels)):
            l1 = f"Hits@ {labels[i]}"
            (line,) = ax1.plot(x, hits[i], label=l1, color=color_cycle[i], ls="--")
            lines.append(line)

        ax2 = ax1.twinx()
        ax2.set_ylabel("RMSE")

        for i in range(len(labels)):
            l2 = f"RMSE {labels[i]}"
            (line,) = ax2.plot(
                x, rmse[i], label=l2, color=color_cycle[i + len(labels)], ls="--"
            )
            lines.append(line)

        labels_total = [f"Hits@ {label}" for label in labels] + [
            f"RMSE {label}" for label in labels
        ]
        ax1.legend(
            lines,
            labels_total,
            loc="center right",
            bbox_to_anchor=(1, 0.5),
            fontsize="xx-small",
        )
        fig.tight_layout()

    directory = Path(f"{vis_folder}")
    if not directory.exists():
        directory.mkdir(parents=True)

    filename = Path(f"rmse_hits_{d_name}_{parts}_{n_iter}_comb.png")
    file_path = Path(directory, filename)

    plt.savefig(f"{file_path}")

    return


def evaluate(preds: list, y_test: np.ndarray) -> float:
    """
    Calculate OGB Metric for specified graph.
    This can be Hits@X for example.
    Takes in predictions of model and groundtruth
    """
    for y_pred in preds:
        y_pred_neg = []
        y_pred_pos = []
        for i, val in enumerate(y_test):
            if val == 0:
                y_pred_neg.append(float(y_pred[i]))
            elif val > 0:
                y_pred_pos.append(float(y_pred[i]))
            else:
                raise ValueError(
                    f"The test data is not in the correct format. Value < 0 : {val}."
                )
        y_pred_pos = np.float_(y_pred_pos)
        y_pred_neg = np.float_(y_pred_neg)
        input_dict = {"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg}
        evaluator = Evaluator(name=config["STANDARD"]["graph_name"])
        result_dict = evaluator.eval(input_dict)
    return list(result_dict.values())[0]


if __name__ == "__main__":
    main()
