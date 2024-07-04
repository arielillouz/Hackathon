from argparse import ArgumentParser
import logging
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import os


def loss_over_percentage(train_data: pd.DataFrame, y_train: pd.Series, test_samples: pd.DataFrame, y_test: pd.Series):
    model = LinearRegression()
    loss = np.zeros((91, 10))

    for p in range(10, 101):
        print(f"p: {p}")
        for i in range(10):
            processed_samples = train_data.sample(frac=p / 100, axis="index", random_state=50 + i)
            responses = y_train.loc[processed_samples.index]
            model.fit(processed_samples, responses)
            predictions = model.predict(test_samples)
            loss_res = mean_squared_error(y_test, predictions)
            loss[p - 10, i] = loss_res

    loss_over_percentage_plots(loss)


def loss_over_percentage_plots(loss: np.ndarray):
    mean_loss = np.mean(loss, axis=1)
    std_loss = np.std(loss, axis=1)
    p_array = np.arange(10, 101)

    plt.figure(figsize=(10, 6))
    plt.plot(p_array, mean_loss, label="Mean Loss", color="green", linestyle="--", marker="o")
    plt.fill_between(p_array, mean_loss - 2 * std_loss, mean_loss + 2 * std_loss, color="lightgrey", alpha=0.5)

    plt.title("Loss vs Sample Percentage")
    plt.xlabel("Percentage of Training Data Used")
    plt.ylabel("Mean Loss with Confidence Interval")
    plt.legend()
    plt.grid(True)

    # plot_filename = "Loss_vs_Sample_Percentage.png"
    # plt.savefig(plot_filename)
    # plt.close()
    plt.show()
