from argparse import ArgumentParser
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

"""
usage:
    python code/main.py --training_set PATH --test_set PATH --out PATH

for example:
    python code/main.py --training_set /cs/usr/gililior/training.csv --test_set /cs/usr/gililior/test.csv --out predictions/trip_duration_predictions.csv

"""


# implement here your load,preprocess,train,predict,save functions (or any
# other design you choose)


def load_data(path):
    return pd.read_csv(path, encoding='ISO-8859-8')


def preprocess_train(df):
    # trip_durations = df.groupby('trip_id_unique')['arrival_time'].agg(
    #     ['min', 'max'])
    # trip_durations['trip_duration_in_minutes'] = (trip_durations['max'] -
    #                                               trip_durations[
    #                                                   'min']).dt.total_seconds() / 60
    # trip_durations = preprocess_test(trip_durations)
    # return trip_durations

    # Ensure 'passengers_continue' is non-negative integers
    if 'passengers_continue' in df.columns:
        df['passengers_continue'] = df['passengers_continue'].apply(
            lambda x: abs(x) if x < 0 else x).round().astype(int)

    # Process 'passengers_continue' column to create 'alternative' column
    df['alternative'] = df['alternative'].apply(lambda x: 1 if x == '#' else 0)

    df['direction'] = df['direction'] - 1

    df['cluster'] = df['cluster'].astype('category').cat.codes

    # Convert arrival_time to datetime
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S')

    # Calculate trip duration in minutes
    trip_durations = df.groupby('trip_id_unique')['arrival_time'].agg(['min', 'max'])
    trip_durations['trip_duration_in_minutes'] = (trip_durations['max'] -
                                                  trip_durations[
                                                      'min']).dt.total_seconds() / 60

    # Calculate the average of passengers up and passengers continuing
    trip_durations['avg_passengers_up'] = df.groupby('trip_id_unique')[
        'passengers_up'].mean()
    trip_durations['avg_passengers_continue'] = df.groupby('trip_id_unique')[
        'passengers_continue'].mean()

    # Extract direction and alternative information
    trip_durations['direction'] = df.groupby('trip_id_unique')['direction'].first()
    trip_durations['alternative'] = df.groupby('trip_id_unique')[
        'alternative'].first()
    trip_durations['cluster'] = df.groupby('trip_id_unique')[
        'cluster'].first()


    # Reset index to convert the groupby index (trip_id) into a column
    trip_durations.reset_index(inplace=True)

    # Select desired columns for the new table
    trip_durations = trip_durations[
        ['trip_id_unique', 'trip_duration_in_minutes', 'direction',
         'alternative', 'cluster', 'avg_passengers_up', 'avg_passengers_continue']]
    trip_durations.index = trip_durations['trip_id_unique']
    trip_durations.drop('trip_id_unique', inplace=True, axis=1)

    return trip_durations


def preprocess_test(df):
    # Ensure 'passengers_continue' is non-negative integers
    if 'passengers_continue' in df.columns:
        df['passengers_continue'] = df['passengers_continue'].apply(
            lambda x: abs(x) if x < 0 else x).round().astype(int)

    # Process 'passengers_continue' column to create 'alternative' column
    df['alternative'] = df['alternative'].apply(lambda x: 1 if x == '#' else 0)

    df['direction'] = df['direction'] - 1

    df['cluster'] = df['cluster'].astype('category').cat.codes

    # Calculate trip duration in minutes
    #trip_durations = df.groupby('trip_id_unique')['arrival_time'].first()
    ##trip_durations['trip_duration_in_minutes'] = (trip_durations['max'] -
                                                  #trip_durations[
                                                   #   'min']).dt.total_seconds() / 60

    # # Calculate the average of passengers up and passengers continuing
    # trip_durations['avg_passengers_up'] = df.groupby('trip_id_unique')[
    #     'passengers_up'].mean()
    # trip_durations['avg_passengers_continue'] = df.groupby('trip_id_unique')[
    #     'passengers_continue'].mean()
    #
    # # Extract direction and alternative information
    # trip_durations['direction'] = df.groupby('trip_id_unique')[
    #     'direction'].first()
    # trip_durations['alternative'] = df.groupby('trip_id_unique')[
    #     'alternative'].first()
    #
    # # Reset index to convert the groupby index (trip_id) into a column
    # trip_durations.reset_index(inplace=True)

    # Select desired columns for the new table
    trip_durations = df[
        ['trip_id_unique', 'direction',
         'alternative', 'cluster', 'passengers_up', 'passengers_continue']]

    # Calculate the average of passengers up and passengers continuing
    merged = trip_durations.groupby('trip_id_unique').agg({
         'passengers_up': 'mean',
         'passengers_continue': 'mean',
         'direction': 'first',
         'alternative': 'first',
         'cluster': 'first'}).rename(columns={
         'passengers_up': 'avg_passengers_up',
         'passengers_continue': 'avg_passengers_continue'}).reset_index()

    merged.index = merged['trip_id_unique']
    merged.drop('trip_id_unique', inplace=True, axis=1)
    return merged


def train_model(X_train, y_train):
    """
    Train a linear regression model.

    Parameters:
    - X_train (pd.DataFrame): Features for training.
    - y_train (pd.Series): Target variable for training.

    Returns:
    - sklearn.linear_model.LinearRegression: Trained linear regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """
    Predict using a trained model.

    Parameters:
    - model (sklearn.linear_model.LinearRegression): Trained linear regression model.
    - X_test (pd.DataFrame): Features for testing.

    Returns:
    - np.ndarray: Predicted values.
    """
    return model.predict(X_test)


def save_predictions(predictions, output_path, test_data):
    """
    Save predictions to a CSV file.

    Parameters:
    - predictions (np.ndarray): Predicted values.
    - output_path (str): Path to save the CSV file.
    """
    test_data['trip_id_unique'] = test_data.index
    output_df = pd.DataFrame({
        'trip_id_unique': test_data['trip_id_unique'],
        # Assuming 'trip_id_unique' exists in test_data
        'trip_duration_in_minutes': predictions
    })
    output_df.to_csv(output_path, index=False)


def loss_over_percentage(train_data: pd.DataFrame, y_train: pd.Series,
                         test_samples: pd.DataFrame, y_test: pd.Series):
    model = LinearRegression()
    loss = np.zeros((91, 10))

    for p in range(10, 101):
        print(f"p: {p}")
        for i in range(10):
            processed_samples = train_data.sample(frac=p / 100,
                                                  axis="index",
                                                  random_state=50 + i)
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
    plt.plot(p_array, mean_loss, label="Mean Loss", color="green",
             linestyle="--", marker="o")
    plt.fill_between(p_array, mean_loss - 2 * std_loss,
                     mean_loss + 2 * std_loss, color="lightgrey",
                     alpha=0.5)

    plt.title("Loss vs Sample Percentage")
    plt.xlabel("Percentage of Training Data Used")
    plt.ylabel("Mean Loss with Confidence Interval")
    plt.legend()
    plt.grid(True)

    # plot_filename = "Loss_vs_Sample_Percentage.png"
    # plt.savefig(plot_filename)
    # plt.close()
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True,
                        help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True,
                        help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    args = parser.parse_args()

    # Load training set
    logging.info(f"Loading training set from {args.training_set}...")
    train_data = load_data(args.training_set)

    # Preprocess training set
    logging.info("Preprocessing training set...")
    train_data = preprocess_train(train_data)

    # Preprocess test set to calculate mean_boarding for adjustment
    logging.info("Preprocessing test set for mean boarding calculation...")
    real_test_data = load_data(args.test_set)
    real_test_data = preprocess_test(real_test_data)

    # Split preprocessed data into 80% training+validation and 20% test
    train_val_data, test_data = train_test_split(train_data, test_size=0.2,
                                                 random_state=42)  # add stratify=sampled_data[target]?

    # Split training+validation into 70% training and 30% validation
    train_data, val_data = train_test_split(train_val_data, test_size=0.25,
                                            random_state=42)  # 0.25 * 80% = 20%

    # for the case of testing using train set
    # true_values = test_data["trip_duration_in_minutes"]

    # Separate features and target variable for training
    X_train = train_data.drop(columns=['trip_duration_in_minutes'])
    y_train = train_data['trip_duration_in_minutes']

    # Train the model
    logging.info("Training model...")
    model = train_model(X_train, y_train)
    #
    # Prepare test data for prediction for the case of testing using train
    X_test = test_data.drop(columns=['trip_duration_in_minutes'])

    # Make predictions
    logging.info("Making predictions...")
    predictions = predict(model, real_test_data)

    # make predictions using train (for loss caculation)
    # predictions = predict(model, X_test)
    # mse = mean_squared_error(true_values, predictions)
    # print(mse)
    # #plot
    # loss_over_percentage(X_train, y_train, X_test, true_values)

    # Save predictions
    logging.info(f"Saving predictions to {args.out}...")
    save_predictions(predictions, args.out, real_test_data)
