from argparse import ArgumentParser
import logging
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


# implement here your load,preprocess,train,predict,save functions (or any other design you choose)
def load_data(path):
    return pd.read_csv(path, encoding='ISO-8859-8')


def preprocess_data(df):
    # Handle missing values
    df['door_closing_time'].fillna(df['door_closing_time'].mode()[0], inplace=True)

    # Convert time columns to datetime
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S', errors='coerce')
    df['door_closing_time'] = pd.to_datetime(df['door_closing_time'], format='%H:%M:%S', errors='coerce')

    # Feature engineering
    df['arrival_hour'] = df['arrival_time'].dt.hour
    df['arrival_minute'] = df['arrival_time'].dt.minute

    return df

def calculate_mean(df, column):
    return df[column].mean()


def create_baseline_predictions(test_data, mean_value, id_column, prediction_column):
    predictions = pd.DataFrame({
        id_column: test_data[id_column],
        prediction_column: mean_value
    })
    return predictions


def save_predictions(predictions, path):
    predictions.to_csv(path, index=False)


def plot_histogram(df, column, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 6))
    df[column].plot(kind='hist', bins=50, edgecolor='k', alpha=0.7)
    plt.yscale('log')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_bar(actual, predicted, title, xlabel, ylabel, filename, num_samples=50):
    samples = list(range(min(num_samples, len(actual))))
    plt.figure(figsize=(12, 8))
    width = 0.35  # the width of the bars
    plt.bar(samples, actual[:num_samples], width, label='Actual')
    plt.bar([s + width for s in samples], predicted[:num_samples], width, label='Predicted')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()


def plot_loss_scatter(actual, predicted, title, xlabel, ylabel, filename):
    mse_values = (actual - predicted) ** 2
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(mse_values)), mse_values, alpha=0.7, edgecolor='k')
    plt.yscale('log')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True,
                        help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True,
                        help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    args = parser.parse_args()

    # 1. load the training set (args.training_set)
    train_data = load_data(args.training_set)

    # Sample 5% of the data for the baseline
    sampled_data = train_data.sample(frac=0.05, random_state=42)

    # 2. preprocess the training set
    logging.info("preprocessing train...")
    sampled_data = preprocess_data(sampled_data)

    # Split preprocessed data into 80% training+validation and 20% test
    train_val_data, test_data = train_test_split(sampled_data, test_size=0.2, random_state=42)

    # Split training+validation into 70% training and 30% validation
    train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=42)  # 0.25 * 80% = 20%

    # 3. train a model
    logging.info("training...")
    mean_passengers_up = calculate_mean(train_data, 'passengers_up')

    # Plot the distribution of passengers_up in the training data
    # Plot the distribution of passengers_up in the training data
    plot_histogram(train_data, 'passengers_up', 'Distribution of Passengers Up in Training Data', 'Passengers Up',
                   'Frequency', 'passengers_up_distribution.png')

    # Generate baseline predictions on the validation set
    val_predictions = create_baseline_predictions(val_data, mean_passengers_up, 'trip_id_unique_station',
                                                  'passengers_up')

    # Plot baseline predictions vs actual values using a bar plot
    plot_bar(val_data['passengers_up'], val_predictions['passengers_up'], 'Baseline Predictions vs Actual Values',
             'Samples', 'Passengers Up', 'baseline_vs_actual.png')

    # Plot baseline loss as a scatter plot
    plot_loss_scatter(val_data['passengers_up'], val_predictions['passengers_up'], 'Baseline Loss', 'Sample Index',
                      'Squared Error', 'baseline_loss_scatter.png')

    # Evaluate baseline predictions
    mse = mean_squared_error(val_data['passengers_up'], val_predictions['passengers_up'])
    print(mse)

    # 4. load the test set (args.test_set)
    test_data_full = load_data(args.test_set)

    # 5. preprocess the test set
    logging.info("preprocessing test...")
    test_data_full = preprocess_data(test_data_full)

    # 6. predict the test set using the trained model
    logging.info("predicting...")
    test_baseline_predictions = create_baseline_predictions(test_data_full, mean_passengers_up,
                                                            'trip_id_unique_station', 'passengers_up')

    # 7. save the predictions to args.out
    logging.info("predictions saved to {}".format(args.out))
    save_predictions(test_baseline_predictions, args.out)
