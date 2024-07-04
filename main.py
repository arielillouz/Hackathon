from argparse import ArgumentParser
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
