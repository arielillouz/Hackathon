from argparse import ArgumentParser
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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


def preprocess_data(data_frame: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with any missing values and duplicates
    #data_frame.dropna(inplace=True)
    data_frame.drop_duplicates(inplace=True)

    # Ensure 'passengers_continue' is non-negative integers
    data_frame['passengers_continue'] = data_frame['passengers_continue'].apply(
        lambda x: abs(x) if x < 0 else x).round().astype(int)

    # Convert 'arrival_is_estimated' to lowercase string and then map to 0 and 1
    #data_frame['arrival_is_estimated'] = data_frame['arrival_is_estimated'].astype(str).str.lower().map({'false': 0, 'true': 1})
    data_frame['arrival_is_estimated'] = data_frame[
        'arrival_is_estimated'].astype(str)
    data_frame['arrival_is_estimated'] = data_frame[
        'arrival_is_estimated'].str.lower().map({'false': 0, 'true': 1})
    # Convert 'arrival_time' to datetime and extract hour and minute
    data_frame['arrival_time'] = pd.to_datetime(data_frame['arrival_time'], format='%H:%M:%S', errors='coerce')
    data_frame['arrival_hour'] = data_frame['arrival_time'].dt.hour

    return data_frame


def calculate_mean(df, column):
    return df[column].mean()


def create_baseline_predictions(test_data, mean_value, id_column,
                                prediction_column):
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

    # Load and preprocess the training set
    logging.info("Loading training set...")
    train_data = load_data(args.training_set)

    # Sample 5% of the data for the baseline
    sampled_data = train_data.sample(frac=0.05, random_state=42)

    # 2. preprocess the training set
    logging.info("preprocessing train...")
    sampled_data = preprocess_data(sampled_data)

    # Define features and target
    features = ['arrival_hour', 'arrival_is_estimated', 'passengers_continue']
    target = 'passengers_up'

    # Split preprocessed data into 80% training+validation and 20% test
    train_val_data, test_data = train_test_split(sampled_data, test_size=0.2,
                                                 random_state=42)  # add stratify=sampled_data[target]?

    # Split training+validation into 70% training and 30% validation
    train_data, val_data = train_test_split(train_val_data, test_size=0.25,
                                            random_state=42)  # 0.25 * 80% = 20%

    # Split the data into training and validation sets
    X_train, X_val = train_data[features], val_data[features]
    y_train, y_val = train_data[target], val_data[target]

    # Calculate correlations
    correlations = {}
    for feature in features:
        correlation = np.corrcoef(train_data[feature], train_data[target])[
            0, 1]
        correlations[feature] = correlation

    # Plotting correlations
    plt.figure(figsize=(16, 6))
    for i, feature in enumerate(features, 1):
        plt.subplot(1, len(features), i)
        plt.scatter(train_data[feature], train_data[target], alpha=0.5)
        plt.xlabel(feature)
        plt.ylabel('Passengers Up')
        plt.title(
            f'{feature} vs. Passengers Up\nCorrelation: {correlations[feature]:.2f}')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Train a linear regression model
    logging.info("Training linear regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model on the validation set
    y_val_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)
    logging.info(f"Validation Mean Squared Error: {mse}")
    print(f"Validation Mean Squared Error: {mse}")

    # Load and preprocess the test set
    logging.info("Loading test set...")
    test_data = load_data(args.test_set)
    logging.info("Preprocessing test set...")
    test_data = preprocess_data(test_data)

    # Make predictions on the test set
    logging.info("Making predictions on the test set...")
    X_test = test_data[features]
    test_data['passengers_up'] = model.predict(X_test)

    # Round predictions to nearest positive integer
    test_data['passengers_up'] = np.round(test_data['passengers_up']).astype(
        int)
    test_data['passengers_up'] = test_data['passengers_up'].apply(
        lambda x: max(x, 0))

    # Save the predictions
    logging.info(f"Saving predictions to {args.out}...")
    predictions = test_data[['trip_id_unique_station', 'passengers_up']]
    save_predictions(predictions, args.out)
