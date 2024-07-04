from argparse import ArgumentParser
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from subtask1_plots import loss_over_percentage

"""
usage:
    python code/main.py --training_set PATH --test_set PATH --out PATH

for example:
    python code/main.py --training_set /cs/usr/gililior/training.csv --test_set /cs/usr/gililior/test.csv --out predictions/trip_duration_predictions.csv 

"""


# implement here your load,preprocess,train,predict,save functions (or any other design you choose)
def load_data(path):
    return pd.read_csv(path, encoding='ISO-8859-8')


def preprocess_train(df):
    # Ensure 'passengers_continue' is non-negative integers
    if 'passengers_continue' in df.columns:
        df['passengers_continue'] = df['passengers_continue'].apply(
            lambda x: abs(x) if x < 0 else x).round().astype(int)

    # Convert 'arrival_is_estimated' to 0 and 1
    df['arrival_is_estimated'] = df['arrival_is_estimated'].astype(str).map({'False': 0, 'True': 1})

    # Handle missing door_closing_time by setting it to arrival_time (time difference will be zero)
    df['door_closing_time'].fillna(df['arrival_time'], inplace=True)

    df['direction'] = df['direction'] - 1

    # Convert time columns to datetime
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S', errors='coerce')
    df['door_closing_time'] = pd.to_datetime(df['door_closing_time'], format='%H:%M:%S', errors='coerce')

    # Feature engineering
    df['arrival_hour'] = df['arrival_time'].dt.hour
    df['arrival_minute'] = df['arrival_time'].dt.minute

    # Create 'door_open' feature
    df['door_open'] = (df['door_closing_time'] - df['arrival_time']).dt.total_seconds().astype(int)

    # Calculate mean open door time (excluding zero times)
    mean_open_door_time = df.loc[df['door_open'] > 0, 'door_open'].mean()

    # Adjust 'door_open' based on boarding passengers and door_open time
    df['door_open'] = df.apply(
        lambda row: (mean_open_door_time / row['passengers_up']) if (row['door_open'] <= 0 < row['passengers_up']) else
        row['door_open'], axis=1)

    df = df.drop(
        ["station_name", "part", "trip_id_unique", "arrival_time", "door_closing_time"],
        axis=1)

    # Process 'passengers_continue' column to create 'alternative' column
    df['alternative'] = df['alternative'].apply(lambda x: 1 if x == '#' else 0)

    # Convert 'cluster' column to one-hot encoding (dummy variables)
    if 'cluster' in df.columns:
        df['cluster'] = df['cluster'].astype('category').cat.codes
        df = pd.get_dummies(df, columns=['cluster'], prefix='cluster')

        # Ensure the dummy variables are integers
        dummy_columns = [col for col in df.columns if col.startswith("cluster_")]
        df[dummy_columns] = df[dummy_columns].astype(int)

        # Create 'rush_hour' feature
    df['rush_hour'] = df['arrival_hour'].apply(lambda x: 1 if (8 <= x < 10 or 16 <= x < 19) else 0)

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    return df


def preprocess_test(df, mean_boarding: float):
    # Ensure 'passengers_continue' is non-negative integers
    if 'passengers_continue' in df.columns:
        df['passengers_continue'] = df['passengers_continue'].apply(
            lambda x: abs(x) if x < 0 else x).round().astype(int)

    # Convert 'arrival_is_estimated' to 0 and 1
    df['arrival_is_estimated'] = df['arrival_is_estimated'].astype(str).map({'False': 0, 'True': 1})

    # Handle missing door_closing_time by setting it to arrival_time (time difference will be zero)
    df['door_closing_time'].fillna(df['arrival_time'], inplace=True)

    df['direction'] = df['direction'] - 1

    # Convert time columns to datetime
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S', errors='coerce')
    df['door_closing_time'] = pd.to_datetime(df['door_closing_time'], format='%H:%M:%S', errors='coerce')

    # Feature engineering
    df['arrival_hour'] = df['arrival_time'].dt.hour
    df['arrival_minute'] = df['arrival_time'].dt.minute

    # Create 'door_open' feature
    df['door_open'] = (df['door_closing_time'] - df['arrival_time']).dt.total_seconds().astype(int)

    # Calculate mean open door time (excluding zero times)
    mean_open_door_time = df.loc[df['door_open'] > 0, 'door_open'].mean()

    # Adjust 'door_open' based on boarding passengers and door_open time
    df['door_open'] = df.apply(
        lambda row: (mean_open_door_time / mean_boarding) if (row['door_open'] <= 0 < mean_boarding) else row[
            'door_open'],
        axis=1
    )
    df = df.drop(
        ["station_name", "part", "trip_id_unique", "arrival_time", "door_closing_time"],
        axis=1)

    # Process 'passengers_continue' column to create 'alternative' column
    df['alternative'] = df['alternative'].apply(lambda x: 1 if x == '#' else 0)

    # Convert 'cluster' column to one-hot encoding (dummy variables)
    if 'cluster' in df.columns:
        df['cluster'] = df['cluster'].astype('category').cat.codes
        df = pd.get_dummies(df, columns=['cluster'], prefix='cluster')

        # Ensure the dummy variables are integers
        dummy_columns = [col for col in df.columns if col.startswith("cluster_")]
        df[dummy_columns] = df[dummy_columns].astype(int)

        # Create 'rush_hour' feature
    df['rush_hour'] = df['arrival_hour'].apply(lambda x: 1 if (8 <= x < 10 or 16 <= x < 19) else 0)

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    return df


def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    prediction = model.predict(X_test).round()
    prediction[prediction < 0] = 0
    return prediction


def save_predictions(predictions, ids, path):
    output = pd.DataFrame({'trip_id_unique_station': ids, 'passengers_up': predictions})
    output.to_csv(path, index=False)


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

    # Split the data into 90% training+validation and 10% test
    train_val_data, test_data = train_test_split(train_data, test_size=0.1, random_state=42)

    # Split training+validation into 70% training and 20% validation
    train_data, val_data = train_test_split(train_val_data, test_size=2 / 9,
                                            random_state=42)  # 2/9 because 20% of 90% is 2/9

    # Extract and preserve IDs
    train_ids = train_data['trip_id_unique_station']
    val_ids = val_data['trip_id_unique_station']
    test_ids = test_data['trip_id_unique_station']

    # Drop 'trip_id_unique_station' before preprocessing
    train_data = train_data.drop(columns=['trip_id_unique_station'])
    val_data = val_data.drop(columns=['trip_id_unique_station'])
    test_data = test_data.drop(columns=['trip_id_unique_station'])

    # 2. preprocess the training set
    logging.info("preprocessing train...")
    train_data = preprocess_train(train_data)

    # 2. preprocess the validation set
    logging.info("preprocessing validation...")
    val_data = preprocess_train(val_data)

    # 2. preprocess the test set
    logging.info("preprocessing test...")
    test_data = preprocess_test(test_data, train_data['passengers_up'].mean())

    # Split the training data into features and target variable
    X_train = train_data.drop(columns=['passengers_up'])
    y_train = train_data['passengers_up']

    # Split the validation data into features and target variable
    X_val = val_data.drop(columns=['passengers_up'])
    y_val = val_data['passengers_up']

    # Split the test data into features
    X_test = test_data.drop('passengers_up', axis=1)
    y_test = test_data['passengers_up']

    # 3. train a model
    logging.info("training...")
    model = train_model(X_train, y_train)

    # Evaluate the model on the validation set
    val_predictions = predict(model, X_val)
    val_mse = mean_squared_error(y_val, val_predictions)
    print(val_mse)
    logging.info(f"Validation Mean Squared Error: {val_mse}")

    # 4. predict the test set using the trained model
    logging.info("predicting...")
    predictions = predict(model, X_test)

    # 5. save the predictions to args.out
    save_predictions(predictions, test_ids, args.out)
    logging.info("predictions saved to {}".format(args.out))

    loss_over_percentage(X_train, y_train, X_test, y_test)
