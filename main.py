import argparse
import os.path as path
import pandas as pd
from spotifymoods import train, predict


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", help="Train model with given .csv file")
    parser.add_argument("-p", "--predict", help="Predict data")
    args = parser.parse_args()

    if args.train:
        print("Training the model with data from: % s" % args.train)
        train_data_path = args.train
        if path.isfile(train_data_path) and train_data_path.endswith('.csv'):
            train_data = pd.read_csv(train_data_path)
            train(train_data, trained_output="trained.pkl", scaled_output="scaled.pkl")
        else:
            print(f"The file {train_data_path} is not valid!")
            exit(1)

    elif args.predict:
        print("Predicting the results of the data from: % s" % args.predict)
        test_data_path = args.predict
        if path.isfile(test_data_path) and test_data_path.endswith('.csv'):
            test_data = pd.read_csv(test_data_path)
            result = predict(test_data, trained_path="trained.pkl", scaled_path="scaled.pkl")
            result.to_csv('result.csv', index=False)
        else:
            print(f"The file {test_data_path} is not valid!")
            exit(1)

    else:
        print("No arguments was provided, expected '--train' or '--predict'")
        exit(1)


if __name__ == "__main__":
    run()
