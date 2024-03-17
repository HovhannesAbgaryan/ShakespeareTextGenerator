import os.path

import pandas as pd


def preprocess_data(input_file, output_folder):
    # Read data
    data = pd.read_csv(input_file)

    # Fill missing values and concatenate 'Player' with 'PlayerLine'
    data['Player'] = data['Player'].fillna("")
    data['Player'] = data['Player'].apply(lambda x: x + ":" + "\n" if x != "" else "")
    data['Text'] = data['Player'] + data['PlayerLine']

    # Split data into train, validation, and test sets
    train_size = int(0.7 * len(data))
    valid_size = int(0.8 * len(data))
    test_size = len(data)
    train_data = data.iloc[:train_size]
    valid_data = data.iloc[train_size:valid_size]
    test_data = data.iloc[valid_size:]

    # Convert data to text format
    train_data_text = "\n".join(train_data['Text'].tolist())
    valid_data_text = "\n".join(valid_data['Text'].tolist())
    test_data_text = "\n".join(test_data['Text'].tolist())

    # Write data to files
    with open(os.path.join(output_folder, "shakespeare_train.txt"), "w") as file:
        file.write(train_data_text)
    with open(os.path.join(output_folder, "shakespeare_valid.txt"), "w") as file:
        file.write(valid_data_text)
    with open(os.path.join(output_folder, "shakespeare_test.txt"), "w") as file:
        file.write(test_data_text)


if __name__ == "__main__":
    preprocess_data("data/shakespeare_data/raw/Shakespeare_data.csv",
                    "data/shakespeare_data/processed"
                   )
