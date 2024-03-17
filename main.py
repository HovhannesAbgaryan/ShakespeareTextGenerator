import argparse
import os

from src.data.preprocess import preprocess_data
from src.models.train import train_model
from src.inference.generate_text import generate_text


def main(args):
    # Preprocessing
    print("Preprocessing data...")
    preprocess_data(args.data_file, args.output_folder)

    # Training
    print("Training model...")
    train_txt = os.path.join(args.output_folder, "shakespeare_train.txt")
    valid_txt = os.path.join(args.output_folder, "shakespeare_valid.txt")
    train_model(train_txt, valid_txt, args.model_save_path)

    # Inference
    print("Generating text...")
    generated_text = generate_text(args.model_save_path)
    print("Generated text:")
    print(generated_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to preprocess data, train model, and generate text.")
    parser.add_argument("--data_file", type=str, default="data/shakespeare_data/raw/Shakespeare_data_small.csv",
                        help="Path to the raw data file.")
    parser.add_argument("--model_save_path", type=str, default="models/best_shakespeare_model.pth",
                        help="Path to save the trained model.")
    parser.add_argument("--output_folder", type=str, default="data/shakespeare_data/processed/",
                        help="Folder to save preprocessed data.")
    args = parser.parse_args()

    main(args)
