# Shakespeare Text Generation

This project aims to generate Shakespearean-style text using a fine-tuned GPT-2 model. It includes data preprocessing, model training, and text generation functionalities.

## Description

The project consists of the following components:

- **Data Preprocessing**: Raw Shakespearean text data is preprocessed to prepare it for training the GPT-2 model.
- **Model Training**: The preprocessed data is used to train a GPT-2 model, which is fine-tuned to generate Shakespearean-style text.
- **Text Generation**: The trained model is used to generate text in the style of Shakespeare.

## How to Run

1. Clone the repository:

    ```bash
    git clone https://github.com/HovhannesAbgaryan/ShakespeareTextGenerator
    cd shakespeare-text-generation
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the main script with optional arguments:

    ```bash
    python main.py --data_file path/to/data.csv --model_save_path path/to/save/model.pth --output_folder path/to/save/preprocessed/data/
    ```

    Optional Arguments:
    - `--data_file`: Path to the raw data file. Default: `"data/shakespeare_data/raw/Shakespeare_data.csv"`.
    - `--model_save_path`: Path to save the trained model. Default: `"models/best_shakespeare_model.pth"`.
    - `--output_folder`: Folder to save preprocessed data. Default: `"data/shakespeare_data/processed/"`.

4. Once the script finishes, it will generate Shakespearean-style text.
