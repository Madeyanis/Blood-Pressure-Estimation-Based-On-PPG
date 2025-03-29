# Blood Pressure Estimation Based on PPG using Deep learning

This project aims to estimate blood pressure using photoplethysmogram (PPG) signals. It leverages signal processing and machine learning techniques to provide accurate and non-invasive blood pressure measurements.

## Features

- **PPG Signal Processing**: Extract meaningful features from raw PPG data.
- **Machine Learning Models**: Train and evaluate models for blood pressure estimation.
- **Visualization**: Plot and analyze PPG signals and results.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Madeyanis/Blood-Pressure-Estimation-Based-On-PPG.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Blood-Pressure-Estimation-Based-On-PPG
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Preprocess the PPG data:
    ```bash
    python preprocess.py --input data/raw_ppg.csv --output data/processed_ppg.csv
    ```
2. Train the model:
    ```bash
    python train.py --data data/processed_ppg.csv
    ```
3. Evaluate the model:
    ```bash
    python evaluate.py --model models/trained_model.pkl --test data/test_ppg.csv
    ```

## Acknowledgments

- Thanks to the open-source community for tools and libraries.
- Special thanks to researchers in the field of PPG signal analysis.
