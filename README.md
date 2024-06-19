
# Credit Card Fraud Detection

This repository contains a analysis and model development for detecting credit card fraud using machine learning techniques. The primary objective is to build a robust model that can accurately identify fraudulent transactions from a highly imbalanced dataset.

## Dataset

The dataset used in this project is from [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/shayannaveed/credit-card-fraud-detection/data). It contains credit card transactions made by European cardholders in September 2013. The dataset is highly imbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.

## Project Structure

- **data/**: Directory containing the dataset files.
- **results/**: Directory where results from the model runs are saved.
- **config.json**: Configuration file for setting parameters for the model runs.
- **main.py**: Main script for running the model and evaluations.
- **README.md**: Project documentation.

## Getting Started

### Prerequisites

Ensure you have the following packages installed:

- `pandas`
- `numpy`
- `scikit-learn`
- `imblearn`

You can install the required packages using the following command:

```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

### Downloading the Dataset

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/shayannaveed/credit-card-fraud-detection/data) and place it in the `data/` directory.

### Configuration

Adjust the parameters in the `config.json` file as needed:

```json
{
  "num_runs": 10,
  "test_size": 0.2,
  "random_state": 42,
  "n_estimators": 100,
  "max_iter": 2000
}
```

- **num_runs**: Number of times the model should be run.
- **test_size**: Proportion of the dataset to include in the test split.
- **random_state**: Controls the shuffling applied to the data before applying the split.
- **n_estimators**: Number of trees in the forest for the Random Forest model.
- **max_iter**: Maximum number of iterations for the model training.

### Running the Model

Execute the main script to run the model and save the results:

```bash
python main.py
```

The script will perform the following steps:

1. Load and preprocess the dataset.
2. Handle class imbalance using SMOTE.
3. Split the data into training and testing sets.
4. Train a Random Forest model.
5. Evaluate the model on both training and test data.
6. Run the experiment multiple times as specified in `config.json`.
7. Save metrics for each run and the average metrics in the `results/` directory.

### Results

The results of each run and the average metrics will be saved in the `results/` directory. Each run will have its own subfolder containing the following files:

- `train_metrics.txt`: Metrics on the training data.
- `test_metrics.txt`: Metrics on the test data.
- `average_metrics.txt`: Average metrics across all runs.
   
Since the training and testing data are selected randomly for each run, the results are not deterministic across multiple runs. However, the average statistics over 10 runs are listed below:     

--- Average Training Metrics ---     
Accuracy: 0.99     
Precision: 0.97     
Recall: 0.99     

--- Average Testing Metrics ---    
Accuracy: 0.99     
Precision: 0.98    
Recall: 1.0      
## Example Output

An example of the average metrics output:

```text
--- Average Training Metrics ---
Accuracy: 0.9995
Precision: 0.95
Recall: 0.90
Confusion Matrix:
[[56962     1]
 [    2   100]]

--- Average Testing Metrics ---
Accuracy: 0.9994
Precision: 0.94
Recall: 0.89
Confusion Matrix:
[[56952    11]
 [   12    90]]
```

## Contributing

Feel free to submit issues and enhancement requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Dataset: [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/shayannaveed/credit-card-fraud-detection/data)
- Scikit-learn: Machine Learning library
- Imbalanced-learn: Handling imbalanced datasets in machine learning
