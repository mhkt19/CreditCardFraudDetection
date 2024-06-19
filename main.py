import os
import json
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime
import numpy as np

# Load configuration from config.json
with open('config.json', 'r') as f:
    config = json.load(f)

NUM_RUNS = config["num_runs"]
TEST_SIZE = config["test_size"]
RANDOM_STATE = config["random_state"]
N_ESTIMATORS = config["n_estimators"]
MAX_ITER = config["max_iter"]

def log_function_start(function_name):
    print(f'{function_name} started at {time.strftime("%H:%M:%S")}')

def load_data(filepath):
    log_function_start('load_data')
    data = pd.read_csv(filepath)
    return data

def data_preparation(data):
    log_function_start('data_preparation')
    X = data.drop('Class', axis=1)
    y = data['Class']
    return X, y

def handle_imbalance(X, y):
    log_function_start('handle_imbalance')
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def data_splitting(X, y, test_size=TEST_SIZE):
    log_function_start('data_splitting')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    log_function_start('scale_data')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_random_forest(X_train, y_train):
    log_function_start('train_random_forest')
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    log_function_start('evaluate_model')
    
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_conf_matrix = confusion_matrix(y_train, y_train_pred)
    
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_conf_matrix = confusion_matrix(y_test, y_test_pred)
    
    return (train_accuracy, train_precision, train_recall, train_conf_matrix), (test_accuracy, test_precision, test_recall, test_conf_matrix)

def save_metrics(metrics, folder_path, dataset_type):
    log_function_start('save_metrics')
    metrics_file = os.path.join(folder_path, f'{dataset_type}_metrics.txt')
    with open(metrics_file, 'w') as f:
        accuracy, precision, recall, conf_matrix = metrics
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')
        f.write(f'Confusion Matrix:\n{conf_matrix}\n')

def run_experiment():
    filepath = './dataset/creditcard.csv'
    data = load_data(filepath)
    X, y = data_preparation(data)
    X_resampled, y_resampled = handle_imbalance(X, y)
    
    train_accuracies = []
    train_precisions = []
    train_recalls = []
    train_confusion_matrices = []
    
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_confusion_matrices = []
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_folder_path = f'./results/{timestamp}'
    os.makedirs(main_folder_path, exist_ok=True)
    
    for run in range(NUM_RUNS):
        run_folder_path = os.path.join(main_folder_path, f'run_{run+1}')
        os.makedirs(run_folder_path, exist_ok=True)
        X_train, X_test, y_train, y_test = data_splitting(X, y)
        X_train_S, X_test_S, y_train_S, y_test_S = data_splitting(X_resampled, y_resampled)
        
        model = train_random_forest(X_train_S, y_train_S)
        
        train_metrics, test_metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        
        train_accuracies.append(train_metrics[0])
        train_precisions.append(train_metrics[1])
        train_recalls.append(train_metrics[2])
        train_confusion_matrices.append(train_metrics[3])
        
        test_accuracies.append(test_metrics[0])
        test_precisions.append(test_metrics[1])
        test_recalls.append(test_metrics[2])
        test_confusion_matrices.append(test_metrics[3])
        
        save_metrics(train_metrics, run_folder_path, 'train')
        save_metrics(test_metrics, run_folder_path, 'test')
    
    avg_train_accuracy = np.mean(train_accuracies)
    avg_train_precision = np.mean(train_precisions)
    avg_train_recall = np.mean(train_recalls)
    avg_train_conf_matrix = np.mean(train_confusion_matrices, axis=0)
    
    avg_test_accuracy = np.mean(test_accuracies)
    avg_test_precision = np.mean(test_precisions)
    avg_test_recall = np.mean(test_recalls)
    avg_test_conf_matrix = np.mean(test_confusion_matrices, axis=0)
    
    print(f'--- Average Training Metrics ---')
    print(f'Accuracy: {avg_train_accuracy}')
    print(f'Precision: {avg_train_precision}')
    print(f'Recall: {avg_train_recall}')
    print(f'Confusion Matrix:\n{avg_train_conf_matrix}')
    print('\n')
    
    print(f'--- Average Testing Metrics ---')
    print(f'Accuracy: {avg_test_accuracy}')
    print(f'Precision: {avg_test_precision}')
    print(f'Recall: {avg_test_recall}')
    print(f'Confusion Matrix:\n{avg_test_conf_matrix}')
    print('\n')
    
    avg_metrics_path = os.path.join(main_folder_path, 'average_metrics.txt')
    with open(avg_metrics_path, 'w') as f:
        f.write(f'--- Average Training Metrics ---\n')
        f.write(f'Accuracy: {avg_train_accuracy}\n')
        f.write(f'Precision: {avg_train_precision}\n')
        f.write(f'Recall: {avg_train_recall}\n')
        f.write(f'Confusion Matrix:\n{avg_train_conf_matrix}\n\n')
        
        f.write(f'--- Average Testing Metrics ---\n')
        f.write(f'Accuracy: {avg_test_accuracy}\n')
        f.write(f'Precision: {avg_test_precision}\n')
        f.write(f'Recall: {avg_test_recall}\n')
        f.write(f'Confusion Matrix:\n{avg_test_conf_matrix}\n')

if __name__ == "__main__":
    run_experiment()
