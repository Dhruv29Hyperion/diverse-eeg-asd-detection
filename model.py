import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (GradientBoostingClassifier, ExtraTreesClassifier,
                              RandomForestClassifier, AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import json
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BEST_PARAMS_FILE = "../Results/combined_rqa_timeFreq/best_model_params.json"


# Load features from a single file
def load_combined_features(feature_file):
    """
    Load all features and labels from a single CSV file.

    Parameters:
    -----------
    feature_file : str
        Path to the combined feature file.

    Returns:
    --------
    pd.DataFrame
        Combined features with labels.
    """
    if not os.path.exists(feature_file):
        logging.error(f"Feature file not found: {feature_file}")
        raise FileNotFoundError(f"{feature_file} not found.")
    return pd.read_csv(feature_file)


# Preprocess the dataset by scaling features
def preprocess_data(data):
    """
    Preprocess the dataset by scaling features.

    Parameters:
    -----------
    data : pd.DataFrame
        Combined dataset with features and labels.

    Returns:
    --------
    pd.DataFrame, pd.Series
        Scaled features and corresponding labels.
    """
    X = data.drop(columns=['Autistic', 'participant_id', 'channel'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns), data['Autistic']


# Define hyperparameter grids for each model
parameter_grids = {
    'SVM': {
        'C': np.logspace(-3, 3, 7),
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'class_weight': [None, 'balanced'],
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 10, 15],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    'GradientBoosting': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.5, 1],
        'max_depth': [3, 5, 7, 10],
        'subsample': [0.5, 0.7, 1.0],
        'min_samples_split': [2, 5, 10],
        'max_features': ['None', 'sqrt', 'log2']
    },
    'ExtraTrees': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': ['None', 'sqrt', 'log2'],
    },
    'RandomForest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': ['None', 'sqrt', 'log2'],
        'bootstrap': [True, False]
    },
    'DecisionTree': {
        'criterion': ['gini', 'entropy', 'log loss'],
        'min_samples_split': [2, 5, 10],
        'max_depth': [None, 10, 20, 30]
    },
    'LogisticRegression': {
        'solver': ['liblinear', 'lbfgs', 'saga'],
        'penalty': ['l1', 'l2', 'none'],
        'class_weight': [None, 'balanced'],
        'C': np.logspace(-3, 3, 7),
    },
    'AdaBoost': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.1, 1.0, 10.0],
        'algorithm': ['SAMME', 'SAMME.R']
    },
    'MLP': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'lbfgs', 'sgd'],
        'learning_rate': ['constant', 'adaptive'],
        'alpha': np.logspace(-4, 2, 7)
    },
    'NaiveBayes': {}  # No hyperparameters for tuning
}

# Define model classes
models = {
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'ExtraTrees': ExtraTreesClassifier(),
    'RandomForest': RandomForestClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'LogisticRegression': LogisticRegression(),
    'AdaBoost': AdaBoostClassifier(),
    'MLP': MLPClassifier(),
    'NaiveBayes': GaussianNB()
}


# Save best parameters to a file
def save_best_params(model_name, best_params):
    """
    Save the best parameters of a model to a JSON file.

    Parameters:
    -----------
    model_name : str
        Name of the model.
    best_params : dict
        Best parameters for the model.
    """
    if os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE, 'r') as f:
            all_params = json.load(f)
    else:
        all_params = {}

    all_params[model_name] = best_params

    with open(BEST_PARAMS_FILE, 'w') as f:
        json.dump(all_params, f, indent=4)
    logging.info(f"Saved best parameters for {model_name}.")


# Load best parameters from a file
def load_best_params(model_name):
    """
    Load the best parameters for a model from a JSON file, if available.

    Parameters:
    -----------
    model_name : str
        Name of the model.

    Returns:
    --------
    dict or None
        Best parameters for the model, or None if not found.
    """
    if os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE, 'r') as f:
            all_params = json.load(f)
        return all_params.get(model_name, None)
    return None


# Aggregate channel-level predictions to patient-level predictions
def aggregate_patient_predictions(data, predictions, test_data, method='majority_vote'):
    """
    Aggregate channel-level predictions to patient-level predictions.

    Parameters:
    -----------
    data : pd.DataFrame
        The input data containing 'participant_id' and channel-level predictions.
    predictions : np.ndarray
        The channel-level predictions for the test set.
    test_data : pd.DataFrame
        The test dataset with 'participant_id' for correct aggregation.
    method : str
        Aggregation method ('majority_vote' or 'average_probability').

    Returns:
    --------
    pd.DataFrame
        Patient-level predictions with participant IDs.
    """
    test_data_copy = test_data.copy()
    test_data_copy['prediction'] = predictions
    patient_results = test_data_copy.groupby('participant_id').agg({'prediction': 'mean'}).reset_index()

    if method == 'majority_vote':
        patient_results['final_prediction'] = (patient_results['prediction'] > 0.5).astype(int)
    elif method == 'average_probability':
        patient_results['final_prediction'] = patient_results['prediction']
    else:
        raise ValueError("Invalid aggregation method.")

    return patient_results[['participant_id', 'final_prediction']]


# Train and evaluate models with hyperparameter tuning
def train_and_evaluate_models(data, models, parameter_grids):
    """
    Train and evaluate models with hyperparameter tuning.

    Parameters:
    -----------
    data : pd.DataFrame
        Combined dataset with features and labels.
    models : dict
        Dictionary of model instances.
    parameter_grids : dict
        Hyperparameter grids for each model.

    Returns:
    --------
    None
    """
    X = data.drop(columns=['Autistic', 'participant_id', 'channel'])
    y = data['Autistic']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Combine test set data for aggregation
    test_data = data.loc[X_test.index, ['participant_id']]

    results = []  # Store results for comparison

    for model_name, model in models.items():
        logging.info(f"Training {model_name}...")

        try:
            best_params = load_best_params(model_name)
            if best_params:
                logging.info(f"Using previously saved best parameters for {model_name}: {best_params}")
                model.set_params(**best_params)
            else:
                if parameter_grids.get(model_name):
                    grid_search = GridSearchCV(estimator=model, param_grid=parameter_grids[model_name],
                                               scoring='accuracy', cv=5, verbose=2)
                    grid_search.fit(X_train, y_train)
                    best_params = grid_search.best_params_
                    save_best_params(model_name, best_params)
                    model = grid_search.best_estimator_
                    logging.info(f"Best parameters for {model_name}: {best_params}")
                else:
                    model.fit(X_train, y_train)

            # Always train the model on the training data
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            # Aggregating patient-level predictions
            patient_results = aggregate_patient_predictions(data, y_pred, test_data, method='majority_vote')

            # Collect performance metrics
            metrics = {
                'Model': model_name,
                'Accuracy': classification_report(y_test, y_pred, output_dict=True)['accuracy'],
                'ROC AUC Score': roc_auc_score(y_test, y_prob) if y_prob is not None else None
            }
            results.append(metrics)

            logging.info(f"Classification Report for {model_name}:\n{classification_report(y_test, y_pred)}")
            logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        except Exception as e:
            logging.error(f"Error training {model_name}: {e}")

    compare_model_performance(results)


# Compare model performances in a tabular format
def compare_model_performance(results):
    """
    Compare model performances in a tabular format.

    Parameters:
    -----------
    results : list of dict
        Performance metrics for each model.

    Returns:
    --------
    None
    """
    comparison_df = pd.DataFrame(results)
    logging.info("\nModel Performance Comparison:\n")
    logging.info(comparison_df.sort_values(by="ROC AUC Score", ascending=False))


# Main function
def main():
    """
    Main function to load features, train models, and evaluate them.
    """
    feature_file = os.path.join('C:/Users/Dhruv/PycharmProjects/DeepLearning/autism_marker_EEG/combined_rqa_timeFreq'
                                '.csv')
    data = load_combined_features(feature_file)
    X_scaled, y = preprocess_data(data)
    data = pd.concat([X_scaled, data[['Autistic', 'participant_id', 'channel']]], axis=1)
    train_and_evaluate_models(data, models, parameter_grids)


if __name__ == '__main__':
    main()