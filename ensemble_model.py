import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (GradientBoostingClassifier, ExtraTreesClassifier,
                              RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier)
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

BEST_PARAMS_FILE = "../Results/aging_rqa/best_model_params.json"

# Load features from a single file
def load_combined_features(feature_file):
    if not os.path.exists(feature_file):
        logging.error(f"Feature file not found: {feature_file}")
        raise FileNotFoundError(f"{feature_file} not found.")
    return pd.read_csv(feature_file)

# Preprocess the dataset by scaling features
def preprocess_data(data):
    X = data.drop(columns=['Autistic', 'participant_id', 'channel'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns), data['Autistic']

# Define hyperparameter grids for each model
parameter_grids = {
    # Existing parameter grids (unchanged)
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
    if os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE, 'r') as f:
            all_params = json.load(f)
        return all_params.get(model_name, None)
    return None

# Aggregate channel-level predictions to patient-level predictions
def aggregate_patient_predictions(data, predictions, test_data, method='majority_vote'):
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
    X = data.drop(columns=['Autistic', 'participant_id', 'channel'])
    y = data['Autistic']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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

            # Train and evaluate individual model
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

    # Add Ensemble Models: Voting and Stacking
    # Create Voting Classifier
    voting_clf = VotingClassifier(estimators=[
        ('SVM', models['SVM']),
        ('KNN', models['KNN']),
        ('GradientBoosting', models['GradientBoosting']),
        ('RandomForest', models['RandomForest'])
    ], voting='soft')

    # Train and evaluate the Voting Classifier
    voting_clf.fit(X_train, y_train)
    y_pred_voting = voting_clf.predict(X_test)
    y_prob_voting = voting_clf.predict_proba(X_test)[:, 1]
    patient_results_voting = aggregate_patient_predictions(data, y_pred_voting, test_data, method='majority_vote')

    metrics_voting = {
        'Model': 'VotingClassifier',
        'Accuracy': classification_report(y_test, y_pred_voting, output_dict=True)['accuracy'],
        'ROC AUC Score': roc_auc_score(y_test, y_prob_voting) if y_prob_voting is not None else None
    }
    results.append(metrics_voting)

    logging.info(f"Classification Report for VotingClassifier:\n{classification_report(y_test, y_pred_voting)}")
    logging.info(f"Confusion Matrix for VotingClassifier:\n{confusion_matrix(y_test, y_pred_voting)}")

    # Create Stacking Classifier
    stacking_clf = StackingClassifier(estimators=[
        ('SVM', models['SVM']),
        ('KNN', models['KNN']),
        ('GradientBoosting', models['GradientBoosting']),
        ('RandomForest', models['RandomForest'])
    ], final_estimator=LogisticRegression())

    # Train and evaluate the Stacking Classifier
    stacking_clf.fit(X_train, y_train)
    y_pred_stacking = stacking_clf.predict(X_test)
    y_prob_stacking = stacking_clf.predict_proba(X_test)[:, 1]
    patient_results_stacking = aggregate_patient_predictions(data, y_pred_stacking, test_data, method='majority_vote')

    metrics_stacking = {
        'Model': 'StackingClassifier',
        'Accuracy': classification_report(y_test, y_pred_stacking, output_dict=True)['accuracy'],
        'ROC AUC Score': roc_auc_score(y_test, y_prob_stacking) if y_prob_stacking is not None else None
    }
    results.append(metrics_stacking)

    logging.info(f"Classification Report for StackingClassifier:\n{classification_report(y_test, y_pred_stacking)}")
    logging.info(f"Confusion Matrix for StackingClassifier:\n{confusion_matrix(y_test, y_pred_stacking)}")

    # Compare model performances
    compare_model_performance(results)

# Compare model performances in a tabular format
def compare_model_performance(results):
    comparison_df = pd.DataFrame(results)
    logging.info("\nModel Performance Comparison:\n")
    logging.info(comparison_df.sort_values(by="ROC AUC Score", ascending=False))

# Main function
def main():
    feature_file = os.path.join('C:/Users/Dhruv/PycharmProjects/DeepLearning/autism_marker_EEG/combined_rqa_timeFreq.csv')
    data = load_combined_features(feature_file)
    X_scaled, y = preprocess_data(data)
    data = pd.concat([X_scaled, data[['Autistic', 'participant_id', 'channel']]], axis=1)
    train_and_evaluate_models(data, models, parameter_grids)


if __name__ == '__main__':
    main()
