import os
import logging
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Load and preprocess data
def load_and_preprocess_data(feature_file):
    if not os.path.exists(feature_file):
        logging.error(f"Feature file not found: {feature_file}")
        raise FileNotFoundError(f"{feature_file} not found.")

    # Load dataset
    data = pd.read_csv(feature_file)

    # Drop unnecessary columns
    X = data.drop(columns=['Autistic', 'participant_id', 'channel'])
    y = data['Autistic']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# Build the neural network model with hyperparameters
def build_nn_model(hp, input_dim):
    model = Sequential()

    # Input layer
    model.add(Dense(units=hp['units_input'], activation='relu', input_dim=input_dim))

    # Hidden layers
    for i in range(hp['num_layers']):
        model.add(Dense(units=hp[f'units_{i}'], activation='relu'))
        model.add(Dropout(rate=hp[f'dropout_{i}']))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=hp['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# Perform hyperparameter tuning
def hyperparameter_tuning(X, y, input_dim, tuning_file):
    if os.path.exists(tuning_file):
        logging.info("Tuning file exists. Loading saved hyperparameters.")
        with open(tuning_file, 'r') as file:
            best_hps = json.load(file)
        return best_hps

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert labels to binary arrays if needed
    y_train = y_train.values
    y_test = y_test.values

    # Hyperparameter tuner
    def model_builder(hp):
        return build_nn_model(
            {
                'units_input': hp.Int('units_input', min_value=64, max_value=256, step=32),
                'num_layers': hp.Int('num_layers', 1, 3),
                'units_0': hp.Int('units_0', min_value=32, max_value=128, step=16),
                'dropout_0': hp.Float('dropout_0', 0.2, 0.5, step=0.1),
                'units_1': hp.Int('units_1', min_value=32, max_value=128, step=16),
                'dropout_1': hp.Float('dropout_1', 0.2, 0.5, step=0.1),
                'learning_rate': hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])
            },
            input_dim
        )

    tuner = RandomSearch(
        model_builder,
        objective='val_accuracy',
        max_trials=10,  # Number of combinations to try
        executions_per_trial=1,  # Average over multiple runs
        directory='hyperparam_tuning',
        project_name='autism_nn'
    )

    # Run the tuner
    tuner.search(X_train, y_train, epochs=50, validation_split=0.2,
                 callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

    # Get the best hyperparameters
    best_hp_obj = tuner.get_best_hyperparameters(num_trials=1)[0]

    best_hps = {
        'units_input': best_hp_obj.get('units_input'),
        'num_layers': best_hp_obj.get('num_layers'),
        'units_0': best_hp_obj.get('units_0'),
        'dropout_0': best_hp_obj.get('dropout_0'),
        'units_1': best_hp_obj.get('units_1'),
        'dropout_1': best_hp_obj.get('dropout_1'),
        'learning_rate': best_hp_obj.get('learning_rate')
    }

    # Save the best hyperparameters to a file
    with open(tuning_file, 'w') as file:
        json.dump(best_hps, file)

    return best_hps


# Train and evaluate the model with best hyperparameters
def train_and_evaluate_with_best_hps(X, y, best_hps):
    """
    Train and evaluate the neural network model with the best hyperparameters.

    Parameters:
    -----------
    X : np.ndarray
        Scaled feature matrix.
    y : pd.Series
        Target labels.
    best_hps : dict
        Best hyperparameters.

    Returns:
    --------
    None
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and compile the model with the best hyperparameters
    model = build_nn_model(best_hps, input_dim=X.shape[1])

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, y_train,
                        validation_split=0.2,
                        epochs=150,
                        batch_size=32,
                        callbacks=[early_stopping],
                        verbose=1)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Make predictions
    y_pred = model.predict(X_test).flatten()
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Performance metrics
    logging.info("\nClassification Report:")
    logging.info(classification_report(y_test, y_pred_binary))
    # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
    if len(set(y_test)) > 1:
        logging.info(f"ROC AUC Score: {roc_auc_score(y_test, y_pred)}")


# Main function
def main():
    """
    Main function to load data, perform hyperparameter tuning,
    and evaluate the neural network model.
    """
    feature_file = os.path.join(
        'C:/Users/Dhruv/PycharmProjects/DeepLearning/autism_marker_EEG/combined_time_freq.csv')
    tuning_file = 'best_hyperparameters.json'
    X, y = load_and_preprocess_data(feature_file)

    # Hyperparameter tuning
    best_hps = hyperparameter_tuning(X, y, input_dim=X.shape[1], tuning_file=tuning_file)

    # Train and evaluate with the best hyperparameters
    train_and_evaluate_with_best_hps(X, y, best_hps)


if __name__ == '__main__':
    main()
