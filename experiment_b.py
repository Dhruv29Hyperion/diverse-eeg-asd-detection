import os
import numpy as np
import scipy.io as sio
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb  # For XGBoost
from sklearn.neighbors import KNeighborsClassifier  # For KNN
from tqdm import tqdm
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.sigmoid(x)


class BinocularRivalryAnalysis:
    def __init__(self, fft_data_path, rls_data_path):
        self.fft_data_path = fft_data_path
        self.rls_data_path = rls_data_path
        
    def _is_autism_participant(self, filename):
        return filename.startswith('cumulus')
    
    def _extract_neural_rivalry_index(self, rls_filepath):
        try:
            rls_data = sio.loadmat(rls_filepath)
            transitions = rls_data.get('transitions', None)
            if transitions is None:
                print(f"No transitions data found in {rls_filepath}")
                return 0
            
            rivalry_indices = []
            for t in range(2):
                try:
                    transition_cell = transitions[0, t]
                    left_data = transition_cell['left'][0, 0]
                    right_data = transition_cell['right'][0, 0]
                    mask = np.isnan(left_data) | np.isnan(right_data)
                    rivalry = np.abs(left_data - right_data)
                    rivalry[mask] = np.nan
                    avg_rivalry = np.nanmean(rivalry)
                    rivalry_indices.append(avg_rivalry)
                except Exception as transition_error:
                    print(f"Error processing transition {t}: {transition_error}")
                    continue
            
            if rivalry_indices:
                return np.mean(rivalry_indices)
            else:
                return 0
        except Exception as e:
            print(f"Error in extract_neural_rivalry_index: {e}")
            return 0
    
    def _extract_fft_features(self, fft_filepath):
        try:
            mat_data = sio.loadmat(fft_filepath)
            freq_axis = mat_data['freqAxis'].flatten()
            elec_fft = mat_data['elecFFT'].flatten()
            freq_5_7_idx = np.argmin(np.abs(freq_axis - 5.7))
            freq_8_5_idx = np.argmin(np.abs(freq_axis - 8.5))
            amplitude_5_7 = np.abs(elec_fft[freq_5_7_idx])
            amplitude_8_5 = np.abs(elec_fft[freq_8_5_idx])
            amplitude_diff = amplitude_8_5 - amplitude_5_7
            amplitude_ratio = amplitude_8_5 / (amplitude_5_7 + 1e-10)
            return np.array([amplitude_diff, amplitude_ratio])
        except Exception as e:
            print(f"Error processing {fft_filepath}: {e}")
            return np.array([0, 0])
    
    def load_data(self):
        X = []
        y = []
        print("loading data for your model!")
        for filename in tqdm(os.listdir(self.fft_data_path)):
            if filename.endswith('.mat'):
                is_autism = self._is_autism_participant(filename)
                fft_filepath = os.path.join(self.fft_data_path, filename)
                fft_features = self._extract_fft_features(fft_filepath)
                participant_base = filename.split('_')[0]
                rls_filename = f"{participant_base}.mat"
                rls_filepath = os.path.join(self.rls_data_path, rls_filename)
                if os.path.exists(rls_filepath):
                    nri = self._extract_neural_rivalry_index(rls_filepath)
                    combined_features = np.concatenate([fft_features, [nri]])
                    X.append(combined_features)
                    y.append(1 if is_autism else 0)
        
        return np.array(X), np.array(y)
    
    def run_svm_classification(self):
        """
        Run Support Vector Machine classification
        
        Returns:
        --------
        dict
            Classification results including accuracy and report
        """
        # Load data
        X, y = self.load_data()
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Prepare Leave-One-Out Cross-Validation
        loo = LeaveOneOut()
        
        # Predictions and true labels
        predictions = []
        true_labels = []
        
        # Leave-One-Out Cross-Validation
        for train_index, test_index in loo.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Train SVM
            svm = SVC(kernel='rbf', gamma='auto')
            svm.fit(X_train, y_train)
            
            # Predict
            pred = svm.predict(X_test)
            
            predictions.extend(pred)
            true_labels.extend(y_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(true_labels, predictions)
        
        # Generate classification report
        report = classification_report(true_labels, predictions, 
                                       target_names=['Control', 'Autism'])
        
        return {
            'accuracy': accuracy,
            'report': report,
            'X_shape': X.shape,
            'y_distribution': np.bincount(y)
        }
    
    def run_classification(self, model):
        X, y = self.load_data()
        
        # Check for NaN or infinite values and handle them
        if np.isnan(X).any() or np.isinf(X).any():
            print("Detected NaN or infinite values in the dataset. Replacing with 0.")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        loo = LeaveOneOut()
        predictions = []
        true_labels = []
        
        for train_index, test_index in tqdm(loo.split(X_scaled)):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Train the classifier
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            
            predictions.extend(pred)
            true_labels.extend(y_test)
        
        # Evaluate performance
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, target_names=['Control', 'Autism'])
        
        return {
            'accuracy': accuracy,
            'report': report,
            'X_shape': X.shape,
            'y_distribution': np.bincount(y)
        }
    
    def run_neural_network_classification(self, epochs=100, batch_size=16, learning_rate=0.001):
        """
        Run neural network classification.

        Parameters:
        -----------
        epochs : int
            Number of training epochs.
        batch_size : int
            Size of each mini-batch.
        learning_rate : float
            Learning rate for the optimizer.

        Returns:
        --------
        dict
            Neural network results including accuracy and classification report.
        """
        # Load and preprocess data
        X, y = self.load_data()
        if np.isnan(X).any() or np.isinf(X).any():
            print("Detected NaN or infinite values in the dataset. Replacing with 0.")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        # Dataset and DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Define model, loss, and optimizer
        input_dim = X_scaled.shape[1]
        model = SimpleNN(input_dim)
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

        # Testing
        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor).round().numpy()
        
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions, target_names=['Control', 'Autism'])

        return {
            'accuracy': accuracy,
            'report': report,
            'X_shape': X.shape,
            'y_distribution': np.bincount(y)
        }
        

# Example usage with multiple classifiers
if __name__ == "__main__":
    analysis = BinocularRivalryAnalysis(
        fft_data_path="./FFTData",
        rls_data_path="./RLSData"
    )

    # Initialize classifiers
    

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, criterion='gini', n_jobs=10) 
    dt_model = DecisionTreeClassifier(random_state=42, criterion='gini', splitter='random')
    xgb_model = xgb.XGBClassifier(tree_method='hist', n_estimators=100, random_state=42)
    knn_model = KNeighborsClassifier(n_neighbors=3, weights='distance')
    # Run classification for each model
    print("Running Random Forest")
    results_rf = analysis.run_classification(rf_model)
    print("Running Decision Tree")
    results_dt = analysis.run_classification(dt_model)
    print("Running XGBoost")
    results_xgb = analysis.run_classification(xgb_model)
    print("Running KNN")
    results_knn = analysis.run_classification(knn_model)
    
    # Print results
    print("Random Forest Results:")
    print(f"Accuracy: {results_rf['accuracy'] * 100:.2f}%")
    print("\nClassification Report:")
    print(results_rf['report'])
    
    print("Decision Tree Results:")
    print(f"Accuracy: {results_dt['accuracy'] * 100:.2f}%")
    print("\nClassification Report:")
    print(results_dt['report'])
    
    print("XGBoost Results:")
    print(f"Accuracy: {results_xgb['accuracy'] * 100:.2f}%")
    print("\nClassification Report:")
    print(results_xgb['report'])
    
    print("KNN Results:")
    print(f"Accuracy: {results_knn['accuracy'] * 100:.2f}%")
    print("\nClassification Report:")
    print(results_knn['report'])

    
    results = analysis.run_svm_classification()
    
    print("SVC Classification Results:")
    print(f"Accuracy: {results['accuracy'] * 100:.2f}%")
    print("\nClassification Report:")
    print(results['report'])

    print("Running Neural Network")
    results_nn = analysis.run_neural_network_classification(epochs=700, batch_size=16, learning_rate=0.001)
    print("Neural Network Results:")
    print(f"Accuracy: {results_nn['accuracy'] * 100:.2f}%")
    print("\nClassification Report:")
    print(results_nn['report'])