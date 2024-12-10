import numpy as np
import pandas as pd
import logging
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import mne
import spkit
from featurewiz import featurewiz
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor
import os

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_individual_channels(raw_data, cleaned_data, start_time=0, end_time=5):
    """
    Plot EEG signals for individual channels before and after artifact removal.

    Parameters:
    -----------
    raw_data : mne.io.Raw
        Original raw EEG data.
    cleaned_data : mne.io.Raw
        Cleaned EEG data after artifact removal.
    start_time : float
        Start time (in seconds) for the segment to plot.
    end_time : float
        End time (in seconds) for the segment to plot.
    """
    logging.info(f"Plotting EEG signals from {start_time}s to {end_time}s.")
    sfreq = int(raw_data.info['sfreq'])
    start_idx, end_idx = int(start_time * sfreq), int(end_time * sfreq)
    raw_data_array = raw_data.get_data()[:, start_idx:end_idx]
    cleaned_data_array = cleaned_data.get_data()[:, start_idx:end_idx]
    times = raw_data.times[start_idx:end_idx]

    n_channels = len(raw_data.ch_names)
    fig, axes = plt.subplots(n_channels, 2, figsize=(15, 3 * n_channels), sharex=True)
    fig.suptitle(f'EEG Artifact Removal (Segment: {start_time}s to {end_time}s)', fontsize=16)

    for i, ch_name in enumerate(raw_data.ch_names):
        axes[i, 0].plot(times, raw_data_array[i], label=f'Raw: {ch_name}', color='blue')
        axes[i, 0].set_title(f'Raw EEG ({ch_name})')
        axes[i, 0].grid(True)

        axes[i, 1].plot(times, cleaned_data_array[i], label=f'Cleaned: {ch_name}', color='green')
        axes[i, 1].set_title(f'Cleaned EEG ({ch_name})')
        axes[i, 1].grid(True)

    for ax in axes[:, 0]:
        ax.set_ylabel("Amplitude")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    logging.info(f"Plotting completed.")


def apply_atar_artifact_removal(raw_data):
    """
    Apply ATAR (Automatic and Tunable Artifact Removal) to clean EEG data.

    Parameters:
    -----------
    raw_data : mne.io.Raw
        Raw EEG data.

    Returns:
    --------
    mne.io.Raw
        Cleaned EEG data after artifact removal.
    """
    logging.info(f"Applying ATAR artifact removal.")
    data = raw_data.get_data()

    # ATAR parameters
    beta = 0.5
    k1 = np.std(data) * 0.1
    k2 = np.std(data) * 3

    cleaned_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        cleaned_data[i] = spkit.eeg.ATAR(data[i], beta=beta, k1=k1, k2=k2)

    cleaned_raw = raw_data.copy()
    cleaned_raw._data = cleaned_data
    logging.info(f"ATAR artifact removal completed.")
    return cleaned_raw


def extract_time_domain_features(data):
    """
    Extract time-domain features from EEG signals.

    Parameters:
    -----------
    data : ndarray
        EEG signal array (channels x timepoints).

    Returns:
    --------
    dict
        Time-domain features for each channel.
    """
    logging.info("Extracting time-domain features.")
    features = {}
    for i, channel_data in enumerate(data):
        features[i] = {
            'mean': np.mean(channel_data),
            'variance': np.var(channel_data),
            'rms': np.sqrt(np.mean(channel_data ** 2)),
            'hjorth_activity': np.var(channel_data),
            'hjorth_mobility': np.sqrt(np.var(np.diff(channel_data)) / np.var(channel_data)),
            'hjorth_complexity': np.sqrt(np.var(np.diff(np.diff(channel_data))) / np.var(np.diff(channel_data))),
            'skewness': skew(channel_data),
            'kurtosis': kurtosis(channel_data),
            'peak_to_peak': np.ptp(channel_data)
        }
    logging.info("Time-domain feature extraction completed.")
    return features


def extract_frequency_domain_features(data, sfreq, bands={'delta': (0.5, 4), 'theta': (4, 8),
                                                          'alpha': (8, 13), 'beta': (13, 30)}):
    """
    Extract frequency-domain features using Welch's method.

    Parameters:
    -----------
    data : ndarray
        EEG signal array (channels x timepoints).
    sfreq : int
        Sampling frequency of the EEG data.
    bands : dict
        Frequency bands for feature extraction.

    Returns:
    --------
    dict
        Frequency-domain features for each channel.
    """
    logging.info("Extracting frequency-domain features.")
    features = {}
    for i, channel_data in enumerate(data):
        features[i] = {}
        freqs, psd = welch(channel_data, fs=sfreq, nperseg=sfreq * 2)

        for band, (low, high) in bands.items():
            band_power = np.sum(psd[(freqs >= low) & (freqs < high)])
            features[i][f'{band}_power'] = band_power

        psd_normalized = psd / np.sum(psd)
        spectral_entropy = -np.sum(psd_normalized * np.log(psd_normalized))
        features[i]['spectral_entropy'] = spectral_entropy
    logging.info("Frequency-domain feature extraction completed.")
    return features


def save_features_to_csv(participant_id, time_features, freq_features, output_path):
    """
    Save extracted features to a CSV file with the Autistic label.

    Parameters:
    -----------
    participant_id : str
        Identifier for the participant.
    time_features : dict
        Extracted time-domain features.
    freq_features : dict
        Extracted frequency-domain features.
    output_path : str
        Directory path to save the CSV file.
    """
    logging.info(f"Saving features for participant: {participant_id}.")
    label = 1 if participant_id.startswith('ASD') else 0

    all_features = []
    for channel, t_features in time_features.items():
        row = {'participant_id': participant_id, 'channel': channel, 'Autistic': label}
        row.update(t_features)
        row.update(freq_features[channel])
        all_features.append(row)

    df = pd.DataFrame(all_features)
    csv_path = os.path.join(output_path, f'{participant_id}_features.csv')
    df.to_csv(csv_path, index=False)
    logging.info(f"Features saved to {csv_path}.")


def perform_feature_selection(input_csv, output_csv):
    """
    Perform feature selection using Featurewiz.

    Parameters:
    -----------
    input_csv : str
        Path to the CSV file containing extracted features.
    output_csv : str
        Path to save the CSV file after feature selection.
    """
    logging.info("Performing feature selection.")
    target_column = 'Autistic'

    # Perform feature selection
    selected_features, df_selected = featurewiz(
        dataname=input_csv,  # Input CSV file
        target=target_column,  # Target column for selection
        corr_limit=0.7,  # Correlation threshold for SULOV
        verbose=2  # Display process details
    )

    # Save the reduced dataset
    df_selected.to_csv(output_csv, index=False)
    logging.info(f"Selected features saved to {output_csv}.")


def load_and_preprocess_eeg(participant_id, path):
    """
    Load and preprocess EEG data for a participant.

    Parameters:
    -----------
    participant_id : str
        Identifier for the participant.
    path : str
        Path to the EEG dataset.

    Returns:
    --------
    tuple
        Extracted time-domain and frequency-domain features.
    """
    try:
        logging.info(f"Loading EEG data for participant: {participant_id}")
        file_path = os.path.join(path, f'{participant_id}_Resting.set')
        raw_data = mne.io.read_raw_eeglab(file_path, preload=True)

        montage = mne.channels.make_standard_montage('standard_1020')
        raw_data.set_montage(montage)

        #common_channels = ['AF4', 'F8', 'F4', 'FC6', 'T8', 'P8', 'O2', 'Oz', 'Fp1']

        common_channels = ['C3', 'Cz', 'C4', 'CPz', 'P3', 'Pz', 'P4', 'POz']

        # Check which channels are available in the raw EEG data
        available_channels = [ch for ch in common_channels if ch in raw_data.info['ch_names']]
        missing_channels = list(set(common_channels) - set(available_channels))

        if missing_channels:
            logging.warning(f"Missing channels for participant {participant_id}: {', '.join(missing_channels)}")

        # Pick only the available channels
        raw_data.pick_channels(available_channels)

        target_sfreq = 512
        if raw_data.info['sfreq'] != target_sfreq:
            raw_data.resample(target_sfreq)

        raw_data.filter(1, 40, fir_design='firwin')
        cleaned_data = apply_atar_artifact_removal(raw_data)

        data = cleaned_data.get_data()
        time_features = extract_time_domain_features(data)
        freq_features = extract_frequency_domain_features(data, target_sfreq)

        logging.info(f"Preprocessing completed for participant: {participant_id}")
        return time_features, freq_features

    except Exception as e:
        logging.error(f"Error processing data for participant {participant_id}: {e}")
        return None, None


def process_participant(participant_id, path, output_path):
    """
    Process a single participant's EEG data, extract features, and save them.
    """
    logging.info(f"Processing participant: {participant_id}")
    time_features, freq_features = load_and_preprocess_eeg(participant_id, path)

    if time_features and freq_features:
        save_features_to_csv(participant_id, time_features, freq_features, output_path)
    else:
        logging.error(f"Failed to process data for participant {participant_id}.")


def combine_all_features(output_path, combined_csv_path):
    """
    Combine features of all participants into a single CSV file.

    Parameters:
    -----------
    output_path : str
        Path where individual participant feature files are stored.
    combined_csv_path : str
        Path to save the combined CSV file.
    """
    logging.info("Combining all participant features into one CSV file.")
    feature_files = [os.path.join(output_path, f) for f in os.listdir(output_path) if f.endswith('_features.csv')]
    combined_data = pd.concat([pd.read_csv(file) for file in feature_files], ignore_index=True)
    combined_data.to_csv(combined_csv_path, index=False)
    logging.info(f"Combined features saved to {combined_csv_path}.")


def perform_feature_selection(input_csv, output_csv):
    """
    Perform feature selection using Featurewiz.

    Parameters:
    -----------
    input_csv : str
        Path to the CSV file containing extracted features.
    output_csv : str
        Path to save the CSV file after feature selection.
    """
    logging.info("Performing feature selection.")
    target_column = 'Autistic'

    # Load dataset
    df = pd.read_csv(input_csv)

    # Drop non-feature columns (e.g., identifiers)
    non_feature_columns = ['participant_id', 'channel']
    feature_columns = [col for col in df.columns if col not in non_feature_columns and col != target_column]

    # Perform feature selection
    selected_features, df_selected = featurewiz(
        dataname=df[feature_columns + [target_column]],  # Only keep features and target for selection
        target=target_column,  # Target column for selection
        corr_limit=0.7,  # Correlation threshold for SULOV
        verbose=2  # Display process details
    )



def main():
    """
    Main function to process EEG data for all participants, save extracted features,
    combine features into one dataset, and perform feature selection.
    """
    path = 'C:/Users/Dhruv/PycharmProjects/DeepLearning/autism_marker_EEG/Aging/'
    output_path = 'C:/Users/Dhruv/PycharmProjects/DeepLearning/autism_marker_EEG/features/'
    combined_csv_path = ('C:/Users/Dhruv/PycharmProjects/DeepLearning/autism_marker_EEG'
                         '/features_combined.csv')
    reduced_combined_csv_path = ('C:/Users/Dhruv/PycharmProjects/DeepLearning/autism_marker_EEG'
                                 '/features_combined_reduced.csv')

    os.makedirs(output_path, exist_ok=True)

    participant_ids = [
        'ASD1', 'ASD2', 'ASD3', 'ASD4', 'ASD5', 'ASD6', 'ASD7', 'ASD8', 'ASD9', 'ASD10', 'ASD11',
        'ASD12', 'ASD13', 'ASD14', 'ASD15', 'ASD16', 'ASD17', 'ASD18', 'ASD19', 'ASD20', 'ASD21',
        'ASD22', 'P51', 'ASD24', 'ASD25', 'ASD26', 'ASD27', 'ASD28', 'ASD29', 'P1', 'P5', 'P6',
        'P9', 'P10', 'P12', 'P16', 'P17', 'P18', 'P20', 'P24', 'P25', 'P26', 'P29', 'P31', 'P32',
        'P37', 'P38', 'P41', 'P42', 'P43', 'P44', 'P52', 'P53', 'P54', 'P56', 'P60'
    ]

    # Parallelize the processing of participants
    logging.info("Starting parallel processing of participants.")
    with ProcessPoolExecutor() as executor:
        executor.map(process_participant, participant_ids, [path] * len(participant_ids),
                     [output_path] * len(participant_ids))

    logging.info("All participants processed. Combining features.")
    # Combine all participant features into one CSV
    combine_all_features(output_path, combined_csv_path)

    logging.info("Feature selection on combined dataset.")
    # Perform feature selection on the combined dataset
    perform_feature_selection(combined_csv_path, reduced_combined_csv_path)


if __name__ == '__main__':
    main()
