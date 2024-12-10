import pandas as pd
import logging
import numpy as np
import pywt
import pyrqa
import nolds
import os
import mne
from concurrent.futures import ProcessPoolExecutor
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius
from pyrqa.computation import RPComputation
import inspect

# Setting up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_rqa_features(data, embedding_dimension=3, time_delay=1, radius=0.5, minimum_line_length=2):
    logging.debug("Extracting RQA features for the given data.")
    data = (data - np.mean(data)) / np.std(data)  # Normalize the data
    time_series = TimeSeries(data, embedding_dimension=embedding_dimension, time_delay=time_delay)
    settings = Settings(time_series, neighbourhood=FixedRadius(radius), similarity_measure=EuclideanMetric)
    computation = RPComputation.create(settings)
    rp_result = computation.run()
    recurrence_matrix = rp_result.recurrence_matrix

    total_points = len(recurrence_matrix) ** 2
    rr = np.sum(recurrence_matrix) / total_points  # Recurrence Rate

    diagonal_lines = np.sum(np.sum(recurrence_matrix, axis=1) >= minimum_line_length)
    det = diagonal_lines / np.sum(recurrence_matrix)  # Determinism

    vertical_lines = np.sum(np.sum(recurrence_matrix, axis=0) >= minimum_line_length)
    lam = vertical_lines / np.sum(recurrence_matrix)  # Laminarity

    l_max = np.max(np.sum(np.triu(recurrence_matrix), axis=1))  # Maximum Diagonal Line Length

    diagonal_lengths = np.sum(np.triu(recurrence_matrix), axis=1)
    diagonal_lengths = diagonal_lengths[diagonal_lengths > 0]
    diagonal_prob = diagonal_lengths / np.sum(diagonal_lengths)
    l_entr = -np.sum(diagonal_prob * np.log(diagonal_prob))  # Entropy of Diagonal Lines

    l_mean = np.mean(diagonal_lengths)  # Mean Diagonal Line Length

    tt = np.max(np.sum(recurrence_matrix, axis=0))  # Trapping Time

    rqa_features = {
        'RR': rr,
        'DET': det,
        'LAM': lam,
        'L_max': l_max,
        'L_entr': l_entr,
        'L_mean': l_mean,
        'TT': tt
    }
    
    logging.debug(f"RQA features extracted: {rqa_features}")
    return rqa_features

def extract_complexity_features(data):
    try:
        sample_entropy = nolds.sampen(data)
        data = (data - np.mean(data)) / np.std(data)
        dfa = nolds.dfa(data, nvals=[4, 16, 64])
        complexity_features = {
            'sample_entropy': sample_entropy,
            'dfa': dfa
        }
        logging.debug(f"Complexity features extracted: {complexity_features}")
        return complexity_features
    except Exception as e:
        logging.error(f"Error calculating complexity features: {e}")
        return {
            'sample_entropy': np.nan,
            'dfa': np.nan
        }

def extract_features_from_eeg(data, channel_name, embedding_dimension=3, time_delay=1, radius=0.1,
                              minimum_line_length=2):
    features = {}
    rqa_features = extract_rqa_features(data, embedding_dimension, time_delay, radius, minimum_line_length)
    features.update(rqa_features)
    # complexity_features = extract_complexity_features(data)
    # features.update(complexity_features)
    features['channel'] = channel_name
    logging.debug(f"Features for channel {channel_name}: {features}")
    return features

def load_and_preprocess_eeg(participant_id, path):
    try:
        logging.info(f"Loading EEG data for participant: {participant_id}")
        file_path = os.path.join(path, f'{participant_id}_Resting.set')
        raw_data = mne.io.read_raw_eeglab(file_path, preload=True)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw_data.set_montage(montage)

        # Only retain the channels available in the data
        common_channels = ['C3', 'Cz', 'C4', 'CPz', 'P3', 'Pz', 'P4', 'POz']
        available_channels = [ch for ch in common_channels if ch in raw_data.info['ch_names']]
        raw_data.pick_channels(available_channels)

        # Resample data to 512 Hz if necessary
        target_sfreq = 512
        if raw_data.info['sfreq'] != target_sfreq:
            raw_data.resample(target_sfreq)

        raw_data.filter(1, 40, fir_design='firwin')

        data = raw_data.get_data()
        all_features = {}
        for i, channel_name in enumerate(raw_data.ch_names):
            all_features[channel_name] = extract_features_from_eeg(data[i], channel_name)

        logging.info(f"Preprocessing completed for participant: {participant_id}")
        return all_features

    except Exception as e:
        logging.error(f"Error processing data for participant {participant_id}: {e}")
        return None

def save_features_to_csv(participant_id, all_features, output_path):
    all_features_list = []
    for channel_name, features in all_features.items():
        row = {'participant_id': participant_id, 'channel': channel_name}
        row.update(features)
        all_features_list.append(row)

    df = pd.DataFrame(all_features_list)
    csv_path = os.path.join(output_path, f'{participant_id}_features.csv')
    df.to_csv(csv_path, index=False)
    logging.info(f"Features saved to {csv_path}.")

def combine_all_features(output_path, combined_csv_path):
    feature_files = [os.path.join(output_path, f) for f in os.listdir(output_path) if f.endswith('_features.csv')]
    combined_data = pd.concat([pd.read_csv(file) for file in feature_files], ignore_index=True)
    combined_data.to_csv(combined_csv_path, index=False)
    logging.info(f"Combined features saved to {combined_csv_path}.")

def process_participants(participant_ids, data_path, output_path):
    all_features = load_and_preprocess_eeg(participant_ids, data_path)

    if all_features:
        save_features_to_csv(participant_ids, all_features, output_path)

    combine_all_features(output_path, os.path.join(output_path, 'features2_combined.csv'))

def main():
    for id in ['ASD1', 'ASD2', 'ASD3', 'ASD4', 'ASD5', 'ASD6', 'ASD7', 'ASD8', 'ASD9', 'ASD10', 'ASD11',
        'ASD12', 'ASD13', 'ASD14', 'ASD15', 'ASD16', 'ASD17', 'ASD18', 'ASD19', 'ASD20', 'ASD21',
        'ASD22', 'P51', 'ASD24', 'ASD25', 'ASD26', 'ASD27', 'ASD28', 'ASD29', 'P1', 'P5', 'P6',
        'P9', 'P10', 'P12', 'P16', 'P17', 'P18', 'P20', 'P24', 'P25', 'P26', 'P29', 'P31', 'P32',
        'P37', 'P38', 'P41', 'P42', 'P43', 'P44', 'P52', 'P53', 'P54', 'P56', 'P60']:
        participant_ids = [f'{id}']
        data_path = 'Aging/'
        output_path = 'Aging/features'
        combined_csv_path = 'features_combined.csv'
        
        os.makedirs(output_path, exist_ok=True)

        logging.info("Starting processing of participants.")
        with ProcessPoolExecutor() as executor:
            executor.map(process_participants, participant_ids, [data_path] * len(participant_ids),
                        [output_path] * len(participant_ids))

        logging.info("All participants processed. Combining features.")
        combine_all_features(output_path, combined_csv_path)

if __name__ == "__main__":
    main()