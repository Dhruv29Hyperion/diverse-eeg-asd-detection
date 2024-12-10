# Diverse EEG-ASD Detection

This project aims to develop a robust method for detecting Autism Spectrum Disorder (ASD) using Electroencephalography (EEG) data. The key objectives of this research are:

1. Integrating diverse EEG datasets to create a comprehensive and representative dataset.
2. Exploring both single-channel and multi-channel analysis approaches for ASD detection.
3. Identifying the optimal combination of preprocessing techniques, feature sets, and machine learning models.

## Methodology

1. **Data Integration**:
   - Combined two EEG datasets: Aging and BCIAUT
   - Employed data harmonization techniques such as Advanced Time-Aligned Resampling (ATAR), band-pass filtering, channel standardization, and feature normalization.

2. **Feature Extraction**:
   - Extracted time-domain features (mean, standard deviation, variance, etc.) and frequency-domain features (Welch's Method, spectral entropy).
   - Utilized Recurrence Quantitative Analysis (RQA) to derive advanced non-linear features.

3. **Model Approaches**:
   - Evaluated 10 different machine learning models, including ensemble methods like Voting Classifier and Stacking Classifier.
   - Performed hyperparameter optimization using GridSearchCV.

## Key Findings

1. **Single-Channel Analysis**:
   - Achieved a peak accuracy of 91.98% using the XGBoost classifier.
   - Performed better with smaller datasets compared to multi-channel analysis.

2. **Multi-Channel Analysis**:
   - Obtained the highest accuracy of 0.92 using time-frequency features.
   - Multi-channel approaches showed improved performance with larger datasets.

## Contributions and Future Work

- Successfully harmonized EEG datasets from different sources, demonstrating flexibility in handling single and multi-channel recordings.
- Developed a comprehensive approach for ASD detection using machine learning, which could assist healthcare professionals in early and objective assessment.
- Further research is needed to refine dataset integration techniques and address the limitations caused by variations in experimental setups.

## Potential Impact

This research represents a promising step towards using advanced computational techniques for early ASD diagnosis, offering a non-invasive alternative to traditional behavioral assessments. The developed methods have the potential to improve the accuracy and reliability of ASD detection, ultimately benefiting healthcare professionals and individuals with ASD.
