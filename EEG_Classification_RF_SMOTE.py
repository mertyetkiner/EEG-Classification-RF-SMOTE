# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import time

# Function to extract features from EEG channels
def extract_features(row, eeg_channels):
    try:
        features = {}
        # Extract basic statistical features for each EEG channel
        for channel in eeg_channels:
            features[f'{channel}_value'] = row[channel]
            features[f'{channel}_mean'] = np.mean(row[channel])
            features[f'{channel}_median'] = np.median(row[channel])
            features[f'{channel}_std'] = np.std(row[channel])
            features[f'{channel}_var'] = np.var(row[channel])
            features[f'{channel}_max'] = np.max(row[channel])
        return features
    except Exception as e:
        print(f"Error in extract_features function: {e}")
        return {}

try:
    # Load the dataset
    print("Loading data set...")
    df = pd.read_csv('C:\\Masaüstü\\sade_csv_dosyalari\\combined_data\\combined_data.csv')
    print("Data set loaded.")

    # Split the data into features (X) and labels (y)
    X = df.drop('label', axis=1)
    y = df['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Identify EEG channels in the dataset
    eeg_channels = [col for col in X.columns if 'eeg' in col]

    # Apply Synthetic Minority Over-sampling Technique (SMOTE) for handling class imbalance
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print("SMOTE applied.")

    # Perform feature extraction for the resampled dataset
    print("Feature extraction is in progress...")
    total_samples = X_resampled.shape[0]
    print(f"Total number of samples: {total_samples}")
    start_time = time.time()
    feature_list = []
    for index, row in X_resampled.iterrows():
        features = extract_features(row, eeg_channels)
        feature_list.append(features)
        if index % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"Number of samples processed: {index}/{total_samples} ({(index/total_samples)*100:.2f}%) - Elapsed time: {elapsed_time:.2f} seconds")
    features_df = pd.DataFrame(feature_list)
    print("Feature extraction completed.")
    
    # Train a Random Forest Classifier using the extracted features
    print("Model is being trained...")
    model = RandomForestClassifier(random_state=42)
    model.fit(features_df, y_resampled)
    print("Model training completed.")

    # Perform feature extraction for the test set
    print("Feature extraction for the test set...")
    test_features_list = []
    for index, row in X_test.iterrows():
        features = extract_features(row, eeg_channels)
        test_features_list.append(features)
    test_features_df = pd.DataFrame(test_features_list)
    print("Completed feature extraction for the test set.")

    # Make predictions on the test set using the trained model
    print("The model predicts on the test set...")
    y_pred = model.predict(test_features_df)
    print("The model's prediction process is complete.")

    # Evaluate the performance of the model
    print("Evaluating the performance of the model...")
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Display the evaluation metrics
    print("Model Accuracy:", accuracy)
    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", class_report)
except Exception as e:
    print(f"Error in main process flow: {e}")
