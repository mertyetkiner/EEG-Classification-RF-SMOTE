# EEG Signal Classification with Random Forest

## Overview

This project focuses on the classification of EEG signals using a Random Forest Classifier. EEG (Electroencephalogram) signals are processed to extract features, and a machine learning model is trained to predict the labels of these signals. The dataset used for this project is loaded from a CSV file, and various steps are performed, including data preprocessing, feature extraction, model training, and evaluation.

## Project Structure

The project consists of a Python script containing the following main components:

1. **Importing Libraries**: The necessary libraries are imported, including pandas, numpy, scikit-learn, and imbalanced-learn.

2. **Feature Extraction Function**: A function named `extract_features` is defined to extract statistical features from EEG channels, such as mean, median, standard deviation, variance, and maximum value.

3. **Loading Dataset**: The EEG dataset is loaded from a CSV file, and the data is split into features (X) and labels (y).

4. **Data Splitting**: The dataset is further split into training and testing sets using the `train_test_split` function.

5. **Identifying EEG Channels**: The EEG channels in the dataset are identified based on column names.

6. **Handling Class Imbalance**: Synthetic Minority Over-sampling Technique (SMOTE) is applied to address class imbalance in the training set.

7. **Feature Extraction for Resampled Dataset**: Statistical features are extracted from the resampled dataset, and a new DataFrame is created.

8. **Model Training**: A Random Forest Classifier is trained using the extracted features and the resampled labels.

9. **Feature Extraction for Test Set**: Features are extracted from the test set using the same procedure as the training set.

10. **Model Prediction**: The trained model is used to make predictions on the test set.

11. **Model Evaluation**: The performance of the model is evaluated using accuracy, confusion matrix, and classification report metrics.

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/mertyetkiner/EEG-Classification-RF-SMOTE.git
   ```

2. Navigate to the project directory:

   ```bash
   cd your-repository
   ```

3. Install the required dependencies:

   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn
   ```

4. Run the script:

   ```bash
   python your_script.py
   ```

## Dependencies

- pandas
- numpy
- scikit-learn
- imbalanced-learn

## Author

Niyazi Mert Yetkiner

## License

This project is licensed under the [MIT License](LICENSE).
