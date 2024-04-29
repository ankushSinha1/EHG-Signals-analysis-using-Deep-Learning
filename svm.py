import wfdb
import numpy as np
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def preprocess_signal(signal_data, fs):
    if fs <= 0:
        raise ValueError("Invalid sampling frequency")
    nyquist = 0.5 * fs
    lowcut = 0.5
    highcut = min(45.0, nyquist - 1)
    low = lowcut / nyquist
    high = highcut / nyquist
    order = 4
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, signal_data)
    normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
    return normalized_signal

# Function to pad or truncate signals to a fixed length
def adjust_signal_length(signals, target_length):
    adjusted_signals = []
    for signal in signals:
        if len(signal) < target_length:
            padded_signal = np.pad(signal, (0, target_length - len(signal)), mode='constant')
            adjusted_signals.append(padded_signal)
        elif len(signal) > target_length:
            truncated_signal = signal[:target_length]
            adjusted_signals.append(truncated_signal)
        else:
            adjusted_signals.append(signal)
    return adjusted_signals

# Load and preprocess signals for each class
def load_and_preprocess_records(class_name, num_records):
    signals = []
    max_len = 0
    for i in range(1, num_records + 1):
        record = wfdb.rdrecord(f'./term-preterm-ehg-dataset-with-tocogram-1.0.0/tpehgt_{class_name}{i:03}', channels=[0, 1, 2, 3, 4, 5, 6, 7])
        max_len = max(max_len, min([len(record.p_signal[:, j]) for j in range(record.p_signal.shape[1])]))
        for j in range(record.p_signal.shape[1]):
            signal_data = record.p_signal[:, j]
            preprocessed_signal = preprocess_signal(signal_data, record.fs)
            signals.append(preprocessed_signal)
    return adjust_signal_length(signals, max_len)

# Load and preprocess signals for each class
signals_n = load_and_preprocess_records('n', 5)
signals_t = load_and_preprocess_records('t', 13)
signals_p = load_and_preprocess_records('p', 13)

# Create labels for the data
labels_n = [0] * len(signals_n)
labels_t = [1] * len(signals_t)
labels_p = [2] * len(signals_p)

# Combine data and labels

# Pad or truncate signals to a fixed length before concatenating
max_len = max(len(signal) for signal in signals_n + signals_t + signals_p)
signals_n = adjust_signal_length(signals_n, max_len)
signals_t = adjust_signal_length(signals_t, max_len)
signals_p = adjust_signal_length(signals_p, max_len)

X = np.concatenate([signals_n, signals_t, signals_p])
y = np.concatenate([labels_n, labels_t, labels_p])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the number of folds for cross-validation
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Lists to store accuracy scores for each fold
accuracy_scores = []

# Perform k-fold cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(X_scaled)):
    print(f'Fold {fold + 1}/{n_splits}')
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Create and train the SVM model
    svm_model = SVC(kernel='rbf', C=1, gamma='scale')
    svm_model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy}')
    accuracy_scores.append(accuracy)

# Calculate and print the average accuracy across all folds
avg_accuracy = np.mean(accuracy_scores)
print(f'Average Test Accuracy: {avg_accuracy}')
