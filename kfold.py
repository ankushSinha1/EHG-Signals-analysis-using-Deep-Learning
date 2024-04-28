import wfdb
import numpy as np
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
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

# Define the model architecture
def create_model():
    model = models.Sequential([
        layers.Input(shape=X_scaled.shape[1]),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    return model

# Define the number of folds for cross-validation
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Lists to store accuracy scores for each fold
accuracy_scores = []
best_model = None
best_accuracy = 0.0
best_history = None

# Perform k-fold cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(X_scaled)):
    print(f'Fold {fold + 1}/{n_splits}')
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Create and compile the model
    model = create_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_data=(X_val, y_val))
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy}')
    accuracy_scores.append(accuracy)
    
    # Check if this model has the highest accuracy so far
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_history = history
        best_model = model

# Calculate and print the average accuracy across all folds
avg_accuracy = np.mean(accuracy_scores)
print(f'Average Test Accuracy: {avg_accuracy}')
print(f'Best Test Accuracy: {best_accuracy}')


# Plot training history of the best model
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(best_history.history['accuracy'])
plt.plot(best_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(best_history.history['loss'])
plt.plot(best_history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()