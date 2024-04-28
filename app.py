import wfdb
import numpy as np
from scipy import signal
from sklearn.model_selection import train_test_split
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

# Load and preprocess signals for n records
n_records = 5
max_len_n = 0
signals_n = []
for i in range(1, n_records + 1):
    record = wfdb.rdrecord(f'./term-preterm-ehg-dataset-with-tocogram-1.0.0/tpehgt_n{i:03}', channels=[0, 1, 2, 3, 4, 5, 6, 7])
    max_len_n = max(max_len_n, min([len(record.p_signal[:, j]) for j in range(record.p_signal.shape[1])]))
    for j in range(record.p_signal.shape[1]):
        signal_data = record.p_signal[:, j]
        preprocessed_signal = preprocess_signal(signal_data, record.fs)
        signals_n.append(preprocessed_signal)

# Load and preprocess signals for t records
t_records = 13
max_len_t = 0
signals_t = []
for i in range(1, t_records + 1):
    record = wfdb.rdrecord(f'./term-preterm-ehg-dataset-with-tocogram-1.0.0/tpehgt_t{i:03}', channels=[0, 1, 2, 3, 4, 5, 6, 7])
    max_len_t = max(max_len_t, min([len(record.p_signal[:, j]) for j in range(record.p_signal.shape[1])]))
    for j in range(record.p_signal.shape[1]):
        signal_data = record.p_signal[:, j]
        preprocessed_signal = preprocess_signal(signal_data, record.fs)
        signals_t.append(preprocessed_signal)

# Load and preprocess signals for p records
p_records = 13
max_len_p = 0
signals_p = []
for i in range(1, p_records + 1):
    record = wfdb.rdrecord(f'./term-preterm-ehg-dataset-with-tocogram-1.0.0/tpehgt_p{i:03}', channels=[0, 1, 2, 3, 4, 5, 6, 7])
    max_len_p = max(max_len_p, min([len(record.p_signal[:, j]) for j in range(record.p_signal.shape[1])]))
    for j in range(record.p_signal.shape[1]):
        signal_data = record.p_signal[:, j]
        preprocessed_signal = preprocess_signal(signal_data, record.fs)
        signals_p.append(preprocessed_signal)

# Adjust signal lengths to the maximum length
max_len = max(max_len_n, max_len_t, max_len_p)
signals_n = adjust_signal_length(signals_n, max_len)
signals_t = adjust_signal_length(signals_t, max_len)
signals_p = adjust_signal_length(signals_p, max_len)

# Create labels for the data
labels_n = [0] * len(signals_n)
labels_t = [1] * len(signals_t)
labels_p = [2] * len(signals_p)

# Combine data and labels
X = np.concatenate([signals_n, signals_t, signals_p])
y = np.concatenate([labels_n, labels_t, labels_p])
# Check the shape of the concatenated signals
print("Shape of concatenated signals:", X.shape)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the deep learning model
model = models.Sequential([
    layers.Input(shape=X_train_scaled.shape[1]),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model and store the training history
history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Accuracy: {accuracy}')

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test, y_pred_classes)
print(f'Accuracy: {accuracy}')

# Plot training history
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()