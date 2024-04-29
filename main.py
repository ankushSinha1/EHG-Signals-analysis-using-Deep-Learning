import wfdb
import matplotlib.pyplot as plt

header = wfdb.rdheader('./term-preterm-ehg-dataset-with-tocogram-1.0.0/tpehgt_n001')
print(header.comments)

# Waveforms for non-pregnant
for i in range(1,6):
    record = wfdb.rdrecord(f'./term-preterm-ehg-dataset-with-tocogram-1.0.0/tpehgt_n{i:03}', channels=[0, 1, 2, 3, 4, 5, 6, 7])  # Adjust channels as needed
    num_channels = len(record.sig_name)
    fig, axs = plt.subplots(num_channels, 1, figsize=(10, 4*num_channels))

    for j in range(num_channels):
        signal_name = record.sig_name[j]
        signal_data = record.p_signal[:, j]
        
        axs[j].plot(signal_data)
        axs[j].set_title(f'{signal_name}', fontsize=10)  # Change title font size
        axs[j].set_ylabel('Amplitude', fontsize=10)  # Change y-label font size
        axs[j].grid(True)
plt.subplots_adjust(hspace=1.5) 
plt.show()
plt.close()
# Waveforms for term
for i in range(1,14):
    record = wfdb.rdrecord(f'./term-preterm-ehg-dataset-with-tocogram-1.0.0/tpehgt_t{i:03}', channels=[0, 1, 2, 3, 4, 5, 6, 7])  # Adjust channels as needed
    num_channels = len(record.sig_name)
    fig, axs = plt.subplots(num_channels, 1, figsize=(10, 4*num_channels))

    for j in range(num_channels):
        signal_name = record.sig_name[j]
        signal_data = record.p_signal[:, j]
        
        axs[j].plot(signal_data)
        axs[j].set_title(f'{signal_name}', fontsize=10)  # Change title font size
        axs[j].set_ylabel('Amplitude', fontsize=10)  # Change y-label font size
        axs[j].grid(True)
plt.subplots_adjust(hspace=1.5) 
plt.show()
plt.close()

# Waveforms for pre-term
# for i in range(1,14):
#     record = wfdb.rdrecord(f'./term-preterm-ehg-dataset-with-tocogram-1.0.0/tpehgt_p{i:03}', channels=[0, 1, 2, 3, 4, 5, 6, 7])  # Adjust channels as needed
#     num_channels = len(record.sig_name)
#     fig, axs = plt.subplots(num_channels, 1, figsize=(10, 4*num_channels))

#     for j in range(num_channels):
#         signal_name = record.sig_name[j]
#         signal_data = record.p_signal[:, j]
        
#         axs[j].plot(signal_data)
#         axs[j].set_title(f'{signal_name}', fontsize=10)  # Change title font size
#         axs[j].set_ylabel('Amplitude', fontsize=10)  # Change y-label font size
#         axs[j].grid(True)

# plt.subplots_adjust(hspace=1.5) 
# plt.show()
# plt.close()