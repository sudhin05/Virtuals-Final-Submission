import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from sklearn.decomposition import FastICA ,PCA
import matplotlib.pyplot as plt
import os

def load_frames(folder_path):
    frames = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(folder_path, filename))
            img = cv2.resize(img,(100,100))
            frames.append(img)
    return np.array(frames)

def extract_color_signals(frames):
    signals = []
    for channel in range(3):  # RGB channels
        signal = frames[:, :, :, channel].mean(axis=(1, 2))
        signals.append(signal)
    return np.array(signals).T

def preprocess_signals(signals, fps):
    # Detrend
    signals = signals - np.mean(signals, axis=0)
    
    # Normalize
    signals = signals / np.std(signals, axis=0)
    
    # Apply bandpass filter (0.7 - 4 Hz for heart rate 42-240 BPM)
    nyquist = fps / 2
    low = 0.7 / nyquist
    high = 4 / nyquist
    b, a = butter(3, [low, high], btype='band')
    signals = filtfilt(b, a, signals, axis=0)
    
    return signals

def perform_ica(signals):
    ica = FastICA(n_components=3)
    sources = ica.fit_transform(signals)
    return sources
def perform_pca(signals):
    pca = PCA(n_components=3)
    sources = pca.fit_transform(signals)
    return sources


def estimate_heart_rate(source, fps):
    # Find peaks in the signal
    peaks, _ = find_peaks(source, distance=fps//2)  # At least 0.5 seconds between peaks
    
    # Calculate heart rate
    if len(peaks) > 1:
        heart_rate = 60 * fps * (len(peaks) - 1) / (peaks[-1] - peaks[0])
        return heart_rate
    else:
        return None
def estimate_heart_rate_welch(source, fps):
    # Compute Welch's periodogram
    freqs, psd = welch(source, fs=fps, nperseg=len(source)//2)
    
    # Find the frequency with maximum power in the range of 0.7-4 Hz (42-240 BPM)
    mask = (freqs >= 0.7) & (freqs <= 4)
    peak_freq = freqs[mask][np.argmax(psd[mask])]
    
    # Convert frequency to BPM
    heart_rate = peak_freq * 60
    
    return heart_rate
def plot_signals(original_signals, processed_signals, sources):
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    axes[0].plot(original_signals)
    axes[0].set_title('Original Color Signals')
    axes[0].legend(['Red', 'Green', 'Blue'])
    
    axes[1].plot(processed_signals)
    axes[1].set_title('Preprocessed Signals')
    axes[1].legend(['Red', 'Green', 'Blue'])
    
    axes[2].plot(sources)
    axes[2].set_title('Sources')
    axes[2].legend(['Source 1', 'Source 2', 'Source 3'])
    
    plt.tight_layout()
    plt.show()

# Main process
for i in os.listdir("/home/harshit/images_good_face"):
    print(i)
    folder_path = '/home/harshit/images_good_face/'+i
    fps = 20
    try:

        frames = load_frames(folder_path)
        original_signals = extract_color_signals(frames)
        processed_signals = preprocess_signals(original_signals, fps)
        #sources = perform_ica(processed_signals)
        sources=perform_pca(processed_signals)
        #plot_signals(original_signals, processed_signals, sources)
        # print(sources)
        hr = estimate_heart_rate(sources.T[2], fps)
        print(f"Estimated heart rate from source 3: {hr:.2f} BPM")
        return hr




        # # Estimate heart rate from each source
        # for i, source in enumerate(sources.T):
            
        #     hr = estimate_heart_rate(source, fps)
        #     # hr_w=estimate_heart_rate_welch(source,fps)
        #     if hr:
        #         print(f"Estimated heart rate from source {i+1}: {hr:.2f} BPM")
        #     else:
        #         print(f"Could not estimate heart rate from source {i+1}")
    except:
        print("hoho")
