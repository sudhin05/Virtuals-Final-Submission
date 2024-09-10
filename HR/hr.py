import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from sklearn.decomposition import FastICA ,PCA
import matplotlib.pyplot as plt
import os
import sys
import threading
import argparse
from PIL import Image

sys.path.append('/grand_finale/CLIP_VIRTUAL_LODA')
from report_publisher import pub  

def load_frames(folder_path):
    frames = []
    folder = folder_path.split('/')[-1]
    for img_name in sorted(os.listdir(folder_path)):
        txt_name = os.path.splitext(img_path)[0] + '.txt'
        img_path = os.path.join(folder_path, img_name)
        txt_path = os.path.join(folder_path, txt_name)
        if img_name.endswith(('.jpg', '.jpeg', '.png')) and os.path.exists(txt_path):
            
            last_img = os.path.splitext(img_name)[0]
            
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            image = Image.open(img_path)
            coords1 = [float(coord) for coord in lines[0].split()]
            cropped_image1 = image.crop((coords1[0], coords1[1], coords1[2], coords1[3]))
            
            coords2 = [float(coord) for coord in lines[1].split()]
            double_cropped_image = cropped_image1.crop((coords2[0], coords2[1], coords2[2], coords2[3]))
            
            img = cv2.imread(os.path.join(folder_path, double_cropped_image))
            img = cv2.resize(img,(100,100))
            frames.append(img)

    obs_end = last_img.split('_')[-1] 
    return np.array(frames) , obs_end , folder

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
    
def hr_result_maker(folder,observation_end,results):
    parts = folder.split('_')
    observation_start = parts[5]   
    casualty_id = parts[1]        
    latitude = parts[2]            
    longitude = parts[3]           
    altitude = parts[4]           
    
    json_file = {
        "observation_start": float(observation_start),
        "observation_end": float(observation_end),  
        "assessment_time": float(observation_end), 
        "casualty_id": int(casualty_id),
        "drone_id": 0,  
        "location": {
            "lon": float(longitude),
            "lat": float(latitude),
            "alt": float(altitude)
        },
        "vitals": {
            "heart_rate": results
            }
        
    }
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HR")
    parser.add_argument('--root_path', type=str, required=True, help='Path to the root folder containing images')
    fps = 20
    args = parser.parse_args()
    frames,obs_end,folder = load_frames(args.root_path)
    original_signals = extract_color_signals(frames)
    processed_signals = preprocess_signals(original_signals, fps)
    sources=perform_pca(processed_signals)
    hr = estimate_heart_rate(sources.T[2], fps)
    print(hr)
    
    json_report = hr_result_maker(folder,obs_end,hr)
    print(json_report)

    t3 = threading.Thread(target=pub,args=(json_report,))
    t3.start()
    # for i in os.listdir("/home/harshit/images_good_face"):
    #     print(i)
    #     folder_path = '/home/harshit/images_good_face/'+i

    #     fps = 20
    #     try:

    #         frames = load_frames(folder_path)
    #         original_signals = extract_color_signals(frames)
    #         processed_signals = preprocess_signals(original_signals, fps)
    #         #sources = perform_ica(processed_signals)
    #         sources=perform_pca(processed_signals)
    #         #plot_signals(original_signals, processed_signals, sources)
    #         # print(sources)
    #         hr = estimate_heart_rate(sources.T[2], fps)
    #         print(f"Estimated heart rate from source 3: {hr:.2f} BPM")
    #         return hr




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
