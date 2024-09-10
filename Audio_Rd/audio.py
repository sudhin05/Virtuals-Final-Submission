import wave
import json
from vosk import Model, KaldiRecognizer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModel

import torch
import os
import argparse
import sys
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
import threading

sys.path.append('/home/uas-dtu/SudhinDarpa/grand_finale/CLIP_VIRTUAL_LODA')
from report_publisher import pub  

vosk_model_path = "/content/vosk-model-en-us-0.22-lgraph"
voskmodel = Model(vosk_model_path)


model_name = "sentence-transformers/all-MiniLM-L6-v2"  # You can choose any suitable model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


"""
 This following code needs to be triggered after listening for an audio file 
 of the casualty for an X amount of time.
 So after listening for an audio file, if it gets the audio file then it executes the code else, it should publish null
 Also if we get null, publish for respiratory distrss
"""

def transcribe_audio(audio_file):
    wf = wave.open(audio_file, "rb")
    rec = KaldiRecognizer(voskmodel, wf.getframerate())
    rec.SetWords(True)

    results = []
    while True:

        """ISKO YAAD SE DEKHLENA"""
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            part_result = json.loads(rec.Result())
            results.append(part_result)
    part_result = json.loads(rec.FinalResult())
    results.append(part_result)

    return " ".join([r['text'] for r in results if 'text' in r])

def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def audio_model(input_text):
    class_labels = ["This sentence describes a person in an emergency who is giving clear, logical, and reality-based information about their situation, injury, or need for help", "This sentence describes a person who is confused, disoriented, or hallucinating, making nonsensical or detached-from-reality statements in an emergency situation."]
    class_embeddings = get_embeddings(class_labels)

    # List of input texts to classify
    # input_texts = [
    #     "I'm only scratched don't worry about me", 
    #     "Help Help I am stuck", 
    #     "The stars are falling", 
    #     "I'm fine", 
    #     "Help others I only have a slight injury", 
    #     "Where are my glasses, I can't see", 
    #     "It's just a headache where are the first responders", 
    #     "I'm Sahil", 
    #     "Woof woof"
    # ]



    # for input_text in input_texts:
        # Get the embedding for the input text
    input_embedding = get_embeddings([input_text])

    # Compute cosine similarity between the input and each class
    similarities = cosine_similarity(input_embedding, class_embeddings)

    # Find the class with the highest similarity
    predicted_class_idx = np.argmax(similarities)

    print(input_text)
    print(class_labels[predicted_class_idx])
    print("-------------------------------------------------")

    if predicted_class_idx == 0:
        return "normal"
    
    if predicted_class_idx == 1:
        return "abnormal"
    

def audio_result_maker(folder,results):
    parts = folder.split('_')
    observation_start = parts[5]   
    casualty_id = parts[1]        
    latitude = parts[2]            
    longitude = parts[3]           
    altitude = parts[4]   
    observation_end = observation_start + 10        
    
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
        "injuries": {
            "respiratory_distress": False,
            "alertness": {
                "verbal": results
        } 
            }
        
    }
        
    return json_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AUDIO in the specified folder.")
    parser.add_argument('--root_path', type=str, required=True, help='Path to the root folder containing images')
    parser.add_argument('--read_path', type=str, required=True, help='Path to the read folder or file')

    # Parse arguments
    args = parser.parse_args()
    transcription = transcribe_audio(args.read_path)
    results = audio_model(transcription)
    
    json_report = audio_result_maker(args.root_path,results)
    
    t4 = threading.Thread(target=pub, args=(json_report,))
    t4.start()

    
# def main():
#     for i in os.listdir("/content"):
#       if i.endswith(".wav"):
#         audio_file = os.path.join("/content",i)
#         # Transcribe audio
#         transcription = transcribe_audio(audio_file, vosk_model_path)

#         if transcription:
#             print(f"Transcription: {i}  :  {transcription}")

#         else:
#             print("Failed to transcribe the audio.")

