import torch
import torch.nn.functional as F
import os
import torchvision.transforms as transforms 
from utils import *
from torchvision.utils import save_image

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers
import cv2
from collections import defaultdict
from report_publisher import pub
import threading



def evaluate_lora(args, clip_model, loader, dataset):
    clip_model.eval()    
    with torch.no_grad():
        template = dataset.template[0]
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
    
    results = defaultdict(lambda: defaultdict(int))
    
    with torch.no_grad():
        for images, target, paths in loader:
            images = images.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            predicted_classes = cosine_similarity.argmax(dim=-1)
            
            for img_path, pred in zip(paths, predicted_classes):
                folder_name = img_path.split('/')[-2]                 
                category = folder_name
                # print(dataset.classnames[pred],paths)
                predicted_class_name = dataset.classnames[pred]
                results[category][predicted_class_name] += 1
                
    final_results = {}
    final_results["upper_extremity"] = "normal"
    final_results["lower_extremity"] = "normal"
    upper_extremity = {"larm", "rarm"}
    lower_extremity = {"lleg", "rleg"}

    for folder, counts in results.items():
        max_class = max(counts, key=counts.get)

        if max_class == 'injury_and_amputation':
            final_results[folder] = 'amputation'
        elif max_class == 'injury':
            final_results[folder] = 'wound'
        elif max_class == 'no_injury':
            final_results[folder] = 'normal'

        # Check for upper or lower extremity conditions
        if folder in upper_extremity and final_results[folder] == 'wound':
            final_results["upper_extremity"] = "wound"
        elif folder in lower_extremity and final_results[folder] == 'wound':
            final_results["lower_extremity"] = "wound"


    
    for key in ['larm','rarm','lleg','rleg']:
        del final_results[key]
        
    l_dict = {'head':final_results['face'],'torso':final_results['chest'],'upper_extremity':final_results['upper_extremity'],'lower_extremity':final_results['lower_extremity']}
    # Print the final results
    # print(l_dict)
    # required_categories = ['face', 'chest', 'upper_extremity', 'lower_extremity']
    # for category in required_categories:
    #     if category not in final_results:
    #         # final_results[category] = 'absent'  # or another default value
    #         continue
    count_dict = {}
    for folder, counts in results.items():
        count_dict[folder] = {
            'no_injury': counts.get('no_injury', 0),
            'injury': counts.get('injury', 0),
            'injury_and_amputation': counts.get('injury_and_amputation', 0)
        }
    
    return l_dict,count_dict
        
    # return l_dict

def run_lora(args, clip_model, logit_scale, dataset,test_loader):
    
    VALIDATION = False
    
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    test_features = pre_load_features(clip_model, test_loader)
    
    test_features = test_features.cuda()
 
    clip_logits = logit_scale * test_features @ textual_features
    
    test_features = test_features.cpu()    
    
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda() 
    
    if args.eval_only:
        load_lora(args, list_lora_layers,"lora_weights/lora_weights_960_2.9566854533582632e-05.pt")
        results = evaluate_lora(args, clip_model, test_loader,dataset)
        # results['pat=puth'].append(args.root_path)

    json_file ={
            "observation_start": args.observation_start,
            "observation_end": args.observation_end,
            "assessment_time": args.observation_end,
            "casualty_id": int(args.root_path.split("/")[-1].split("_")[1]),  
            "drone_id": 0,  
            "location": {
                "lon": float(args.root_path.split("/")[-1].split("_")[3]),
                "lat": float(args.root_path.split("/")[-1].split("_")[2]),
                "alt": float(args.root_path.split("/")[-1].split("_")[4])
            },
            "injuries": {
                "trauma": results
                }
            }
    t1=threading.Thread(target=pub,args=(json_file,))
    t1.start()
    
    reuslts_txt_path = "results_trauma.txt"
    if not os.path.exists(reuslts_txt_path):
        with open(reuslts_txt_path, 'w') as f:
            f.write(f'{args.root_path.split("/")[-1]} : {results}\n')
            print(results)
    else:
        with open('results_trauma.txt','a') as f:
            f.write(f'{args.root_path.split("/")[-1]} : {results}\n')
            print(results) 
    
    return

