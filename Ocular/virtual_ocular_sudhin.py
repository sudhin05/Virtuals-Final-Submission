import torch
from torchvision import transforms as T
from PIL import Image
import os
import threading
import sys
import argparse
sys.path.append('/home/uas-dtu/SudhinDarpa/grand_finale/CLIP_VIRTUAL_LODA')
from report_publisher import pub  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "/home/uas-dtu/SudhinDarpa/Ocular/final_weights.pt"
model = torch.jit.load(model_path)
model = model.to(device)

def ocular_model(folder_path):
    folder = folder_path.split('/')[-1]
    threshold = 0.8
    classes = ['Close-Eyes', 'Open-Eyes']
    transform = T.Compose([
        T.Resize((284, 284)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def eye_model(image):
        image = transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(device)
        model.eval()
        output = model(image)
        _, pred = torch.max(output, 1)
        return classes[pred.item()]
    
    def occ_result_maker(folder,observation_end,results):
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
            "injuries": {
                "alertness": {
                "ocular": results
            } 
                }
            
        }
        
        return json_file

    open_count = 0
    close_count = 0
    total_images = 0
 
    for img_name in sorted(os.listdir(folder_path)):
        txt_name = os.path.splitext(img_name)[0] + '.txt'
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

            prediction = eye_model(double_cropped_image)

            if prediction == 'Close-Eyes':
                close_count += 1
            elif prediction == 'Open-Eyes':
                open_count += 1

            total_images += 1
      
    obs_end = last_img.split('_')[-1]      
            
    if total_images > 0:
        print("#########################")
        print(f"Total images processed: {total_images}")
        print(f"Open eyes count: {open_count}")
        print(f"Close eyes count: {close_count}")
        print(f"Percentage open eyes: {open_count / total_images:.2%}")
        print(f"Percentage close eyes: {close_count / total_images:.2%}")
        print("#########################")
        if float(open_count / total_images) >= threshold:
            print("Open")
            ocular_result = "open"
        else:
            print("Close")
            ocular_result = "close"
            
    

    json_report = occ_result_maker(folder,obs_end,ocular_result)
    
    t2 = threading.Thread(target=pub, args=(json_report,))
    t2.start()

    return 

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run ocular calculator on images in the specified folder.")
    parser.add_argument('--root_path', type=str, required=True, help='Path to the root folder containing images')

    # Parse arguments
    args = parser.parse_args()
    ocular_model(args.root_path)
