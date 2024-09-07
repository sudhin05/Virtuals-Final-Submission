import torch
from torchvision import transforms as T
from PIL import Image
import os
import time 
# st = time.time()
def ocular_model(foldername,img_list,txt_list):
    # In the input directory there are images 
    threshold = 0.8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "/home/uas-dtu/SudhinDarpa/Occular/MRL_dataset_deit-384.pt"
    model = torch.jit.load(model_path)
    model = model.to(device)
    classes = ['Close-Eyes', 'Open-Eyes']
    transform = T.Compose([
        # T.Resize((384, 384)),
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
    

        
    open_count = 0
    close_count = 0
    total_images = 0
    for img_name in os.listdir(input_dir):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            txt_name = img_name[:-4] + '.txt'
            img_path = os.path.join(input_dir, img_name)
            txt_path = os.path.join(input_dir, txt_name)
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            image = Image.open(img_path)
            coords1 = [float(coord) for coord in lines[0].split()]
            cropped_image1 = image.crop((coords1[0], coords1[1], coords1[2], coords1[3]))
            
            # Convert the second line to coordinates and crop the cropped image
            coords2 = [float(coord) for coord in lines[1].split()]
            double_cropped_image = cropped_image1.crop((coords2[0], coords2[1], coords2[2], coords2[3]))

            prediction = eye_model(double_cropped_image)

            # Increment counts based on the prediction
            if prediction == 'Close-Eyes':
                close_count += 1
            elif prediction == 'Open-Eyes':
                open_count += 1

            total_images += 1
            
    if total_images > 0:
        print("#########################")
        print(f"Total images processed: {total_images}")
        print(f"Open eyes count: {open_count}")
        print(f"Close eyes count: {close_count}")
        print(f"Percentage open eyes: {open_count / total_images:.2%}")
        print(f"Percentage close eyes: {close_count / total_images:.2%}")
        print(int(open_count / total_images))
        print("#########################")
        if float(open_count / total_images) >= threshold :
            print("Open")
            return "Open"
        else :
            print("Close")
            return "Close"
    else:
        print("No images found in the specified folder.")
if __name__ == '__main__':
    ocular_model("/home/uas-dtu/SudhinDarpa/Ocular/waypt2_15_48.927230464786334_8.109561293040809_-32.223052978515625_10")