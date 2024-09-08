import torch
from torchvision import transforms as T
from PIL import Image
import os
import time 

def ocular_model(input_dir):
    threshold = 0.8
    threshold_non_testable = 0.9
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "final_weights.pt"
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
    not_able_to_count = 0
    for img_name in os.listdir(input_dir):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            txt_name = os.path.splitext(img_path)[0] + '.txt'
            img_path = os.path.join(input_dir, img_name)
            txt_path = os.path.join(input_dir, txt_name)
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            image = Image.open(img_path)
            coords1 = [float(coord) for coord in lines[0].split()]
            cropped_image1 = image.crop((coords1[0], coords1[1], coords1[2], coords1[3]))
            
            # Convert the second line to coordinates and crop the cropped image
            coords2 = [float(coord) for coord in lines[1].split()]
            if coords2[0] == 0 and coords2[1] == 0 and coords2[2] == 0 and coords2[3] == 0:
                not_able_to_count += 1
            else : 
                double_cropped_image = cropped_image1.crop((coords2[0], coords2[1], coords2[2], coords2[3]))
            
                
                prediction = eye_model(double_cropped_image)

                if prediction == 'Close-Eyes':
                    close_count += 1
                elif prediction == 'Open-Eyes':
                    open_count += 1        
    if total_images > 0:
        print("#########################")
        print(f"Total images processed: {total_images}")
        print(f"Open eyes count: {open_count}")
        print(f"Close eyes count: {close_count}")
        print(f"Not able to count: {not_able_to_count}")
        print(f"Percentage open eyes: {open_count / total_images:.2%}")
        print(f"Percentage close eyes: {close_count / total_images:.2%}")
        print(f"Percentage not able to count: {not_able_to_count / total_images:.2%}")
        print("#########################")
        if float(open_count / total_images) >= threshold :
            print("Open")
            return "Open"
        elif float(not_able_to_count / total_images) >= threshold_non_testable:
            print("Non Testable")
            # ocular_result = "Non Testable"
        else :
                print("Close")
                return "Close"
    else:
        print("No images found in the specified folder.")
if __name__ == '__main__':
    input_dir = "/home/uas-dtu/nikhil-darpa/images_sahil_test"
    for i in os.listdir(input_dir):
        print(i)
        ocular_model(os.path.join(input_dir, i))
    # ocular_model("/home/uas-dtu/nikhil-darpa/images_sahil_test/waypt10_5_48.92691625763328_8.110598888465962_-30.286706924438477")
