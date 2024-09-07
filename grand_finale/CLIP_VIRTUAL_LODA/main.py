import torch
import torchvision.transforms as transforms
import clip
from datasets.tromnet import trom_net
from utils import *
from run_utils import *
from lora import run_lora
from PIL import Image
import cv2
from ultralytics import YOLO
# from dataset import trom_net
import time

def main():
    args = get_arguments()
    set_random_seed(args.seed)
    
    clip_model,_ = clip.load(args.backbone)
    clip_model.eval()
    logit_scale = 100
    # print(clip_model)
    # Prepare dataset
    print("Preparing dataset.")
    
    # test_image_paths = ['/home/uas-dtu/CLIP_VIRTUAL_LODA/normal/1.png', '/home/uas-dtu/CLIP_VIRTUAL_LODA/normal/2.png']
    # test_images = [Image.open(path).convert('RGB') for path in test_image_paths]
        
    # dataset = build_dataset(args.dataset, args.root_path)
    dataset=trom_net(args.root_path)
    test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=160, num_workers=8, shuffle=False, pin_memory=True)
    run_lora(args, clip_model, logit_scale, dataset,test_loader)
    # dataset.delete_images()

if __name__ == '__main__':
    st = time.time()
    main()
    print(F"TIME TAKEN : {time.time()-st}")


    