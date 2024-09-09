from ultralytics import YOLO
import sys
sys.path.append('grand_finale/ViTPose')

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
import os
import subprocess
from mmpose.datasets import DatasetInfo
import warnings
import numpy as np
import cv2
import time
from collections import deque   
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
from threading import Lock


def loader():
    pose_model = init_pose_model(
       'grand_finale/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192.py', 'grand_finale/ViTPose/demo/vitpose_small.pth', device='cuda:0')
    person_model = YOLO('grand_finale/ViTPose/demo/best_body.pt')
    face_model = YOLO('grand_finale/ViTPose/demo/yolov8n-face.pt')
    return pose_model,person_model,face_model

def run_pose_person_face(img_path, pose_model, person_model, face_model):
    try:
        img = cv2.imread(img_path)
        result_person = person_model.predict(img, verbose=False, conf=0.5)
        
        for r in result_person:
            box = r.boxes.xyxy.to("cpu").numpy()
            cls = r.boxes.cls.to("cpu").numpy()
            if box.shape[0] == 0:
                print(f"No Persons Detected")
                return None
            if cls[0] != 0:  # Skip non-person detections
                continue

        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        areas = widths * heights
        max_area_index = np.argmax(areas)
        max_area_box = box[max_area_index]        
        x1_body, y1_body, x2_body, y2_body = map(int, max_area_box)
        person_cropped = img[y1_body:y2_body, x1_body:x2_body]

        face_results = face_model.predict(person_cropped, verbose=False, conf=0.3)
        face_p = None
        if len(face_results) > 0:
            face_result = face_results[0]
            face_box = face_result.boxes.xyxy.to('cpu').numpy()
            if len(face_box) > 0:
                x1_face, y1_face, x2_face, y2_face = map(int, face_box[0])
                face = person_cropped[y1_face:y2_face, x1_face:x2_face]
                face = np.array(face, dtype=np.uint8)
                face_p = [x1_face, y1_face, x2_face, y2_face]

        dataset = pose_model.cfg.data['test']['type']
        dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
        if dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
        else:
            dataset_info = DatasetInfo(dataset_info)

        return_heatmap = False
        output_layer_names = None

        person_results = [{'bbox': np.array([0, 0, x2_body - x1_body, y2_body - y1_body])}]
        
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            person_cropped,
            person_results,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        keypoints = np.array(pose_results[0]['keypoints'])  # (17, 3) -> [x, y, score]

        # Prepare the save path for the text file
        txt_path = os.path.splitext(img_path)[0] + '.txt'

        # Save the results to the text file
        with open(txt_path, 'w') as f:
            # Save body bounding box
            f.write(f"{x1_body} {y1_body} {x2_body} {y2_body}\n")
            
            # Save face bounding box (if available)
            if face_p:
                f.write(f"{face_p[0]} {face_p[1]} {face_p[2]} {face_p[3]}\n")
            else:
                f.write("0 0 0 0\n")  # or you can choose not to save face bbox if not detected

            # Save keypoints (x y confidence)
            for kp in keypoints:
                f.write(f"{int(kp[0])} {int(kp[1])} {kp[2]:.4f}\n")

        # print(f"Results saved to {txt_path}")
    except Exception as e:
        print(e)
        print("exception in pose body face")

def get_gaussian_and_bfs_limbs(image,keypoints,sigmas,conf_thres):
    limb_images_list = []
    height, width = image.shape[:2]
    splatted_images = [np.zeros((height,width,3),dtype=np.float64),np.zeros((height,width,3),dtype=np.float64),np.zeros((height,width,3),dtype=np.float64),np.zeros((height,width,3),dtype=np.float64)]
    sigmas_mean = np.mean(sigmas,axis=1)
    radius = (2 * sigmas_mean).astype(np.int16)

    gaussian_weights_list = []
    # print(f"Sigmas Mean shape : {sigmas_mean.shape}")
    for j in range(sigmas_mean.shape[1]):
        x = np.arange(-radius[0,j], radius[0,j] + 1)
        y = np.arange(-radius[0,j], radius[0,j] + 1)
        sigma = sigmas_mean[0,j]
        X, Y = np.meshgrid(x, y)
        gaussian_weights = gaussian(X, Y, sigma) / np.sum(gaussian(X, Y, sigma))
        gaussian_weights_list.append(gaussian_weights)
    
    # print(len(gaussian_weights_list))
    # keypoints = np.round(keypoints)
    midpoints = []
    
    # print(keypoints)
    for i in range(keypoints.shape[0] - 1):

        midpoint1 = (4 * keypoints[i, 0, :] + keypoints[i + 1, 0, :]) / 5, (4 * keypoints[i, 1, :] + keypoints[i + 1, 1, :]) / 5, \
                    (4 * keypoints[i, 2, :] + keypoints[i + 1, 2, :]) / 5, keypoints[i,3,:] * keypoints[i+1,3,:]
        midpoint1 = np.expand_dims(np.stack(midpoint1), axis=0)
        
        midpoint2 = (3 * keypoints[i, 0, :] + 2 * keypoints[i + 1, 0, :]) / 5, (3 * keypoints[i, 1, :] + 2 * keypoints[i + 1, 1, :]) / 5, \
                    (3 * keypoints[i, 2, :] + 2 * keypoints[i + 1, 2, :]) / 5, keypoints[i,3,:] * keypoints[i+1,3,:]
        midpoint2 = np.expand_dims(np.stack(midpoint2), axis=0)
        
        midpoint3 = (2 * keypoints[i, 0, :] + 3 * keypoints[i + 1, 0, :]) / 5, (2 * keypoints[i, 1, :] + 3 * keypoints[i + 1, 1, :]) / 5, \
                    (2 * keypoints[i, 2, :] + 3 * keypoints[i + 1, 2, :]) / 5, keypoints[i,3,:] * keypoints[i+1,3,:]
        midpoint3 = np.expand_dims(np.stack(midpoint3), axis=0)
        
        midpoint4 = (keypoints[i, 0, :] + 4 * keypoints[i + 1, 0, :]) / 5, (keypoints[i, 1, :] + 4 * keypoints[i + 1, 1, :]) / 5, \
                    ( keypoints[i, 2, :] + 4 * keypoints[i + 1, 2, :]) / 5, keypoints[i,3,:] * keypoints[i+1,3,:]
        midpoint4 = np.expand_dims(np.stack(midpoint4), axis=0)

        midpoints.append(np.concatenate((midpoint1, midpoint2, midpoint3, midpoint4), axis=0))

    midpoints = np.concatenate(midpoints, axis=0)  
    del midpoint1,midpoint2,midpoint3,midpoint4

    all_points = np.concatenate([keypoints,midpoints],axis=0)
    del keypoints,midpoints

    conf_score=all_points[:, -2, :]
    conf_mask=conf_score>=conf_thres
    all_points[:,-2,:]=conf_mask
    del conf_score, conf_mask

    calculation_matrix = np.zeros([all_points.shape[0], 8, all_points.shape[2]])
    calculation_matrix[:,0,:] = np.maximum(0, all_points[:,0,:] - radius)
    calculation_matrix[:,1,:] = np.minimum(width, all_points[:,0,:] + radius + 1)
    calculation_matrix[:,2,:] = np.maximum(0, all_points[:,1,:] - radius)
    calculation_matrix[:,3,:] = np.minimum(height, all_points[:,1,:] + radius + 1)
    calculation_matrix[:,4,:] = np.maximum(0, radius - all_points[:,0,:])
    calculation_matrix[:,5,:] = np.minimum(2 * radius + 1, radius - all_points[:,0,:] + width)
    calculation_matrix[:,6,:] = np.maximum(0, radius - all_points[:,1,:])
    calculation_matrix[:,7,:] = np.minimum(2 * radius + 1, radius - all_points[:,1,:] + height)

    calculation_matrix = calculation_matrix.astype(np.int16)
    
    for i in range(all_points.shape[2]):
       
        limb_points = all_points[:,:,i]
        calculation_matrix_limb = calculation_matrix[:,:,i]
        splatted_image_limb = splatted_images[i]
        gaussian_weights_limb = gaussian_weights_list[i]
        # print(gaussian_weights_limb.shape)
        for j in range(limb_points.shape[0]):
            
            if limb_points[j,2] == 0 or limb_points[j,3]==0:
                continue

            roi_image = image[calculation_matrix_limb[j,2]:calculation_matrix_limb[j,3],calculation_matrix_limb[j,0]:calculation_matrix_limb[j,1]]
            roi_splatted_image = splatted_image_limb[calculation_matrix_limb[j,2]:calculation_matrix_limb[j,3],calculation_matrix_limb[j,0]:calculation_matrix_limb[j,1]]
            roi_gaussian_weights = gaussian_weights_limb[calculation_matrix_limb[j,6]:calculation_matrix_limb[j,7], calculation_matrix_limb[j,4]:calculation_matrix_limb[j,5]]

            if roi_gaussian_weights.shape[0] != roi_image.shape[0] or roi_gaussian_weights.shape[1] != roi_image.shape[1]:
                # # print(f"Shape mismatch before multiplication: roi_image: {roi_image.shape}, roi_gaussian_weights: {roi_gaussian_weights.shape}")
                min_height = min(roi_image.shape[0], roi_gaussian_weights.shape[0])
                min_width = min(roi_image.shape[1], roi_gaussian_weights.shape[1])
                roi_image = roi_image[:min_height, :min_width]
                roi_gaussian_weights = roi_gaussian_weights[:min_height, :min_width]
                roi_splatted_image = roi_splatted_image[:min_height, :min_width]

            if roi_image.ndim == 3:
                roi_gaussian_weights = roi_gaussian_weights[..., np.newaxis]

            roi_splatted_image += roi_gaussian_weights * roi_image

        if np.array_equal(splatted_image_limb, np.zeros((height, width, 3), dtype=np.float64)):
            limb_image = np.array(splatted_image_limb,dtype=np.uint8)
        else:
            limb_image  = np.array(splatted_image_limb/np.max(splatted_image_limb)*255,dtype=np.uint8)
        try:
            img_resized = cv2.cvtColor(cv2.resize(limb_image, (limb_image.shape[1]//8, limb_image.shape[0]//8)), cv2.COLOR_BGR2GRAY)
            visited= np.zeros(img_resized.shape, dtype=bool)
            max_x, max_y, min_x, min_y = bfs(img_resized ,0,0, visited)
            image_cropped = cv2.resize(limb_image[8*min_x:8*max_x,8*min_y:8*max_y], (256,256))
            limb_images_list.append(image_cropped)        
        
        except Exception as e:
            print(e)
            continue

    return limb_images_list
  
def load_data(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # Load the body bounding box (first line)
    x1_body, y1_body, x2_body, y2_body = map(int, lines[0].strip().split())

    # Load the face bounding box (second line)
    x1_face, y1_face, x2_face, y2_face = map(int, lines[1].strip().split())

    # Load the 17 keypoints from lines 3 to 19
    keypoints = []
    for line in lines[2:19]:  # Read lines 3 to 19 (index 2 to 18)
        x, y, confidence = map(float, line.strip().split())
        keypoints.append([x, y, confidence])

    # Convert keypoints list to a NumPy array of shape (17, 3)
    keypoints_array = np.array(keypoints)

    return (x1_body, y1_body, x2_body, y2_body), (x1_face, y1_face, x2_face, y2_face), keypoints_array
 
def bfs(image, startrow,startcol, visited):
    # Create a queue for BFS
    q = deque()
    row,col = image.shape
    max_x, min_x, max_y, min_y = 0, row, 0 ,col 
    # Mark the current node as visited and enqueue it
    visited[startrow][startcol] = True
    q.append([startrow,startcol])

    # Iterate over the queue
    while q:
        # Dequeue a vertex from queue and print it
        currentnode = q.popleft()
        currentrow,currentcol = currentnode
        color = image[currentrow, currentcol]
        if color!=0:
            if currentrow>max_x:
                max_x = currentrow
            if currentrow<min_x:
                min_x = currentrow
            if currentcol>max_y:
                max_y = currentcol
            if currentcol<min_y:
                min_y = currentcol

        # Get all adjacent vertices of the dequeued vertex
        # If an adjacent has not been visited, then mark it visited and enqueue it
        for i in range(-1,2):
            for j in range(-1,2):
                if currentrow-i>=0 and currentrow-i<row and currentcol-j>=0 and currentcol-j<col:
                    if not visited[currentrow-i][currentcol-j]:
                        visited[currentrow-i][currentcol-j] = True
                        q.append([currentrow-i,currentcol-j])
    return max_x, max_y, min_x, min_y

def calculate_sigma(keypoints, k=0.24):
    sigma = np.ones([1, 3, 4], dtype=np.float16)
    for i in range(keypoints.shape[2]): 
            sigma[0, 0, i] = np.linalg.norm(keypoints[0, :2, i] - keypoints[1, :2, i]) * k * keypoints[0,3,i] * keypoints[1,3,i]
            sigma[0, 1, i] = np.linalg.norm(keypoints[1, :2, i] - keypoints[2, :2, i]) * k * keypoints[1,3,i] * keypoints[2,3,i]
            sigma[0, 2, i] = np.linalg.norm(keypoints[2, :2, i] - keypoints[0, :2, i]) * k * keypoints[2,3,i] * keypoints[0,3,i]
    return sigma

def gaussian(x, y, sigma):
    return np.exp(-(x**2 + y**2) / (2.0 * sigma**2))

def make_gaussian(image_path, txt_path, output_folder,count):
    try:
        limb_folders = ['face','chest','larm', 'rarm', 'lleg', 'rleg']
        for limb in limb_folders:
            os.makedirs(os.path.join(output_folder, limb), exist_ok=True)

        # print(f"INSIDE GAUSSIAN CODE")
        count_k = count + 1
        os.makedirs(output_folder, exist_ok=True)
        img = cv2.imread(image_path)
        (x1_body, y1_body, x2_body, y2_body), (x1_face, y1_face, x2_face, y2_face), keypoints = load_data(txt_path)
        # print((x1_body, y1_body, x2_body, y2_body), (x1_face, y1_face, x2_face, y2_face), keypoints)
        
        person_cropped = img[y1_body:y2_body,x1_body:x2_body]
        
        left_shoulder = keypoints[5,:2]
        right_shoulder = keypoints[6,:2]
        left_hip = keypoints[11,:2]
        right_hip = keypoints[12,:2]
        
        x_min = int(min(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0]))
        y_min = int(min(left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]))
        x_max = int(max(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0]))
        y_max = int(max(left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]))
        if x_min<0 or x_max<0 or y_min<0 or y_max<0:
            pass
        else:
            chest_region = person_cropped[y_min:y_max, x_min:x_max]
            cv2.imwrite(os.path.join(output_folder, 'chest', f"chest_{count_k}.png"), chest_region)            
        
        # Left arm keypoints
        larm = np.expand_dims(np.concatenate((np.expand_dims(keypoints[5, :], axis=0), 
                                            np.expand_dims(keypoints[7, :], axis=0), 
                                            np.expand_dims(keypoints[9, :], axis=0)), axis=0), axis=2)

        # Right arm keypoints
        rarm = np.expand_dims(np.concatenate((np.expand_dims(keypoints[6, :], axis=0), 
                                            np.expand_dims(keypoints[8, :], axis=0), 
                                            np.expand_dims(keypoints[10, :], axis=0)), axis=0), axis=2)

        # Left leg keypoints
        lleg = np.expand_dims(np.concatenate((np.expand_dims(keypoints[11, :], axis=0), 
                                            np.expand_dims(keypoints[13, :], axis=0), 
                                            np.expand_dims(keypoints[15, :], axis=0)), axis=0), axis=2)

        # Right leg keypoints
        rleg = np.expand_dims(np.concatenate((np.expand_dims(keypoints[12, :], axis=0), 
                                            np.expand_dims(keypoints[14, :], axis=0), 
                                            np.expand_dims(keypoints[16, :], axis=0)), axis=0), axis=2)
        
        # Concatenating all four arrays along the channels (axis 0)
        bool_array = np.ones((3, 1, 4), dtype=bool)

        keypoints = np.concatenate((larm, rarm, lleg, rleg), axis=2)
        threshold = 1e-9
        
        # Get the indexes of elements that are 0 or very close to 0
        mask = np.abs(keypoints[:,:2,:]) >= threshold
        first_two_columns = mask[:, :2, :]

        # Combine the first two columns into a single column
        combined_column = np.any(first_two_columns, axis=1,keepdims=True)
        keypoints=np.concatenate((keypoints,combined_column),axis=1)

        del larm,rarm,lleg,rleg,bool_array
        sigmas = calculate_sigma(keypoints)
        # output_folder = "/home/uas-dtu/Documents/grand_finale/CLIP_VIRTUAL_LODA/DATASET/test/"
        # os.makedirs(output_folder,exist_ok=True)
        gaussian_start_time = time.time()
        
        limb_list = get_gaussian_and_bfs_limbs(person_cropped,keypoints,sigmas,conf_thres=0.0)            
        if not (x1_face == 0 and x2_face == 0):
            face = person_cropped[y1_face:y2_face,x1_face:x2_face]
            cv2.imwrite(os.path.join(output_folder, 'face', f"face_{count_k}.png"), face)
        # else:
            # print("face region image is empty so not saving.")
        # try:   
        # print(f"Saving images tp {output_folder}") 
        cv2.imwrite(os.path.join(output_folder, 'larm', f"larm_{count_k}.png"), limb_list[0])
        cv2.imwrite(os.path.join(output_folder, 'rarm', f"rarm_{count_k}.png"), limb_list[1])
        cv2.imwrite(os.path.join(output_folder, 'lleg', f"lleg_{count_k}.png"), limb_list[2])
        cv2.imwrite(os.path.join(output_folder, 'rleg', f"rleg_{count_k}.png"), limb_list[3])
        return count_k
    
    except Exception as e:
        print("HEllo")
        print(e)


def check_run_3(main_folder_path, pose_model, person_model, face_model, output_folder):
    
    os.makedirs(output_folder, exist_ok=True)
    visited_folders = []
    flag_dict = {}
    count_dict = {}
    occ_dict = {}
    txt_dict = {}
    lock = Lock()  
    number_of_txt_files = 200
    while True:
        if os.path.exists(main_folder_path):
            while True:
                folders = sorted(os.listdir(main_folder_path))
                with ThreadPoolExecutor() as executor:
                    
                    for folder in folders:
                        if folder not in visited_folders:
                            
                            # print(f"Inside Folder : {folder}")
                            
                            txt_files_number = 0
                            image_number = 0
                            
                            
                            with lock:
                                if folder not in flag_dict:
                                    flag_dict[folder] = True
                                    count_dict[folder] = 0
                                    occ_dict[folder] = 0
                                    txt_dict[folder] = 0
                                
                            folder_path = os.path.join(main_folder_path, folder)
                            images_and_txts = sorted(os.listdir(folder_path))
                            
                            # print(f"Images and Txt : {len(images_and_txts)}")
                            
                            image_list = []  
                            txt_list = []  
                            
                            for file in images_and_txts:
                                if file.endswith(('.jpg', '.png')):  
                                    image_number += 1
                                
                                    image_name, extension = os.path.splitext(file)
                                    image_path = os.path.join(folder_path, file)
                                    txt_path = os.path.join(folder_path, f"{image_name}.txt")
                                    output_subfolder = os.path.join(output_folder, folder)
                                    os.makedirs(output_subfolder, exist_ok=True)
                                    
                                    if not os.path.exists(txt_path) and txt_files_number < number_of_txt_files:
                                        
                                        # print("Processing image:", image_path)
                                        
                                        run_pose_person_face(image_path, pose_model, person_model, face_model)
                                        
                                        if os.path.exists(txt_path):
                                            txt_files_number +=1
                                            image_list.append(image_path)
                                            txt_list.append(txt_path)
                                            with lock:
                                                if count_dict[folder] < 50:
                                                    executor.submit(make_gaussian, image_path, txt_path, output_folder=output_subfolder, count=count_dict[folder])
                                                    count_dict[folder] += 1
                                    
                                            with lock:
                                                if flag_dict[folder] and count_dict[folder] >= 50:
                                                    images_and_txts = sorted(os.listdir(folder_path))
                                                    observation_start = float(images_and_txts[0].split("_")[-1].split(".")[0])
                                                    observation_end = float(images_and_txts[99].split("_")[-1].split(".")[0])
                                                    
                                                    # print(f"Running shell script for folder: {folder}")
                                                    
                                                    subprocess.Popen(["sh", "somin.sh", f"../../{output_subfolder}", f"{observation_start}", f"{observation_end}"])
                                                    flag_dict[folder] = False
                                                    
                                                    
                                        # else:
                                            # print(f"Txt file not created for {file}") 

                                # print(f"Lenght of Images List : {len(image_list)} Txt List : {len(txt_list)}")
                                print("images numer and text number",image_number,txt_files_number)
                                if txt_files_number >= 100 and image_number >= 100:
                                    print("yaha pr haiii")
                                    
                                    with lock:
                                        if occ_dict[folder] == 0: 
                                            
                                            # observation_end = float(image_list[99].split("_")[-1].split(".")[0])
                                            # print(observation_end)
                                            print(f"Checking ocular alertness for folder: {folder}")
                                            
                                            # result = ocular_model(folder,image_list, txt_list, observation_end)  
                                            subprocess.Popen(["sh", "ocular.sh", f"{folder}"])
                                            # print(f"Ocular alertness result: {result}")
                                            
                                            occ_dict[folder] = 1

                            if len(folders) >= len(visited_folders) + 2 and txt_files_number >= number_of_txt_files:
                                visited_folders.append(folder)
                                
                                print(f"Visited folders: {visited_folders}")


subprocess_queue = queue.Queue()

# Function to process tasks from the queue in the background
def worker():
    while True:
        script_args = subprocess_queue.get()  # Get the next subprocess args
        if script_args is None:  # Exit condition
            break
        subprocess.run(script_args)  # Run the shell script
        subprocess_queue.task_done()  # Mark task as done

# Start the worker thread
worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()


def check_run_4(main_folder_path, pose_model, person_model, face_model, output_folder):
    
    os.makedirs(output_folder, exist_ok=True)
    visited_folders = []
    clip_dict = {}
    count_dict = {}
    occ_dict = {}
    txt_dict = {}
    lock = Lock()  
    number_of_txt_files = 200
    while True:
        if os.path.exists(main_folder_path):
            while True:
                folders = sorted(os.listdir(main_folder_path))
                with ThreadPoolExecutor() as executor:
                    
                    for folder in folders:
                        if folder not in visited_folders:
                            
                            print(f"Inside Folder : {folder}")
                            
                            image_number = 0                            
                            
                            with lock:
                                if folder not in clip_dict:
                                    output_subfolder = os.path.join(output_folder, folder)
                                    os.makedirs(output_subfolder, exist_ok=True)
                                    clip_dict[folder] = True
                                    count_dict[folder] = 0
                                    occ_dict[folder] = True
                                    txt_dict[folder] = 0
                                
                            folder_path = os.path.join(main_folder_path, folder)
                            images_and_txts = sorted(os.listdir(folder_path))
                            
                            print(f"Images and Txt : {len(images_and_txts)}")
                            
                            image_list = []  
                            txt_list = []  
                            
                            for file in images_and_txts:
                                if file.endswith(('.jpg', '.png')):  
                                    image_number += 1
                                
                                    image_name, extension = os.path.splitext(file)
                                    image_path = os.path.join(folder_path, file)
                                    txt_path = os.path.join(folder_path, f"{image_name}.txt")
                                    
                                    if not os.path.exists(txt_path) and txt_dict[folder] < number_of_txt_files:
                                        
                                        print("Processing image:", image_path)
                                        
                                        run_pose_person_face(image_path, pose_model, person_model, face_model)
                                        
                                        if os.path.exists(txt_path):
                                            txt_dict[folder] +=1
                                            image_list.append(image_path)
                                            txt_list.append(txt_path)
                                            with lock:
                                                if count_dict[folder] < 50:
                                                    executor.submit(make_gaussian, image_path, txt_path, output_folder=output_subfolder, count=count_dict[folder])
                                                    count_dict[folder] += 1
                                    
                                            with lock:
                                                if clip_dict[folder] and count_dict[folder] >= 50:
                                                    images_and_txts = sorted(os.listdir(folder_path))
                                                    observation_start = float(images_and_txts[0].split("_")[-1].split(".")[0])
                                                    observation_end = float(images_and_txts[99].split("_")[-1].split(".")[0])
                                                    
                                                    # print(f"Running shell script for folder: {folder}")
                                                    
                                                    subprocess.Popen(["sh", "somin.sh", f"../../{output_subfolder}", f"{observation_start}", f"{observation_end}"])
                                                    clip_dict[folder] = False
                                                    
                                                    
                                        # else:
                                            # print(f"Txt file not created for {file}") 

                                # print(f"Lenght of Images List : {len(image_list)} Txt List : {len(txt_list)}")
                                
                                print(f"Folder : {folder} | Images : {image_number} | TxTs : {txt_dict[folder]}")
                                
                                # if txt_dict >= 100 and image_number >= 100:
                                #     print("yaha pr haiii")

                                #     with lock:
                                #         if occ_dict[folder] == True: 
                                            
                                #             # observation_end = float(image_list[99].split("_")[-1].split(".")[0])
                                #             # print(observation_end)
                                #             print(f"Checking ocular alertness for folder: {folder}")
                                            
                                #             # result = ocular_model(folder,image_list, txt_list, observation_end)  
                                #             subprocess.Popen(["sh", "ocular.sh", f"{folder}"])
                                #             # print(f"Ocular alertness result: {result}")
                                            
                                #             occ_dict[folder] = False

                            if txt_dict >= number_of_txt_files:
                                visited_folders.append(folder)
                                print(f"Visited folders: {visited_folders}")


if __name__=="__main__":
    pose_model,person_model,face_model = loader()
    output_folder = f"Gaussian_tests"
    check_run_4("images-sudhin",pose_model,person_model,face_model,output_folder)
    
    # image_path = f"images/waypt16_16_48.9272429840087_8.110779011656756_-31.695281982421875_385/image_00006.jpg"
    # txt_path = f"images/waypt16_16_48.9272429840087_8.110779011656756_-31.695281982421875_385/image_00006.txt"
    # output_folder = "Gaussian_test"
    # visualize_txt(image_path,txt_path)
    # st = time.time()
    # make_gaussian(image_path, txt_path, output_folder)
    # print(time.time() - st)