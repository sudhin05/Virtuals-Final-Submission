import argparse
import sys
import threading 

sys.path.append('/home/uas-dtu/SudhinDarpa/grand_finale/CLIP_VIRTUAL_LODA')
from report_publisher import pub  

def rd_result_maker(folder):
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
            "respiratory_distress": True,
            "alertness": {
              "verbal": "absent"
        } 
            }
        
    }
        
    return json_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AUDIO in the specified folder.")
    parser.add_argument('--root_path', type=str, required=True, help='Path to the root folder containing images')

    # Parse arguments
    args = parser.parse_args()

    
    json_report = rd_result_maker(args.root_path)
    
    t5 = threading.Thread(target=pub, args=(json_report,))
    t5.start()