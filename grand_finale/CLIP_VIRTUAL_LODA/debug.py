import sys
sys.path.append('grand_finale/CLIP_VIRTUAL_LODA')
from report_publisher import pub

metadata = {
            "observation_start": 0,
            "observation_end": 0,
            "assessment_time": 0,
            "casualty_id": 0,  
            "drone_id": 0,  
            "location": {
                "lon": 0,
                "lat": 0,
                "alt": 0
            },
            "vitals": {
                "heart_rate": None, 
                "respiration_rate": None 
            },
            "injuries": {
                "severe_hemorrhage": None, 
                "respiratory_distress": None,  
                "trauma": None,  
                "alertness": {
                    "ocular": None, 
                    "verbal": None, 
                    "motor": None 
                }
            }
        }
pub(metadata)