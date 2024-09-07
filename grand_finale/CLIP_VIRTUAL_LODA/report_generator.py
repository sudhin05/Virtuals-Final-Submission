import json

# json_file_path = '/home/uas-dtu/Desktop/nikhil-darpa/trauma_data.json'

# # Read the existing data from the JSON file (if it exists)
# try:
#     with open(json_file_path, 'r') as f:
#         data = json.load(f)
# except FileNotFoundError:
#     data = {}


# json_file_path = '/home/uas-dtu/Desktop/nikhil-darpa/outputs.json'

# # Read the existing data from the JSON file (if it exists)
# try:
#     with open(json_file_path, 'r') as f:
#         data_2 = json.load(f)
# except FileNotFoundError:
#     data_2 = {}
    
# trauma=data.get("trauma", {})
class DtcvcTriageReportGenerator():
    """Utility class for generating random triage reports for debugging and testing"""

    REPORT_VALID = "valid"
    REPORT_RANDOM_VITALS = "random-vitals"
    REPORT_RANDOM_INJURIES = "random-injuries"
    REPORT_RANDOM_COMPLETE = "random-complete"

    REPORT_TYPES = [
        REPORT_VALID,
        REPORT_RANDOM_VITALS,
        REPORT_RANDOM_INJURIES,
        REPORT_RANDOM_COMPLETE,
    ]

    def __init__(self, casuality_id, lat , lon ,alt , obs_start,obs_end,trauma ,drone_count=2, casualty_count=10):
        self.drone_count = drone_count
        self.casualty_count = casualty_count

    def publish_random(self, publisher):
        self.publish(
            publisher,
            random.sample(DtcvcTriageReportGenerator.REPORT_TYPES, 1)[0],
            random.randint(1, self.drone_count),
            random.randint(1, self.casualty_count),
        )

    def publish(self, publisher, report_type: str, drone_id: int, casualty_id: int):
        report = None
        match report_type:
            case DtcvcTriageReportGenerator.REPORT_VALID:
                report = self.generate_valid_victim_report(drone_id=drone_id, casualty_id=casualty_id)
            # case DtcvcTriageReportGenerator.REPORT_RANDOM_VITALS:
            #     report = self.generate_random_victim_vitals_only(drone_id=drone_id, casualty_id=casualty_id)
            # case DtcvcTriageReportGenerator.REPORT_RANDOM_INJURIES:
            #     report = self.generate_random_victim_injuries_only(drone_id=drone_id, casualty_id=casualty_id)
            # case DtcvcTriageReportGenerator.REPORT_RANDOM_COMPLETE:
            #     report = self.generate_random_victim_complete_report(drone_id=drone_id, casualty_id=casualty_id)
        if report is not None:
            publisher.publish(String(data=json.dumps(report)))
    
    
    def generate_valid_victim_report(self, drone_id ,casuality_id, lat , lon ,alt , obs_start,obs_end,trauma) -> dict:
        """
    Generates a report that can dynamically include trauma and other injury details
    Args:
            drone_id (int): The ID of the drone
            casualty_id (int): The ID of the casualty
            trauma (dict): A dictionary containing trauma information to be added to the report

        Returns:
            dict: The victim report with the provided data
    """
            
        default_trauma = {
            "head": "wound",
            "torso": "normal",
            "upper_extremity": "wound",
            "lower_extremity": "normal"
        }

        trauma = trauma if trauma else default_trauma
        print(trauma)
        # {'face': 'normal', 'torso': 'normal', 'upper_extremity': 'wound', 'lower_extremity': 'normal'}

        
        report = {
            "observation_start": float(obs_start),
            "observation_end": float(obs_end),
            "assessment_time": float(obs_end),
            "casualty_id": int(casuality_id),
            "drone_id": int(drone_id),
            "location": {"lon": float(lon), "lat": float(lat), "alt": float(alt)},
            "vitals": {"heart_rate": 80, "respiration_rate": 50},
            "injuries": {
                "severe_hemorrhage": True,
                "respiratory_distress": True,
                "trauma": trauma,
                "alertness": {
                    "ocular": "open",
                    "verbal": "abnormal",
                    "motor": "absent",
                },
            },
        }
        
        return report
    
# {'observation_start': 10.0, 
#  'observation_end': 14.0, 
#  'assessment_time': 14.0, 
#  'casualty_id': '14', 
#  'drone_id': 1, 
#  'location': {'lon': '8.109561293040809', 'lat': '48.927230464786334', 'alt': '-32.223052978515625'},
#  'vitals': {'heart_rate': 80, 'respiration_rate': 50}, 
#  'injuries': {'severe_hemorrhage': True, 
#               'respiratory_distress': True, 
#               'trauma': {'head': 'normal', 'torso': 'normal', 'upper_extremity': 'wound', 'lower_extremity': 'normal'}, 
#               'alertness': {'ocular': 'open', 
#                             'verbal': 'abnormal', 
#                             'motor': 'absent'}}}


    
