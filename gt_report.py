import os
import pandas as pd

folder_path = '/home/uas-dtu/Desktop/final_docker_maybe/dtcvc-public-7e4ce49-r20240822/data/ground_truth'

results = []
results1 = []
results2 = []
results3 = []
results4 = []



for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        
        df = pd.read_csv(file_path)
        
        if 'trauma_lower_ext' in df.columns:
            trauma_lower_ext = df['trauma_lower_ext'].dropna().unique()
            
            if len(trauma_lower_ext) == 1:
                result = f"{filename}: {trauma_lower_ext[0]}"
            else:
                result = f"{filename}: {', '.join(map(str, trauma_lower_ext))}"
        else:
            result = f"{filename}: Column 'trauma_lower_ext' not found"
        
        results.append(result)

        if 'trauma_upper_ext' in df.columns:
            trauma_upper_ext = df['trauma_upper_ext'].dropna().unique()
            
            if len(trauma_upper_ext) == 1:
                result = f"{filename}: {trauma_upper_ext[0]}"
            else:
                result = f"{filename}: {', '.join(map(str, trauma_upper_ext))}"
        else:
            result = f"{filename}: Column 'trauma_upper_ext' not found"
        
        results1.append(result)
        
        if 'trauma_torso' in df.columns:
            trauma_torso = df['trauma_torso'].dropna().unique()
            
            if len(trauma_torso) == 1:
                result = f"{filename}: {trauma_torso[0]}"
            else:
                result = f"{filename}: {', '.join(map(str, trauma_torso))}"
        else:
            result = f"{filename}: Column 'trauma_torso' not found"
        
        results2.append(result)
        if 'trauma_head' in df.columns:
            trauma_head = df['trauma_head'].dropna().unique()
            
            if len(trauma_head) == 1:
                result = f"{filename}: {trauma_head[0]}"
            else:
                result = f"{filename}: {', '.join(map(str, trauma_head))}"
        else:
            result = f"{filename}: Column 'trauma_head' not found"
        
        results3.append(result)
        # if 'alertness_verbal' in df.columns:
        #     alertness_verbal = df['alertness_verbal'].dropna().unique()
            
        #     if len(alertness_verbal) == 1:
        #         result = f"{filename}: {alertness_verbal[0]}"
        #     else:
        #         result = f"{filename}: {', '.join(map(str, alertness_verbal))}"
        # else:
        #     result = f"{filename}: Column 'alertness_verbal' not found"
        
        # results4.append(result)




results.sort()
results1.sort()
results2.sort()
results3.sort()
results4.sort()

with open('trauma_lower_ext_report.txt', 'w') as f:
    for result in results:
        f.write(result + '\n')
        
with open('trauma_upper_ext_report.txt', 'w') as f:
    for result in results1:
        f.write(result + '\n')
        
with open('trauma_torso_report.txt', 'w') as f:
    for result in results2:
        f.write(result + '\n')
        
with open('trauma_head_report.txt', 'w') as f:
    for result in results3:
        f.write(result + '\n')
        
# with open('verbal_report.txt', 'w') as f:
#     for result in results4:
#         f.write(result + '\n')

print("Sorted report generated")
