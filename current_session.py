# coding: utf-8
import json
filename = "./runs-data-famo.json"
with open(filename) as f:
    data = json.load(f)
    
data
uniformity = [d["uniformity"] for d in data]
diversity = [d["diversity"] for d in data]
novelty = [d["novelty"] for d in data]
data
data[0
]
data[0]
data.keys()
data['Scattered-best_reward']
data['Scattered-best_reward'][0]
data['Scattered-best_reward'][0][0]
for run in data['Scattered-best_reward']:
    
    
    r
    
unique_data = []
unique_data = {}
for run in data['Scattered-best_reward']:
    
    
    r
  
for run in data['Scattered-best_reward']:
    for step in run:
        if not f"{step['input_set_id']}-{step['operator']}-{step['parameter']}" in unique_data:
            unique_data[f"{step['input_set_id']}-{step['operator']}-{step['parameter']}"] = step
            
unique_data
len(unique_data
)
len(data['Scattered-best_reward'][0])
filename = "./runs-data.json"
with open(filename) as f:
    data = json.load(f)
    
for run in data['Scattered-best_reward']:
    for step in run:
        if not f"{step['input_set_id']}-{step['operator']}-{step['parameter']}" in unique_data:
            unique_data[f"{step['input_set_id']}-{step['operator']}-{step['parameter']}"] = step
            
len(unique_data
)
with open(filename) as f:
    data = json.load(f)
    
for run in data['Scattered-best_reward']:
    for step in run:
        if not f"{step['input_set_id']}-{step['operator']}-{step['parameter']}" in unique_data:
            unique_data[f"{step['input_set_id']}-{step['operator']}-{step['parameter']}"] = step
            
len(unique_data
)
get_ipython().run_line_magic('save', 'current_session ~0/')
