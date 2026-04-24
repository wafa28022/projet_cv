import json

with open(r'C:\Users\msi\Desktop\ma-part\dmd\gB\10\s2\gB_10_s2_2019-03-11T15;15;21+01;00_rgb_ann_distraction.json', 'r') as f:
    data = json.load(f)

actions = data['openlabel']['actions']
print("Actions dans S2 :")
for action_id, action_data in actions.items():
    print(f"  {action_data['type']}")