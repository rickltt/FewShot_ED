with open("./event_split.json") as f:
    import json
    data = json.load(f)

event_types = []
for k,v in data.items():
    event_types.extend(v)

print(event_types)
print(len(event_types))