import os
import json

fewevent_dir = "FewEvent"
output_dir = "/code/ltt_code/few_event/data/fewevent"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
modes = ["train","dev","test"]

event_split = {}
all_data = {}
for mode in modes:
    data = open(os.path.join(fewevent_dir,"meta_{}_dataset.json".format(mode))).read()
    data = json.loads(data)
    event_split[mode] = list(data.keys())
    for k,v in data.items():
        all_data[k] = v

count = 0
for k,v in all_data.items():
    instances = []
    event_type = k
    for i in v:
        id = count
        tokens = i["tokens"]
        sentence = " ".join(tokens)
        trigger = i["trigger"] 
        start = i["position"][0]
        end = i["position"][1]
        instances.append({
            "id":id,
            "sentence":sentence,
            "tokens":tokens,
            "events":[{
                "text":trigger,
                "start":start,
                "end":end,
                "event_type":event_type
            }]
        })
        count += 1
    all_data[k] = instances

def get_data(mode):
    data = {}
    events = event_split[mode]
    for k,v in all_data.items():
        if k in events:
            data[k] = v
    return data
train, dev, test = get_data("train"), get_data("dev"), get_data("test")

with open(os.path.join(output_dir,"train.json"), "w") as f:
    json.dump(train,f,indent=2)
with open(os.path.join(output_dir,"dev.json"), "w") as f:
    json.dump(dev,f,indent=2)
with open(os.path.join(output_dir,"test.json"), "w") as f:
    json.dump(test,f,indent=2)
with open(os.path.join(output_dir,"event_split.json"), "w") as f:
    json.dump(event_split,f,indent=2)
