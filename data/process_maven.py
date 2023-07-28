import os
import json
import random
from const import *


dataset = 'maven'

labels = LABEL[dataset]
print(labels)
print(len(labels))

ace_dir = os.path.join("/code/ltt_code/event_detection/data/",dataset)
output_dir = os.path.join("./",dataset)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

modes = ["train","dev","test"]
data = []
for mode in modes:
    with open(os.path.join(ace_dir,mode+".json"),"r") as f:
        data.extend(json.load(f))
print("raw_data:",len(data))

new_data = []
for i in data:
    if i["events"] == []:
        continue
    new_data.append(i)
print("remove negative",len(new_data))


raw_dataset = {l:[] for l in labels}
for i in new_data:
    events = []
    for j in i["events"]:
        raw_dataset[j["event_type"]].append(i)

g = sorted(raw_dataset.items(), key = lambda e:len(e[1]),reverse=True)

classes = [ i[0] for i in g]
classes = classes[:100]

train_event = classes[:64]

new_classes = classes[64:]
random.shuffle(new_classes)

dev_event = new_classes[:16]
test_event = new_classes[16:]

print(len(train_event),len(dev_event),len(test_event))


train_dataset = { i:[] for i in train_event}
dev_dataset = { i:[] for i in dev_event}
test_dataset = { i:[] for i in test_event}

for i in new_data:
    events = []
    for j in i["events"]:
        events.append(j["event_type"])
    events = set(events)
    if events < set(train_event):
        for e in events:
            train_dataset[e].append(i)
    elif events < set(dev_event):
        for e in events:
            dev_dataset[e].append(i)
    elif events < set(test_event):
        for e in events:
            test_dataset[e].append(i)
    else:
        pass

def print_each_event(dataset):
    for k,v in dataset.items():
        print(k,len(v))

print("train:")
print_each_event(train_dataset)

print("dev:")
print_each_event(dev_dataset)

print("test:")
print_each_event(test_dataset)

with open(os.path.join(output_dir,"train"+".json"),"w") as f:
    json.dump(train_dataset,f,indent=2)

with open(os.path.join(output_dir,"dev"+".json"),"w") as f:
    json.dump(dev_dataset,f,indent=2)

with open(os.path.join(output_dir,"test"+".json"),"w") as f:
    json.dump(test_dataset,f,indent=2)

event_split = {
    "train": train_event,
    "dev": dev_event,
    "test": test_event
}

with open(os.path.join(output_dir,"event_split"+".json"),"w") as f:
    json.dump(event_split,f,indent=2)

# check