# from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import torch
import json
import os
from nltk.corpus import framenet as fn
# from nltk.corpus import wordnet as wn
from data.const import *

model_name_or_path = "/models/all-MiniLM-L6-v2"
gpt4_frame_definition_path = "./definitions/gpt4_frame_definition.json"

#Mean Pooling - Take attention mask into account for correct averaging
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
#     sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#     return sum_embeddings / sum_mask

# def get_definitions(event):
#     definitions = []
#     words = event.split("_")
#     if len(words) > 1:
#         definitions.append(" ".join(words)) 
#     else:
#         word = words[0]
#         syns = wn.synsets(word)
#         if syns == []:
#             definitions.append(word)
#         else:
#             for w in syns:
#                 definitions.append(w.definition())
#     return definitions

    
def get_framenet(dataset, top_k = 20):
    events = LABEL[dataset]
    frames = fn.frames()
    frame_definition = json.load(open(gpt4_frame_definition_path,"r"))
    event_definition = json.load(open("./definitions/{}_definition.json".format(dataset),"r"))

    #Load AutoModel from huggingface model repository
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # model = AutoModel.from_pretrained(model_name_or_path)

    embedder = SentenceTransformer(model_name_or_path)
    corpus = list(frame_definition.values())
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    # fname = [ f.name for f in frames]
    event2framenet = {}
    for idx,event in enumerate(events):
        result = []
        definition = event_definition[event]

        query_embedding = embedder.encode(definition, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        for i in top_results[1]:
            # result.append(fname[i])
            if frames[i].lexUnit != {}: 
                result.append(frames[i])
        filter_result = result[0]
        for res in result:
            if res.name == event:
                filter_result = res
                break
            if event in res.name:
                filter_result = res
                break
            # synsets = []
            # for word in wn.synsets(event):
            #     synsets.extend(word.lemma_names())
            # synsets = list(set(synsets))

            # for i in synsets:
            #     if i in res.lower():
            #         filter_result.append(res)
            #         break
        event2framenet[event] = filter_result
    return event2framenet
        # print("idx:{}, event:{}, frames:{}".format(idx+1, event, filter_result))

if __name__ == '__main__':
    result = get_framenet('maven')