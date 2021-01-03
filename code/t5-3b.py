
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import torch.nn.functional as F


# %%

tokenizer = T5Tokenizer.from_pretrained("t5-3b")
t5 = T5ForConditionalGeneration.from_pretrained("t5-3b")
device_map = {
            0: [0, 1, 2],

             1: [3, 4, 5, 6, 7, 8, 9],
             2: [10, 11, 12, 13, 14, 15, 16],
             3: [17, 18, 19, 20, 21, 22, 23]}
t5.parallelize(device_map)
torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

import json
import os
import tqdm

realLabel = []
predLabel = []

json_open = open('test.jsonl', 'r')
foutput = open("T53bpredictiontest.txt", "w")
for text in tqdm.tqdm(json_open.readlines()):
    json_load = json.loads(text)
    inputText = "boolq question: " + json_load['question'] + ". " + "passage: " + json_load['passage'] + " answer:"
    
    with torch.no_grad():
        enc = tokenizer(inputText, return_tensors="pt", add_special_tokens=True).to('cuda:0')
        decoder_input_ids = torch.tensor([tokenizer.pad_token_id]).unsqueeze(0).cuda()
        logits = t5(**enc, decoder_input_ids=decoder_input_ids)[0]
        tokens = torch.argmax(logits, dim=2)
        sentiments = tokenizer.batch_decode(tokens)
        # print(sentiments)
        logits = logits.squeeze(1)
        selected_logits = logits[:, [10998, 10747]] # True Fal|se
        probs = F.softmax(selected_logits, dim=1)
        print(probs.cpu().numpy(), file=foutput)
    


# %%
def PrintAnalysis(predLabel):
              # Real Pred
    TP = 0    # 1    1
    FN = 0    # 1    0
    FP = 0    # 0    1
    TN = 0    # 0    0

    for i in range(len(realLabel)):
        if realLabel[i] == 1 and predLabel[i] == 1:
            TP += 1
        if realLabel[i] == 1 and predLabel[i] == 0:
            FN += 1
        if realLabel[i] == 0 and predLabel[i] == 1:
            FP += 1
        if realLabel[i] == 0 and predLabel[i] == 0:
            TN += 1

    print("TP = %d" % TP)
    print("FN = %d" % FN)
    print("FP = %d" % FP)
    print("TN = %d" % TN)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2 * precision * recall) / (precision + recall)

    print("accuracy of human label: %.4f" % accuracy)
    print("precision of human label: %.4f" % precision)
    print("recall of human label: %.4f" % recall)
    print("F1 of human label: %.4f" % F1)


# %%
#PrintAnalysis(predLabel)
# PrintAnalysis(predLabel2)


# %%



