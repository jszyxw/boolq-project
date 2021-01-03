import json
import os

if __name__ = "__main__":

    json1 = open('../data/trainNew.jsonl', 'r')
    json2 = open('../data/data_aug.jsonl', 'r')
    with open('../data/closed_train_aug.jsonl', 'w') as file:
        for text in json1.readlines():
            json_load = json.loads(text)
            dic = {
                'QnP': json_load['question'] + ' $ ' + json_load['passage'],
                'label': json_load['answer']
            }
            file.write(json.dumps(dic) + '\n')
        for text in json2.readlines():
            json_load = json.loads(text)
            dic = {
                'QnP': json_load['question'] + ' $ ' + json_load['passage'],
                'label': json_load['answer']
            }
            file.write(json.dumps(dic) + '\n')

    json_open = open('../data/devNew.jsonl', 'r')
    with open('../data/closed_dev.jsonl', 'w') as file:
        for text in json_open.readlines():
            json_load = json.loads(text)
            dic = {
                'QnP': json_load['question'] + ' $ ' + json_load['passage'],
                'label': json_load['answer']
            }
            file.write(json.dumps(dic) + '\n')

    json_open = open('../data/test.jsonl', 'r')
    with open('../data/closed_test.jsonl', 'w') as file:
        for text in json_open.readlines():
            json_load = json.loads(text)
            dic = {
                'QnP': json_load['question'] + ' $ (' + json_load['title'][:-1] +') ' + json_load['passage'],
            }
            file.write(json.dumps(dic) + '\n')