import json
import os
import sys

from preprocess import preprocess
from lstm import LSTM_Attention
from train import App_Emb



def predict(app):
    with open('./result_close.txt', 'w') as result:
        json_open = open('../data/closed_test.jsonl', 'r')
        for text in json_open.readlines():
            text = json.loads(text)
            l = preprocess([text['QnP']])
            pred = 1 if app.predict(l)>0.5 else 0
            result.write(str(pred)+"\n")


if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1] == 'train':
        app = App_Emb()
        app.train()
    else:
        app = App_Emb(load=True)
        app.test(app.train_iter)
        app.test(app.test_iter)
        predict(app)


