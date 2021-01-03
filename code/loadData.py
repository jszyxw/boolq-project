import json
def load_data():
    train_data = []
    dev_data = []
    label_id = {True: 1, False: 0}

    f_train = open("trainNew.jsonl", encoding='utf-8')
    # f_dev = open("devNew.jsonl", encoding='utf-8')
    f_dev = open("testNew.jsonl", encoding='utf-8')

    for line in f_train:
        dic = json.loads(line)
        train_data.append([dic['question'], dic['passage'], label_id[dic['answer']]])

    for line in f_dev:
        dic = json.loads(line)
        dic['answer'] = True
        dev_data.append([dic['question'], dic['passage'], label_id[dic['answer']]])

    return train_data, dev_data

def load_data_aug():
    f_aug = open("data_aug.jsonl", encoding='utf-8')
    label_id = {True: 1, False: 0}
    aug_data = []
    for line in f_aug:
        dic = json.loads(line)
        aug_data.append([dic['question'], dic['passage'], label_id[dic['answer']]])
    return aug_data

if __name__ == '__main__':
    train_data, dev_data = load_data()
    # print(dev_data[0])
    # print(len(dev_data[0]))
