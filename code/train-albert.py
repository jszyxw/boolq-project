import torch
from transformers import AlbertConfig, AlbertTokenizer, AlbertModel
import transformers
import torch.nn as nn
import torch.nn.functional as F
import loadData
from torch.utils.data import RandomSampler, DataLoader, TensorDataset, SequentialSampler
import numpy as np
from tqdm import tqdm, trange
from torch.utils.checkpoint import checkpoint


class AlbertQA(nn.Module):
    def __init__(self, config, num_label):
        super(AlbertQA, self).__init__()
        self.albert = AlbertModel.from_pretrained('albert-xxlarge-v1')
        self.fc = nn.Linear(config.hidden_size, num_label)
        self.drop = nn.Dropout(config.hidden_dropout_prob)
        self.loss = nn.CrossEntropyLoss(reduction='sum')

        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias, 0.)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        _, pooled_output = self.albert(input_ids=input_ids, token_type_ids=token_type_ids,
                                        attention_mask=attention_mask)
        # _, pooled_output = checkpoint(self.roberta, input_ids, attention_mask, token_type_ids)
        pooled_output = self.drop(pooled_output)
        logits = self.fc(pooled_output)

        if labels is not None:
            loss = self.loss(logits, labels)
            return loss, logits
        else:
            return logits


def get_features(samples, max_len, tknzr, label_list=None):
    '''
    :param samples: sents. sents[i] = [question, "(title) passage", label]
    :param max_len:
    :param tknzr:
    :param label_list: if label_list is not None, should map elements in label_list to 0, 1, ..., len(label_list)-1
    :return: input_id, segment_id (0 vectors), input_mask_id, label_id
    '''
    label_id = []
    input_ids = []
    input_masks_ids = []
    segment_ids = []
    for idx, ls in enumerate(samples):
        sent1 = ls[0]
        sent2 = ls[1]
        label = ls[2]
        dic = tknzr.encode_plus(sent1, sent2, max_length=max_len, pad_to_max_length=True, return_token_type_ids=True,
                                return_tensors='pt', truncation=True)
        label_id.append(label)
        input_ids.append(dic['input_ids'])
        segment_ids.append(dic['token_type_ids'])
        input_masks_ids.append(dic['attention_mask'])
    return input_ids, input_masks_ids, segment_ids, label_id


def create_dataloader(input_ids, mask_ids, segments_ids, label_ids, batch_size, train=True):
    data = TensorDataset(input_ids, mask_ids, segments_ids, label_ids)
    if train:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, drop_last=train)

    return dataloader


def main():
    device_ids = [2,3]

    init_lr = 1e-5
    max_epochs = 10
    max_length = 512
    batch_size = 2
    gradient_accu = 32 // batch_size

    num_label = 2

    train_mode = False

    prev_acc = 0.
    max_acc = 0.

    config = AlbertConfig.from_pretrained('albert-xxlarge-v1')
    tknzr = AlbertTokenizer.from_pretrained('albert-xxlarge-v1')

    train_data, test_data = loadData.load_data()
    train_data = train_data + loadData.load_data_aug()

    train_input_ids, train_mask_ids, train_segment_ids, train_label_ids = get_features(train_data, max_length, tknzr)
    test_input_ids, test_mask_ids, test_segment_ids, test_label_ids = get_features(test_data, max_length, tknzr)

    # print(all_input_ids.shape)

    all_input_ids = torch.cat(train_input_ids, dim=0).long()
    all_input_mask_ids = torch.cat(train_mask_ids, dim=0).long()
    all_segment_ids = torch.cat(train_segment_ids, dim=0).long()
    all_label_ids = torch.Tensor(train_label_ids).long()
    train_dataloader = create_dataloader(all_input_ids, all_input_mask_ids, all_segment_ids, all_label_ids,
                                         batch_size=batch_size, train=True)

    all_input_ids = torch.cat(test_input_ids, dim=0).long()
    all_input_mask_ids = torch.cat(test_mask_ids, dim=0).long()
    all_segment_ids = torch.cat(test_segment_ids, dim=0).long()
    all_label_ids = torch.Tensor(test_label_ids).long()
    test_dataloader = create_dataloader(all_input_ids, all_input_mask_ids, all_segment_ids, all_label_ids,
                                        batch_size=batch_size, train=False)

    model = AlbertQA(config, num_label=num_label).cuda(device_ids[0])
    print(device_ids)
    model = torch.nn.DataParallel(model, device_ids=device_ids)

    optimizer = transformers.AdamW(model.parameters(), lr=init_lr, eps=1e-8)
    optimizer.zero_grad()
    # scheduler = transformers.get_constant_schedule_with_warmup(optimizer, len(train_dataloader) // (batch_size * gradient_accu))
    # scheduler = transformers.get_linear_schedule_with_warmup(optimizer, len(train_dataloader) // (batch_size * gradient_accu), (len(train_dataloader) * max_epochs * 2) // (batch_size * gradient_accu), last_epoch=-1)

    if not train_mode:
        max_epochs = 1
        model.load_state_dict(torch.load("model/albertmodel.ckpt"))

    foutput = open("answer-albert-mnli.txt", "w")

    global_step = 0
    for epoch in range(max_epochs):
        model.train()
        if train_mode:
            loss_avg = 0.
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                global_step += 1
                batch = [t.cuda() for t in batch]
                input_id, input_mask, segment_id, label_id = batch
                #print(input_id)
                #print(input_mask)
               # print(segment_id)
               # print(label_id)
                loss, _ = model(input_id, segment_id, input_mask, label_id)
                loss = torch.sum(loss)
                loss_avg += loss.item()
                loss = loss / (batch_size * gradient_accu)
                loss.backward()
                if global_step % gradient_accu == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    # if epoch == 0:
                    # scheduler.step()
            print(loss_avg / len(train_dataloader))

        model.eval()

        final_acc = 0.
        num_test_sample = 0
        tot = [0, 0]
        correct = [0, 0]
        for input_id, input_mask, segment_id, label_id in test_dataloader:
            input_id = input_id.cuda()
            input_mask = input_mask.cuda()
            segment_id = segment_id.cuda()
            label_id = label_id.cuda()

            
            loss, logit = model(input_id, segment_id, input_mask, label_id)
            logit = logit.detach().cpu().numpy()
            # print model prediction result
            print(logit[0],file=foutput)
            print(logit[1],file=foutput)
            label_id = label_id.to('cpu').numpy()
            acc = np.sum(np.argmax(logit, axis=1) == label_id)
            pred = np.argmax(logit, axis=1)
            for i in range(label_id.shape[0]):
                tot[label_id[i]] += 1
                if pred[i] == label_id[i]:
                    correct[label_id[i]] += 1
            final_acc += acc
            num_test_sample += input_id.size(0)

        print("epoch:", epoch)
        print("final acc:", final_acc / num_test_sample)
        if final_acc / num_test_sample > max_acc:
            max_acc = final_acc / num_test_sample
            print("save...")
            torch.save(model.state_dict(), "model/model1.ckpt")
            print("finish")
        print("Max acc:", max_acc)
        '''
        if final_acc / num_test_sample <= prev_acc:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.8
        '''
        prev_acc = final_acc / num_test_sample
        tp = correct[1]
        tn = correct[0]
        fp = tot[1] - correct[1]
        fn = tot[0] - correct[0]
        rec = tp / (tp + fn)
        pre = tp / (tp + fp)
        print("recall:{0}, precision:{1}".format(rec, pre))
        print("f:", 2 * pre * rec / (pre + rec))
        print("acc:", (tp + tn) / (tp + tn + fp + fn))


if __name__ == '__main__':
    main()
