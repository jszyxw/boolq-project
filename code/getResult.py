
import json
import numpy


def f1_loss(y_true, y_pred, beta=1):
    '''Calculate F1 score.
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    
    tp = (y_true * y_pred).sum()
    tn = ((1 - y_true) * (1 - y_pred)).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    
    f1 = (1 + beta**2)* (precision*recall) / (beta**2 * precision + recall + epsilon)

    print("precision = ", precision)
    print("recall = ", recall)
    print("f1 = ", f1)
    print("accuracy = ", accuracy)


f = open("devNew.jsonl", "r")

answer = []

for line in f.readlines():
    x = json.loads(line)
    # print(x['answer'])
    answer.append({'True':1, 'False':0}[str(x['answer'])])

fileName = ['answer-albert-normalized.txt', 'answer-roberta-normalized.txt', 'answer-T5-3b.txt', 'answer-deberta-normalized.txt']
# fileName = ['answer-albert.txt', 'answer-roberta.txt']

predList = []

for name in fileName:
    filePred = open(name, "r")
    filePred.readline()
    predList.append([list(map(float, line.split())) for line in filePred.readlines()])

wrong = 0

answerSet = [[], [], [], []]

for i in range(len(answer)):
    answerPred = []
    for j in range(len(fileName)):
        if predList[j][i][0] > predList[j][i][1]:
            answerPred.append(0)
        else:
            answerPred.append(1)
        answerSet[j].append(answerPred[-1])

    # print(answer[i], *answerPred)


for j in range(len(fileName)):
    print(fileName[j])
    f1_loss(numpy.array(answer), numpy.array(answerSet[j]))
