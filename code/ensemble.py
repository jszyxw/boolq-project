
import json
import numpy

f = open("devNew.jsonl", "r")

answer = []

for line in f.readlines():
    x = json.loads(line)
    # print(x['answer'])
    answer.append(str(x['answer']))

fileName = ['answer-albert-normalized.txt', 'answer-roberta-normalized.txt', 'answer-T5-3b.txt', 'answer-deberta-normalized.txt']
# fileName = ['answer-albert.txt', 'answer-roberta.txt']

predList = []

for name in fileName:
    filePred = open(name, "r")
    filePred.readline()
    predList.append([list(map(float, line.split())) for line in filePred.readlines()])

wrong = 0
fw = open("dev-result.txt", "w")

for i in range(len(answer)):
    answerPred = []
    PredFalse = []
    PredTrue = []
    conf = []
    for j in range(len(fileName)):
        PredFalse.append(predList[j][i][0])
        PredTrue.append(predList[j][i][1])
        if predList[j][i][0] > predList[j][i][1]:
            answerPred.append('False')
        else:
            answerPred.append('True')
        conf.append(abs(predList[j][i][0] - predList[j][i][1]))
        coefficient = numpy.array([3.4, 1, 0.0, 2])
    if answerPred.count('True') != answerPred.count('False'):
        predAnswer = 'True' if answerPred.count('True') > answerPred.count('False') else 'False'
    else:
        predAnswer = answerPred[numpy.argmax(conf * coefficient)]

    if predAnswer != answer[i]:
        # print(i)
        wrong += 1

    print(predAnswer, file = fw)

    # print(answer[i], *answerPred)

print((len(answer) - wrong) / len(answer))
# print(predList)


