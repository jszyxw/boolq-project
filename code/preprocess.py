from sklearn import preprocessing
import numpy as np

f = open("answer-deberta-large.txt", "r")
fw = open("answer-deberta-normalized.txt", "w")
# f = open("answer-albert.txt", "r")
# fw = open("answer-albert-normalized.txt", "w")

print(f.readline(), file=fw, end='')
predList = [list(map(float, line.split())) for line in f.readlines()]

a = np.array(predList)
# print("Data = ", a.flatten(), file=fw)

# normalize the data attributes
# normalized = preprocessing.scale(a)
normalized = preprocessing.StandardScaler().fit(np.reshape(a.flatten(), (-1, 1)))
for i in a:
    print(*normalized.transform(np.reshape(i, (-1, 1))).flatten().tolist(), file=fw)
