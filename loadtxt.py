import numpy as np

filetrain = open('train.txt', 'r+')
filetest = open('train.txt', 'r+')
filestdtr = open('std_tr.txt', 'r+')
filestdts = open('std_ts.txt', 'r+')

train = np.loadtxt(filetrain)
print(train)

filetrain.close()
filetest.close()
filestdtr.close()
filestdts.close()
