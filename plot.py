import numpy as np
import matplotlib.pyplot as plt
from operator import add
import traceback
import sys

def calAccuracy(file,out,path):
    try:
        with open(path+file+".txt", 'r') as obj:
            lines = obj.readlines()[1:]
        size = len(lines)
        # print(size)
        count = [0 for x in range(3)]

        for line in lines:
            line = list(map(float,line.strip().split()))
            countAdd = [1 if np.isclose(line[:out],line[index:index+out], atol=0.2).all() else 0 for index in range(out, len(line), out)]
            count = list(map(add, count, countAdd))

        accuracy = [round(float(cnt)/size*100,2) for cnt in count]
        return accuracy
    except Exception as e:
        traceback.print_exception(*sys.exc_info())
        return None

if __name__ == '__main__':
    subtask = 4
    filenames_set = [["Iris", "Wine", "Cancer", "Baloon", "Ions", "Zoo", "Lenses", "Balance"], ["CreditApproval","TicTac","Robot-Four", "Robot-TwentyFour"]]
    out_set = [[2, 3, 1, 1, 1, 7, 3, 3], [1, 1, 4, 4]]
    count = 1

    for filenames,out in zip(filenames_set,out_set):
        out = dict(zip(filenames, out))

        for set in ['train','test']:
            accuracydict = {}
            accuracy = []
            path = set+'/'
            # print(path)
            for file in filenames:
                acc = []
                acc = calAccuracy(file, out[file], path)
                if acc == None:
                    continue
                #print acc
                accuracydict[file] = acc
                accuracy.append(acc)

            accuracy = np.asarray(accuracy).transpose()
            print(accuracy)


            # data to plot
            n_groups = len(accuracydict)

            # create plot
            fig, ax = plt.subplots()
            index = np.arange(n_groups)
            bar_width = 0.45
            opacity = 0.8
            capsize = 3
            err = float(5)/100
            
            ytop = accuracy[1]-accuracy[0]
            ybot = accuracy[0]-accuracy[2]

            plt.bar(index +float(bar_width)/2, accuracy[0], bar_width,
                            alpha=opacity,
                            #error_kw=dict(elinewidth=1, ecolor='r'),
                            #yerr=(ybot,ytop),
                            color='c',
                            label=set+' accuracy')




            plt.xlabel('Datasets')
            plt.ylabel('Accuracy')
            plt.title(set.capitalize()+' Data Accuracy')
            plt.xticks(index + bar_width, filenames, rotation = 70)
            plt.legend()

            plt.tight_layout()
            plt.savefig(set.capitalize()+'Data'+str(count)+'.png')
            plt.show()
        count += 1
