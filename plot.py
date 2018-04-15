import numpy as np
import matplotlib.pyplot as plt
from operator import add


def roundClass(val):
    if val <= 0.5:
        val = 0.0
    elif val > 0.5:
        val = 1.0
    return val

def calAccuracy(file,out,path):
    try:
        with open(path+file+".txt", 'r') as obj:
            lines = obj.readlines()[1:]
        size = len(lines)
        # print(size)
        count = [0 for x in range(3)]

        for line in lines:
            line = line.strip().split()
            line = [roundClass(val) for val in list(map(float, line))]
            # print(path+file,len(line),line)
            countAdd = [1 if line[:out] == line[index:index+out] else 0 for index in range(out,len(line),out)]
            # print(file, out, line, countAdd)
            count = list(map(add, count, countAdd))

        accuracy = [round(float(cnt)/size*100,2) for cnt in count]
        return accuracy
    except Exception as e:
        # print Exception("File no file named "+ str(file))
        return None

if __name__ == '__main__':
    # for subtask in range(4,5):
    subtask = 4
    filenames_set = [["Iris", "Wine", "Cancer", "Baloon", "Ions", "Zoo", "Lenses", "Balance", "CreditApproval"]]
    # filenames = ["Iris", "Wine", "Cancer", "Heart", "CreditApproval", "Baloon", "TicTac", "Ions", "Lenses", "Balance"]
    out_set = [[2, 3, 1, 1, 1, 7, 3, 3, 1]]
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
            #print(accuracy)


            # data to plot
            n_groups = len(accuracydict)

            # create plot
            fig, ax = plt.subplots()
            index = np.arange(n_groups)
            bar_width = 0.2
            opacity = 0.8
            capsize = 3
            err = float(5)/100

            # Sub-Task1
            plt.errorbar(index + bar_width, accuracy[0], err*accuracy[0] ,
                            alpha=opacity,
                            color='c',
                            capsize=capsize,
                            capthick=None,
                            fmt=None)

            plt.bar(index , accuracy[0], bar_width,
                            alpha=opacity,
                            color='b',
                            label='fx_mu')


            # Sub-Task 2
            plt.errorbar(index + bar_width, accuracy[1], err*accuracy[1],
                            alpha=opacity,
                            color='b',
                            capsize=capsize,
                            fmt= None)

            plt.bar(index + bar_width, accuracy[1], bar_width,
                            alpha=opacity,
                            color='c',
                            label='fx_high')

            # Sub-Task 3
            plt.errorbar(index + 2*bar_width, accuracy[2], err*accuracy[2],
                            alpha=opacity,
                            color='y',
                            capsize=capsize,
                            fmt=None)

            plt.bar(index + 2*bar_width, accuracy[2], bar_width,
                            alpha=opacity,
                            color='m',
                            label='fx_low')

            # # Sub-Task 4
            # plt.errorbar(index + 3*bar_width, accuracy[3], err*accuracy[3],
            #                 alpha=opacity,
            #                 color='g',
            #                 capsize=capsize,
            #                 fmt= None)
            #
            # plt.bar(index + 3*bar_width, accuracy[3], bar_width,
            #                  alpha=opacity,
            #                  color='y',
            #                  label='ST4')



            plt.xlabel('Datasets')
            plt.ylabel('Accuracy')
            plt.title(set.capitalize()+' Data Accuracy')
            plt.xticks(index + bar_width, filenames, rotation = 70)
            plt.legend()

            plt.tight_layout()
            # plt.savefig(set.capitalize()+'Data'+str(count)+'.png')
            plt.show()
        count += 1