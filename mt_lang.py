# i/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import random
import time
from scipy.stats import multivariate_normal
from scipy.stats import norm
import math
import os
from expdata import setexperimentdata
import sys


# An example of a class
class Network:
    def __init__(self, Topo, Train, Test, learn_rate):
        self.Top = Topo  # NN topology [input, hidden, output]
        self.TrainData = Train
        self.TestData = Test
        np.random.seed()
        self.lrate = learn_rate

        self.W1 = np.random.randn(self.Top[0], self.Top[1]) / np.sqrt(self.Top[0])
        self.B1 = np.random.randn(self.Top[1]) / np.sqrt(self.Top[1])  # bias first layer
        self.W2 = np.random.randn(self.Top[1], self.Top[2]) / np.sqrt(self.Top[1])
        self.B2 = np.random.randn(self.Top[2]) / np.sqrt(self.Top[1])  # bias second layer

        self.hidout = np.zeros((1, self.Top[1]))  # output of first hidden layer
        self.out = np.zeros((1, self.Top[2]))  # output last layer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sampleEr(self, actualout):
        error = np.subtract(self.out, actualout)
        sqerror = np.sum(np.square(error)) / self.Top[2]
        return sqerror

    def ForwardPass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)  # output of first hidden layer
        # print(self.B2.shape)
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = self.sigmoid(z2)  # output second hidden layer

    def BackwardPass(self, Input, desired):
        # print desired.shape
        out_delta = (desired - self.out) * (self.out * (1 - self.out))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))

        # self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
        # self.B2 += (-1 * self.lrate * out_delta)
        # self.W1 += (Input.T.dot(hid_delta) * self.lrate)
        # self.B1 += (-1 * self.lrate * hid_delta)

        layer = 1  # hidden to output
        for x in xrange(0, self.Top[layer]):
            for y in xrange(0, self.Top[layer + 1]):
                self.W2[x, y] += self.lrate * out_delta[y] * self.hidout[x]
        for y in xrange(0, self.Top[layer + 1]):
            self.B2[y] += -1 * self.lrate * out_delta[y]

        layer = 0  # Input to Hidden
        for x in xrange(0, self.Top[layer]):
            for y in xrange(0, self.Top[layer + 1]):
                self.W1[x, y] += self.lrate * hid_delta[y] * Input[x]
        for y in xrange(0, self.Top[layer + 1]):
            self.B1[y] += -1 * self.lrate * hid_delta[y]


    def decode_MTNencodingX(self, w, mtopo, subtasks):

        position = 0

        Top1 = mtopo[0]

        # print self.B2.shape

        for neu in xrange(0, Top1[1]):
            for row in xrange(0, Top1[0]):
                self.W1[row, neu] = w[position]
                #print neu, row, position, '    -----  a '
                position = position + 1

            self.B1[neu] = w[position]
            #print neu,   position, '    -----  b '
            position = position + 1

        for neu in xrange(0, Top1[2]):
            for row in xrange(0, Top1[1]):
                self.W2[row, neu] = w[position]
                #print neu, row, position, '    -----  c '
                position = position + 1


        if subtasks >=1:


            for step  in xrange(1, subtasks+1   ):

                TopPrev = mtopo[step-1]
                TopG = mtopo[step]
                Hid = TopPrev[1]
                Inp = TopPrev[0]


                layer = 0

                for neu in xrange(Hid , TopG[layer + 1]      ) :
                    for row in xrange(0, TopG[layer]   ):
                        #print neu, row, position, '    -----  A '

                        self.W1[row, neu] = w[position]
                        position = position + 1

                    self.B1[neu] = w[position]
                    #print neu,   position, '    -----  B '
                    position = position + 1

                diff = (TopG[layer + 1] - TopPrev[layer + 1]) # just the diff in number of hidden neurons between subtasks

                for neu in xrange(0, TopG[layer + 1]- diff):  # %
                    for row in xrange(Inp , TopG[layer]):
                        #print neu, row, position, '    -----  C '
                        self.W1[row, neu] = w[position]
                        position = position + 1

                layer = 1

                for neu in xrange(0, TopG[layer + 1]):  # %
                    for row in xrange(Hid , TopG[layer]):
                        #print neu, row, position, '    -----  D '
                        self.W2[row, neu] = w[position]
                        position = position + 1

                #print w
                #print self.W1
                #print self.B1
                #print self.W2



    def decode(self, w):
        w_layer1size = self.Top[0] * self.Top[1]
        w_layer2size = self.Top[1] * self.Top[2]

        w_layer1 = w[0:w_layer1size]
        self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))

        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))
        self.B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.Top[1]]
        self.B2 = w[w_layer1size + w_layer2size + self.Top[1]:w_layer1size + w_layer2size + self.Top[1] + self.Top[2]]

    def encode(self):
        w1 = self.W1.ravel()
        w2 = self.W2.ravel()
        w = np.concatenate([w1, w2, self.B1, self.B2])
        return w

    def langevin_gradient(self, data, w, depth, mtopo, subtask):  # BP with SGD (Stocastic BP)

        self.decode_MTNencodingX(w, mtopo, subtask)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros(size)

        for i in xrange(0, depth):
            for i in xrange(0, size):
                pat = i
                Input = data[pat, 0:self.Top[0]]
                Desired = data[pat, -self.Top[2]:]
                self.ForwardPass(Input)
                self.BackwardPass(Input, Desired)

        w_updated = self.encode()
        # print(w_updated.shape, self.Top)

        return w_updated

    def evaluate_proposal(self, data, w , mtopo, subtasks):  # BP with SGD (Stocastic BP)

        self.decode_MTNencodingX(w, mtopo, subtasks)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros((size, self.Top[2]))

        for i in xrange(0, size):  # to see what fx is produced by your current weight update
            Input = data[i, 0:self.Top[0]]
            self.ForwardPass(Input)
            fx[i] = self.out

        return fx


# --------------------------------------------------------------------------

def covert_time(secs):
    if secs >= 60:
        mins = str(secs / 60)
        secs = str(secs % 60)
    else:
        secs = str(secs)
        mins = str(00)

    if len(mins) == 1:
        mins = '0' + mins

    if len(secs) == 1:
        secs = '0' + secs

    return [mins, secs]


# -------------------------------------------------------------------


class MCMC:
    def __init__(self, mtaskNet, samples, traindata, testdata, num_subtasks):
        self.samples = samples  # NN topology [input, hidden, output]
        # self.topology = topology  # max epocs
        self.traindata = traindata  #
        self.testdata = testdata
        self.subtasks = num_subtasks

        self.mtaskNet = mtaskNet
        # ----------------

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def net_size(self, netw):
        return ((netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2])

    def likelihood_func(self, neuralnet, data, w, tausq, subtask):
        y = data[:, -neuralnet.Top[2]:]
        fx = neuralnet.evaluate_proposal(data, w, self.mtaskNet, subtask)
        rmse = self.rmse(fx, y)
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
        return [np.sum(loss), fx, rmse]

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq, topo):
        h = topo[1]  # number hidden neurons
        d = topo[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def taskdata(self, data, taskfeatures, output):
        # group taskdata from main data source.
        # note that the grouping is done in accending order fin terms of features.
        # the way the data is grouped as tasks can change for different applications.
        # there is some motivation to keep the features with highest contribution as first  feature space for module 1.
        datacols = data.shape[1]
        featuregroup = data[:, 0:taskfeatures]
        return np.concatenate((featuregroup[:, range(0, taskfeatures)], data[:, range(datacols - output, datacols)]),
                              axis=1)

    def sampler(self, w_limit, tau_limit, file):

        start = time.time()

        # ------------------- initialize MCMC
        testsize = self.testdata.shape[0]
        trainsize = self.traindata.shape[0]
        samples = self.samples

        self.sgd_depth = 1
        learn_rate = 0.5

        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)

        Netlist = [None] * 10

        netsize = np.zeros(self.subtasks, dtype=np.int)

        for n in xrange(0, self.subtasks):
            module = self.mtaskNet[n]
            trdata = self.taskdata(self.traindata, module[0], module[2])  # make the partitions for task data
            testdata = self.taskdata(self.testdata, module[0], module[2])
            Netlist[n] = Network(self.mtaskNet[n], trdata, testdata, learn_rate)
            # print("Size: "+str(n)+" " + str(trdata)+ " " + str(module[0]))
            # print trdata

        for n in xrange(0, self.subtasks):
            netw = Netlist[n].Top
            netsize[n] = self.net_size(netw)  # num of weights and bias
            # print(netsize[n])

        y_test = self.testdata[:, netw[0]:]
        y_train = self.traindata[:, netw[0]:]

        pos_w = np.ones((samples, self.subtasks, netsize[self.subtasks - 1]))  # posterior of all weights and bias over all samples
        pos_tau = np.ones((samples, self.subtasks))

        fxtrain_samples = np.ones((samples, self.subtasks, trainsize, netw[2]))  # fx of train data over all samples
        fxtest_samples = np.ones((samples, self.subtasks, testsize, netw[2]))  # fx of test data over all samples
        rmse_train = np.zeros((samples, self.subtasks))
        rmse_test = np.zeros((samples, self.subtasks))

        w = np.random.randn(self.subtasks, netsize[self.subtasks-1])
        w_proposal = np.random.randn(self.subtasks, netsize[self.subtasks-1])
        w_gd = np.random.randn(self.subtasks, netsize[self.subtasks-1])
        w_prop_gd = np.random.randn(self.subtasks, netsize[self.subtasks-1])

        # step_w = 0.05;  # defines how much variation you need in changes to w
        # step_eta = 0.2; # exp 0

        step_w = w_limit  # defines how much variation you need in changes to w
        step_eta = tau_limit  # exp 1
        # --------------------- Declare FNN and initialize

        pred_train = Netlist[0].evaluate_proposal(self.traindata, w[0, :netsize[0]], self.mtaskNet, 0)
        pred_test = Netlist[0].evaluate_proposal(self.testdata, w[0, :netsize[0]], self.mtaskNet, 0)
        rmsetrain = np.zeros(self.subtasks)
        rmsetest  = np.zeros(self.subtasks)

        eta = np.log(np.var(pred_train - y_train))
        tau_pro = np.exp(eta)

        pred_train = np.zeros((self.subtasks, trainsize, Netlist[0].Top[2]))
        pred_test = np.zeros((self.subtasks, testsize, Netlist[0].Top[2]))

        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0

        sigma_diagmat = np.zeros((self.subtasks, netsize[self.subtasks-1], netsize[self.subtasks-1]))  # for Equation 9 in Ref [Chandra_ICONIP2017]
        for s in xrange(self.subtasks):
            np.fill_diagonal(sigma_diagmat[s], step_w)

        delta_likelihood = 0.5  # an arbitrary position



        likelihood = np.zeros(self.subtasks)
        likelihood_proposal = np.zeros(self.subtasks)
        likelihood_ignore = np.zeros(self.subtasks)

        prior_current = np.zeros(self.subtasks)

        prior_pro = np.zeros(self.subtasks)

        for n in xrange(0, self.subtasks):
            prior_current[n] = self.prior_likelihood(sigma_squared, nu_1, nu_2, w[n, :netsize[0]], tau_pro, Netlist[n].Top)  # takes care of the gradients

        mh_prob = np.zeros(self.subtasks)



        for s in xrange(0, self.subtasks-1):
            [likelihood[s], fxtrain_samples[0, s, :], rmse_train[0, s]] = self.likelihood_func(Netlist[s], self.traindata, w[s, :netsize[s]], tau_pro, s)
            [likelihood_ignore, fxtest_samples[0, s, :], rmse_test[0, s]] = self.likelihood_func(Netlist[s], self.testdata, w[s, :netsize[s]], tau_pro, s)
            w[s + 1, :netsize[s]] = w[s, :netsize[s]]

        s = self.subtasks - 1
        [likelihood[s], fxtrain_samples[0, s, :], rmse_train[0, s]] = self.likelihood_func(Netlist[s], self.traindata,
                                                                                           w[s, :netsize[s]], tau_pro,
                                                                                           s)
        # print likelihood

        naccept = 0
        # print 'begin sampling using mcmc random walk'
        # plt.plot(x_train, y_train)
        # plt.plot(x_train, pred_train)
        # plt.title("Plot of Data vs Initial Fx")
        # plt.savefig('mcmcresults/begin.png')
        # plt.clf()

        # plt.plot(x_train, y_train)

        diff_prop = np.zeros(self.subtasks)
        diff = np.zeros(self.subtasks)
        prior_prop = np.zeros(self.subtasks)

        for i in range(1, samples - 1):
            # print i
            for s in xrange(self.subtasks):
                # print("B2:")
                # print(Netlist[s].B2.shape)
                # print(netsize[s], Netlist[s].Top)
                w_gd[s, :netsize[s]] = Netlist[s].langevin_gradient(self.traindata, w[s, :netsize[s]].copy(), self.sgd_depth, self.mtaskNet, s)  # Eq 8
                w_proposal[s, :netsize[s]] = w_gd[s, :netsize[s]] + np.random.normal(0, step_w, netsize[s])  # Eq 7
                w_prop_gd[s, :netsize[s]] = Netlist[s].langevin_gradient(self.traindata, w_proposal[s].copy(), self.sgd_depth, self.mtaskNet, s)

            # print(multivariate_normal.pdf(w, w_prop_gd, sigma_diagmat),multivariate_normal.pdf(w_proposal, w_gd, sigma_diagmat))


            for s in xrange(self.subtasks):
                diff_prop[s] = np.log(multivariate_normal.pdf(w[s, :netsize[s]], w_prop_gd[s, :netsize[s]], sigma_diagmat[s, :netsize[s], :netsize[s]]) - np.log(
                multivariate_normal.pdf(w_proposal[s, :netsize[s]], w_gd[s, :netsize[s]], sigma_diagmat[s, :netsize[s], :netsize[s]])))

            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)

            for s in xrange(self.subtasks-1):
                [likelihood_proposal[s], pred_train[s, :], rmsetrain[s]] = self.likelihood_func(Netlist[s], self.traindata, w_proposal[s, :netsize[s]], tau_pro, s)

                [_, pred_test[s, :], rmsetest[s]] = self.likelihood_func(Netlist[s], self.testdata, w_proposal[s, :netsize[s]], tau_pro, s)
                w_proposal[s+1, :netsize[s]] = w_proposal[s, :netsize[s]]

            s = self.subtasks - 1
            [likelihood_proposal[s], pred_train[s, :], rmsetrain[s]] = self.likelihood_func(Netlist[s], self.traindata,
                                                                                            w_proposal[s, :netsize[s]],
                                                                                            tau_pro, s)
            [_, pred_test[s, :], rmsetest[s]] = self.likelihood_func(Netlist[s], self.testdata,
                                                                                        w_proposal[s, :netsize[s]],
                                                                                        tau_pro, s)


            # likelihood_ignore  refers to parameter that will not be used in the alg.

            for s in xrange(self.subtasks):
                prior_prop[s] = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal[s, :netsize[s]],
                                               tau_pro, Netlist[s].Top)  # takes care of the gradients

            diff_prior = prior_prop - prior_current
            diff_likelihood = likelihood_proposal - likelihood

            for s in xrange(self.subtasks):
                diff[s] = min(700, diff_prior[s] + diff_likelihood[s] + diff_prop[s])
                mh_prob[s] = min(1, math.exp(diff[s]))
                # print()
                # print(diff, i)

                # print(mh_prob)

                u = random.uniform(0, 1)

                if u < mh_prob[s]:
                    # Update position
                    # print    i, ' is accepted sample'
                    naccept += 1
                    likelihood[s] = likelihood_proposal[s]
                    prior_current[s] = prior_prop[s]
                    w[s, :netsize[s]] = w_proposal[s, :netsize[s]]
                    eta = eta_pro

                    elapsed_time = ":".join(covert_time(int(time.time() - start)))
                    # sys.stdout.write('\r' + file + ' : ' + str(round(float(i) / (samples - 1) * 100, 2)) + '% complete....'+" time elapsed: " + elapsed_time)
                    # print  likelihood, prior_current, diff_prop, rmsetrain, rmsetest, w, 'accepted'
                    # print w_proposal, 'w_proposal'
                    # print w_gd, 'w_gd'

                    # print w_prop_gd, 'w_prop_gd'

                    pos_w[i + 1, s, :netsize[s]] = w_proposal[s, :netsize[s]]
                    pos_tau[i + 1] = tau_pro
                    fxtrain_samples[i + 1, s, :] = pred_train[s, :]
                    fxtest_samples[i + 1, s, :] = pred_test[s, :]
                    rmse_train[i + 1, s] = rmsetrain[s]
                    rmse_test[i + 1, s] = rmsetest[s]

                    print i, 'accepted'
                    # plt.plot(x_train, pred_train)


                else:
                    pos_w[i + 1, s, :netsize[s]] = pos_w[i, s, :netsize[s]]
                    pos_tau[i + 1, s] = pos_tau[i, s]
                    fxtrain_samples[i + 1, s, :] = fxtrain_samples[i, s, :]
                    fxtest_samples[i + 1, s, :] = fxtest_samples[i, s, :]
                    rmse_train[i + 1, s] = rmse_train[i, s]
                    rmse_test[i + 1, s] = rmse_test[i, s]

                    print i, 'rejected and retained'
        sys.stdout.write('\r' + file + ' : 100% ..... Total Time: ' + ":".join(covert_time(int(time.time() - start))))
        # print naccept, ' num accepted'
        # print naccept / (samples * 1.0), '% was accepted'
        accept_ratio = naccept / (samples * self.subtasks * 1.0) * 100

        # plt.title("Plot of Accepted Proposals")
        # plt.savefig('mcmcresults/proposals.png')
        # plt.savefig('mcmcresults/proposals.svg', format='svg', dpi=600)
        # plt.clf()

        return (pos_w, pos_tau, fxtrain_samples, fxtest_samples, x_train, x_test, rmse_train, rmse_test, accept_ratio)


def main():
    moduledecompratio = [0.25, 0.50, 0.75, 1]
    filenames = ["Iris", "Wine", "Cancer", "Heart", "CreditApproval", "Baloon", "TicTac", "Ions", "Zoo",
                 "Lenses", "Balance", "Robot-Four", "Robot-TwentyFour"]
    problemlist = np.array(range(13))
    input = np.array([4, 13, 9, 13, 15, 4, 9, 34, 16, 4, 4, 4, 24])
    hidden = np.array([6, 6, 6, 16, 20, 5, 30, 8, 6, 5, 8, 14, 14])
    output = np.array([2, 3, 1, 1, 1, 1, 1, 1, 7, 3, 3, 4, 4])

    samplelist = [5000, 80, 10000, 20000, 15000, 5000, 20000, 5000, 3000, 5000, 2000, 20000, 10000]
    x = 3
    subtasks = 4

    # filetrain = open('Results/train.txt', 'r')
    # filetest = open('Results/test.txt', 'r')
    # filestdtr = open('Results/std_tr.txt', 'r')
    # filestdts = open('Results/std_ts.txt', 'r')

    # train_accs = np.loadtxt(filetrain)
    # test_accs = np.loadtxt(filetest)
    #
    # train_stds = np.loadtxt(filestdtr)
    # test_stds = np.loadtxt(filestdts)

    # filetrain.close()
    # filetest.close()
    # filestdtr.close()
    # filestdts.close()

    numproblems = problemlist.size

    train_accs = np.zeros((numproblems, subtasks))
    test_accs = np.zeros((numproblems, subtasks))

    train_stds = np.zeros((numproblems, subtasks))
    test_stds = np.zeros((numproblems, subtasks))


    if x == 3:
        w_limit = 0.02
        tau_limit = 0.2
    # if x == 4:
    # w_limit =  0.02
    # tau_limit = 0.1

    for problem in [0]:

        [traindata, testdata, baseNet] = setexperimentdata(problem)
        # print(baseNet)
        IN = input[problem]
        moduledecomp = [int(r * IN) for r in moduledecompratio]

        mtaskNet = np.array([baseNet, baseNet, baseNet, baseNet])
        # print(mtaskNet)

        for i in xrange(1, subtasks):
            # print(mtaskNet)
            mtaskNet[i - 1][0] = moduledecomp[i - 1]
            mtaskNet[i][1] += (i * 2)  # in this example, we have fixed numner  output neurons. input for each task is termined by feature group size.
            # we adapt the number of hidden neurons for each task.

        print(problem, mtaskNet)


        random.seed(time.time())

        numSamples = samplelist[problem]  # need to decide yourself

        mcmc = MCMC(mtaskNet, numSamples, traindata, testdata, subtasks)  # declare class

        [pos_w, pos_tau, fx_train, fx_test, x_train, x_test, rmse_train, rmse_test, accept_ratio] = mcmc.sampler(
            w_limit, tau_limit, filenames[problem])

        print 'sucessfully sampled: ' + str(accept_ratio) + ' samples accepted'

        burnin = 0.1 * numSamples  # use post burn in samples

        pos_w = pos_w[int(burnin):, ]
        pos_tau = pos_tau[int(burnin):, ]

        print("fx shape:" + str(fx_test.shape))
        print("fx_train shape:" + str(fx_train.shape))


        print(fx_test[int(burnin):], fx_train[int(burnin):])


        fx_tr_1 = fx_train[int(burnin):, 0, :]
        fx_tr_2 = fx_train[int(burnin):, 1, :]
        fx_tr_3 = fx_train[int(burnin):, 2, :]
        fx_tr_4 = fx_train[int(burnin):, 3, :]

        fx_train = np.asarray([fx_tr_1, fx_tr_2, fx_tr_3, fx_tr_4])


        fx_ts_1 = fx_test[int(burnin):, 0, :]
        fx_ts_2 = fx_test[int(burnin):, 1, :]
        fx_ts_3 = fx_test[int(burnin):, 2, :]
        fx_ts_4 = fx_test[int(burnin):, 3, :]

        # print(fx_ts_1.shape)

        fx_test = np.asarray([fx_ts_1, fx_ts_2, fx_ts_3, fx_ts_4])

        pos_w_mean = pos_w.mean(axis=0)
        # np.savetxt(outpos_w, pos_w_mean, fmt='%1.5f')

        rmse_tr = np.mean(rmse_train[int(burnin):])
        rmsetr_std = np.std(rmse_train[int(burnin):])
        rmse_tes = np.mean(rmse_test[int(burnin):])
        rmsetest_std = np.std(rmse_test[int(burnin):])
        # print rmse_tr, rmsetr_std, rmse_tes, rmsetest_std
        # np.savetxt(outres, (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, accept_ratio), fmt='%1.5f')

        ytestdata = testdata[:, input[problem]:]
        ytraindata = traindata[:, input[problem]:]

        train_acc = np.zeros((subtasks, fx_tr_1.shape[0]))
        test_acc = np.zeros((subtasks, fx_ts_1.shape[0]))


        # print(fx_test,fx_train)


        for fx_sub_in in range(fx_train.shape[0]):
            fx_sub = fx_train[fx_sub_in]
            acc = np.zeros(fx_sub.shape[0])
            for fx_in in range(fx_sub.shape[0]):
                count = 0
                for index in range(fx_sub[fx_in].shape[0]):
                    if np.isclose(fx_sub[fx_in][index], ytraindata[index], atol=0.2).all():
                        count += 1
                acc[fx_in] = (float(count) / fx_sub[fx_in].shape[0] * 100)
            train_acc[fx_sub_in] = acc

        for fx_sub_in in range(fx_test.shape[0]):
            fx_sub = fx_test[fx_sub_in]
            acc = np.zeros(fx_sub.shape[0])
            for fx_in in range(fx_sub.shape[0]):
                count = 0
                for index in range(fx_sub[fx_in].shape[0]):
                    if np.isclose(fx_sub[fx_in][index], ytestdata[index], atol=0.5).all():
                        count += 1
                acc[fx_in] = (float(count) / fx_sub[fx_in].shape[0] * 100)
            test_acc[fx_sub_in] = acc

        train_accs[problem] = train_acc.mean(axis=1)
        test_accs[problem] = test_acc.mean(axis=1)

        train_stds[problem] = np.std(train_acc, axis=1)
        test_stds[problem] = np.std(test_acc, axis=1)

        print(train_stds[problem], test_stds[problem], train_accs[problem], test_accs[problem])

        # train_accs[problem] = train_acc_mu
        # test_accs[problem] = test_acc_mu
        # train_stds[problem] = train_std
        # test_stds[problem] = test_std


        # # Write RMSE to
        # with open("Results/" +
        #           filenames[problem] + "_rmse" + ".txt", 'w') as fil:
        #     rmse = [rmse_tr, rmsetr_std, rmse_tes, rmsetest_std]
        #     rmse = "\t".join(list(map(str, rmse))) + "\n"
        #     fil.write(rmse)

    # n_groups = len(filenames)
    # fig, ax = plt.subplots()
    # index = np.arange(n_groups)
    # bar_width = 0.2
    # opacity = 0.8
    # capsize = 3
    #
    # filetrain = open('Results/train.txt', 'w+')
    # filetest = open('Results/test.txt', 'w+')
    # filestdtr = open('Results/std_tr.txt', 'w+')
    # filestdts = open('Results/std_ts.txt', 'w+')
    #
    # np.savetxt(filetrain, train_accs, fmt='%2.2f')
    # np.savetxt(filestdtr, train_stds, fmt='%2.2f')
    # np.savetxt(filetest, test_accs, fmt='%2.2f')
    # np.savetxt(filestdts, test_stds, fmt='%2.2f')
    #
    # filetrain.close()
    # filetest.close()
    # filestdtr.close()
    # filestdts.close()
    #
    # print(train_accs)
    # plt.bar(index + float(bar_width) / 2, train_accs, bar_width,
    #         alpha=opacity,
    #         error_kw=dict(elinewidth=1, ecolor='r'),
    #         yerr=train_stds,
    #         color='c',
    #         label='train')
    #
    # plt.bar(index + float(bar_width) / 2 + bar_width, test_accs, bar_width,
    #         alpha=opacity,
    #         error_kw=dict(elinewidth=1, ecolor='g'),
    #         yerr=test_stds,
    #         color='b',
    #         label='test')
    # plt.xlabel('Datasets')
    # plt.ylabel('Accuracy')
    # plt.xticks(index + bar_width, filenames, rotation=70)
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.savefig('barplt.png')
    # plt.show()


if __name__ == "__main__": main()
