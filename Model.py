#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jasperyang
@license: (C) Copyright 2013-2017, Jasperyang Corporation Limited.
@contact: yiyangxianyi@gmail.com
@software: GibbsLDA
@file: Model.py
@time: 3/7/17 11:00 PM
@desc:    一些变量的解释
          1.一个训练数据的dataset:ptrndata
          2.word-topic的矩阵:nw
          3.某篇文档对应的各种topic的数量的矩阵:nd
          4.某个topic中总的单词数的矩阵:nwsum
          5.某个文档中单词数的矩阵:ndsum
          6.每篇文档中每个单词对应的topic的概率矩阵:z

nw[wordid][k] : 第i个单词被分配到第j个主题的次数，size VxK
nwsum[k] :第j个主题下单词的个数，size K
nd[m][k]:第i篇文档里被指定第j个主题词的次数，size M x K
ndsum[m] :第i篇文档里单词的总数, size M
z[m][per_doc_word_len]:是int二维数组，第i篇文档里第j个单词的主题index,size M x per_doc_word_len表示第m篇文档第n个word被指定的topic index
V:不重复单词个数

输出:theta(doc->topic )和phi(topic->word)、 tassign文件(topic assignment )
theta   MxK
phi     KxV
'''

import Constants
from Utils import *
from DataSet import *
import random
import numpy as np
import math


'''后来发现要声明一个二维数组很简单,b = [[0]*10]*10 ...'''

class Model(object):
    wordmapfile = ''  # file that contains word map [string -> integer id]
    trainlogfile = ''  # training log file
    tassign_suffix = ''  # suffix for topic assignment file
    theta_suffix = ''  # suffix for theta file
    phi_suffix = ''  # suffix for phi file
    others_suffix = ''  # suffix for file containing other parameters
    twords_suffix = ''  # suffix for file containing words-per-topics

    dir = ''  # model directory
    dfile = ''  # data file
    model_name = ''  # model name
    model_status = None  # model status:
    #               MODEL_STATUS_UNKNOWN: unknown status
    #               MODEL_STATUS_EST: estimating from scratch
    #               MODEL_STATUS_ESTC: continue to estimate the model from a previous one
    #               MODEL_STATUS_INF: do inference
    ptrndata = []  # list of training dataset object
    pnewdata = []  # list of new dataset object

    id2word = []  # word map [{int => string}]

    # --- model parameters and variables ---
    M = None  # dataset size (i.e., number of docs)
    V = None  # vocabulary size
    K = None  # number of topics
    KN = None # 多维属性个数
    alpha = None  # LDA hyperparameters
    beta = None
    niters = None  # number of Gibbs sampling iterations
    liter = None  # the iteration at which the model was saved
    savestep = None  # saving period
    twords = None  # print out top words per each topic
    withrawstrs = None

    p = []  # temp variable for sampling
    z = []  # topic assignments for words, size M x doc.size()
    nw = []  # cwt[i][j]: number of instances of word/term i assigned to topic j, size V x K
    nd = []  # na[i][j]: number of words in document i assigned to topic j, size M x K
    nwsum = []  # nwsum[j]: total number of words assigned to topic j, size K
    ndsum = []  # nasum[i]: total number of words in document i, size M
    theta = []  # theta: document-topic distributions, size M x K
    phi = []  # phi: topic-word distributions, size K x V

    # for inference only
    inf_liter = None
    newM = None
    newV = None
    newz = []
    newnw = []
    newnd = []
    newnwsum = []
    newndsum = []
    newtheta = []
    newphi = []
    perplexity= None

    # --------------------------------------

    # @tested
    def __init__(self):
        self.setdefault_value()

    # @tested
    def setdefault_value(self):
        self.wordmapfile = "wordmap.txt"
        self.trainlogfile = "trainlog.txt"
        self.tassign_suffix = ".tassign"
        self.theta_suffix = ".theta"
        self.phi_suffix = ".phi"
        self.others_suffix = ".others"
        self.twords_suffix = ".twords"

        self.dir = "./"
        self.dfile = "trndocs.dat"
        self.model_name = "model-final"
        self.model_status = Constants.MODEL_STATUS_UNKNOWN

        self.ptrndata = None
        self.pnewdata = None

        self.M = 0
        self.V = 0
        self.K = [1]
        self.KN = 0
        self.alpha = 50.0 / self.K[0]
        self.beta = []
        self.niters = 2000
        self.liter = 0
        self.savestep = 200
        self.twords = 0
        self.withrawstrs = 0

        self.p = None
        self.z = None
        self.nw = None
        self.nd = None
        self.nwsum = None
        self.ndsum = None
        self.theta = None
        self.phi = None

        self.newM = 0
        self.newV = 0
        self.newz = None
        self.newnw = None
        self.newnd = None
        self.newnwsum = None
        self.newndsum = None
        self.newtheta = None
        self.newphi = None

    # @tested
    def parse_args(self, argc, argv):
        u = Utils()
        return u.parse_args(argc, argv, self)

    # @tested
    def init(self, argc, argv):
        # call parse_args
        if self.parse_args(argc, argv):
            return 1
        if self.model_status == Constants.MODEL_STATUS_EST:
            # estimating the model from scratch (从头开始分析模型)
            if self.init_est():
                return 1
        elif self.model_status == Constants.MODEL_STATUS_ESTC:
            if self.init_estc():
                return 1
        elif self.model_status == Constants.MODEL_STATUS_INF:
            if self.init_inf():
                return 1
        return 0


    # # @tested
    def load_model(self, model_name):
        filename = self.dir + model_name + self.tassign_suffix
        fin = open(filename)
        if not fin:
            print("Cannot open file ", filename, " to load model")
            return 1

        self.z = []
        for n in range(self.M):
            self.z.append([])

        ptrndata = DataSet(self.M)
        ptrndata.V = self.V

        for i in range(self.M):
            line = fin.readline()
            if not line:
                print("Invalid word-topic assignment file, check the number of docs!\n")
                return 1
            strtok = Strtokenizer(line, ' \t\r\n')
            length = strtok.count_tokens()

            words = []
            topics = []
            for j in range(length):
                token = strtok.token(j)
                tok = Strtokenizer(token,':')
                if tok.count_tokens() != 2:
                    print("Invalid word-topic assignment line!\n")
                    return 1
                words.append(int(tok.token(0)))
                topics.append(int(tok.token(1)))

            pdoc = Document(words)
            ptrndata.add_doc(pdoc, i)

            # assign values for z
            for to in range(len(topics)):
                self.z[i].append(0)
            for j in range(len(topics)):
                self.z[i][j] = topics[j]

        self.ptrndata = ptrndata
        fin.close()
        return 0

    # @testing
    def init_estc(self):
        # estimating the model from a previously estimated one
        self.p = np.zeros(self.K,dtype=float).tolist() # double[K]

        # load model , i.e., read z and ptrndata
        if self.load_model(self.model_name):
            print("Fail to load word-topic assignment file of the model!\n")
            return 1
        self.nw = np.zeros((self.V,self.K),dtype=int).tolist()  # int[VxK]
        self.nd = np.zeros((self.M,self.K),dtype=int).tolist()  # int[MxK]    
        self.nwsum = np.zeros(self.K,dtype=int).tolist()  # int[K]
        self.ndsum = np.zeros(self.M,dtype=int).tolist()  # int[M]

        self.z = []  # int[M]
        for m in range(self.M):
            self.z.append([])

        for m in range(self.ptrndata.M):
            N = self.ptrndata.docs[m].length

            for n in range(N):  # 初始化z[M][N]
                self.z[m].append(0)

            # assign values for nw, nd, nwsum, and ndsum
            for n in range(N):
                w = self.ptrndata.docs[m].words[n]
                topic = self.z[m][n]
                # number of instance of word i assigned to topic j
                self.nw[w][topic] += 1
                # number of words in document i assigned to topic j
                self.nd[m][topic] += 1
                # total number of words assigned to topic j
                self.nwsum[topic] += 1
            # total number of words in document i
            self.ndsum[m] = N

        self.theta = np.zeros((self.M,self.K),dtype=float).tolist()  # double[MxK]
        self.phi = np.zeros((self.K,self.V),dtype=float).tolist()  # double[KxV]

        return 0

    # @tested
    def init_inf(self):
        # estimating the model from a previously estimated one
        self.p = np.zeros(self.K,dtype=float).tolist() # double[K]

        # load model , i.e., read z and ptrndata
        if self.load_model(self.model_name):
            print("Fail to load word-topic assignment file of the model!\n")
            return 1
            
        self.nw = np.zeros((self.V,self.K),dtype=int).tolist()  # int[VxK]
        self.nd = np.zeros((self.M,self.K),dtype=int).tolist()  # int[MxK]    
        self.nwsum = np.zeros(self.K,dtype=int).tolist()  # int[K]
        self.ndsum = np.zeros(self.M,dtype=int).tolist()  # int[M]

        self.z = []  # int[M]
        for m in range(self.M):
            self.z.append([])

        for m in range(self.ptrndata.M):
            N = self.ptrndata.docs[m].length

            for n in range(N):  # 初始化z[M][N]
                self.z[m].append(0)

            # assign values for nw, nd, nwsum, and ndsum
            for n in range(N):
                w = self.ptrndata.docs[m].words[n]
                topic = self.z[m][n]

                # number of instance of word i assigned to topic j
                self.nw[w][topic] += 1
                # number of words in document i assigned to topic j
                self.nd[m][topic] += 1
                # total number of words assigned to topic j
                self.nwsum[topic] += 1
            # total number of words in document i
            self.ndsum[m] = N

        self.pnewdata = DataSet()
        if self.withrawstrs:
            if self.pnewdata.read_newdata_withrawstrs(self.dir + self.dfile, self.dir + self.wordmapfile):
                print("Fail to read self.new data!\n")
                return 1
        else:
            if self.pnewdata.read_newdata(self.dir + self.dfile, self.dir + self.wordmapfile):
                print("Fail to read self.new data!\n")
                return 1
        self.newM = self.pnewdata.M
        self.newV = self.pnewdata.V

        self.newnw = np.zeros((self.newV,self.K),dtype=int).tolist()  # int*[newVxK]
        self.newnd = np.zeros((self.newM,self.K),dtype=int).tolist()   # int*[newMxK]
        self.newnwsum = np.zeros(self.K,dtype=int).tolist()  # int[K]
        self.newndsum = np.zeros(self.newM,dtype=int).tolist()  # int[self.newM]
        self.newz = np.zeros(self.newM,dtype=int).tolist()   # int*[self.newM]


        for m in range(self.pnewdata.M):
            N = self.pnewdata.docs[m].length
            newz_row = []  # int[N]

            # assign values for nw,nd,nwsum, and ndsum
            for n in range(N):
                w = self.pnewdata.docs[m].words[n]
                topic = random.randint(0, self.K-1)
                newz_row.append(topic)

                # number of instances of word i assigned to topic j
                self.newnw[w][topic] += 1
                # number of words in document i assigned to topic j
                self.newnd[m][topic] += 1
                # total number of words assigned to topic j
                self.newnwsum[topic] += 1
            # total number words in document i
            self.newndsum[m] = N
            self.newz[m] = newz_row

        self.newtheta = np.zeros((self.newM,self.K),dtype=float).tolist()  # double[MxK]
        self.newphi = np.zeros((self.K,self.newV),dtype=float).tolist()  # double[KxV]

        return 0

    # @tested
    # 从头开始分析,初始化模型
    def init_est(self):
        #每个属性的主题个数K=[J,K,L]
        #多个属性KN=len(K)
        #p[JxKxL]
        self.p = np.zeros(self.K,dtype=float) # double[JxKxL]

        # + read training data
        ptrndata = DataSet()
        if ptrndata.read_trndata(self.dir + self.dfile, self.dir + self.wordmapfile,self.KN):
            print("Fail to read training data!\n")
            return 1

        # + assign values for variables
        self.M = ptrndata.M
        #不重复单词个数 V[KN]
        self.V = ptrndata.V
        # K: from command line or default value
        # alpha, beta: from command line or default values
        # niters, savestep: from command line or default values

        # nw[wordid][k] : 第i个单词被分配到第j个主题的次数，size VxK
        # nwsum[k] :第j个主题下单词的个数，size K
        # nd[m][k] :第i篇文档里被指定第j个主题词的次数，size M x K
        # ndsum[m] :第i篇文档里单词的总数, size M
        # z[m][per_doc_word_len]:是int二维数组，第i篇文档里第j个单词的主题index,size M x per_doc_word_len表示第m篇文档第n个word被指定的topic index

        #nw[[V0xMK0],[V1xMK1]...]
        #nw[0][0][0]第0类属性第0个单词被分配给K0中第0个主题的次数        
        self.nw = []
        #nwsum[[MK0][MK1]...]
        #nwsum[[0,1,2..j][0,1,2..k]...[0,1,2..l]]
        #nwsum[0][0]=num第0类属性K0中第0个主题的词个数
        self.nwsum = []
        for i in range(self.KN):
            self.nw.append(np.zeros((self.V[i],self.K[i]),dtype=int).tolist())
            self.nwsum.append(np.zeros(self.K[i],dtype=int).tolist())

        #nd[[JxKxL],[JxKxL]...M个]
        #nd[0][0][0][0]第0篇文档被分配为0 0 0的主题词个数
        dims=[self.M]
        dims.extend(self.K)
        self.nd = np.zeros(dims,dtype=int) 

        #ndsum包含的是单词总数，多个属性的单词是同时出现，因此，计算只需要算某一个属性单词的总个数即可
        self.ndsum = np.zeros(self.M,dtype=int).tolist()  # int[M]

        #z[[[j,k,l],[j1,k1,l1]...word_len个] ...M 个]
        #z[0][0]=[j,k,l]
        self.z = []  # int[M]
        for m in range(self.M):
            self.z.append([])

        for m in range(self.M):
            N = ptrndata.docs[m].length   # 该篇文档的单词数
            
            for n in range(N):  # 初始化z[M][N]
                self.z[m].append(0)

            # initialize for z
            for n in range(N):  # 遍历每个单词
                topic=[]
                for kn in range(self.KN):
                    topic.append(random.randint(0, self.K[kn]-1))
                self.z[m][n] = topic                # 给单词随机赋予主题

                for kn in range(self.KN):
                    # number of instance of word i assigned to topic j 单词对应文档 V * K
                    self.nw[kn][ptrndata.docs[m].words[kn][n]][topic[kn]] += 1
                    # total number of words assigned to topic j
                    self.nwsum[kn][topic[kn]] += 1
                # number of words in document i assigned to topic j  M * K
                self.nd[m][tuple(topic)] += 1
            # total number of words in document i
            self.ndsum[m] = N


        #theta[[JxKxL],[JxKxL]...M个]
        self.theta = np.zeros(dims,dtype=float)   
        #phi[[MK0xV0],[MK1xV1]...]
        self.phi = []
        for i in range(self.KN):
            self.phi.append(np.zeros((self.K[i],self.V[i]),dtype=int).tolist())

        self.ptrndata = ptrndata
        return 0

    # @tested
    def save_model(self, model_name):
        if self.save_model_tassign(self.dir + model_name + self.tassign_suffix):
            return 1
        if (self.save_model_others(self.dir + model_name + self.others_suffix)):
            return 1
        if (self.save_model_theta(self.dir + model_name + self.theta_suffix)):
            return 1
        if (self.save_model_phi(self.dir + model_name + self.phi_suffix)):
            return 1
        if self.twords[0] > 0:
            if (self.save_model_twords(self.dir + model_name + self.twords_suffix)):
                return 1
        return 0

    # @tested
    def save_model_tassign(self, filename):
        fout = open(filename, 'w')
        if not fout:
            print("Cannot open file ", filename, " to save!\n")
            return 1

        # write docs with topic assignments for words
        for i in range(self.ptrndata.M):
            for j in range(self.ptrndata.docs[i].length):
                topic=self.z[i][j]
                tstr=str(self.ptrndata.docs[i].words[0][j])+ ":" + str(topic[0])
                for kn in range(1,self.KN):
                    tstr+='_'+str(self.ptrndata.docs[i].words[kn][j])+ ":" + str(topic[kn])
                tmp = tstr + " "
                fout.write(tmp)
            fout.write('\n')
        fout.close()
        return 0

    # @tested
    def save_model_theta(self, filename):
        fout = open(filename, 'w')
        if not fout:
            print("Cannot open file ", filename, " to save!\n")
            return 1

        # write docs with topic assignments for words
        prodK = np.prod(self.K)
        for numk in range(prodK):
            coords=self.compute_coorindex(numk)
            num_list_new = [str(x) for x in coords]
            title='_'.join(num_list_new)
            fout.write('theme:'+title+" ")
        fout.write('\n')
        for i in range(self.M):                   
            for numk in range(prodK):
                coords=self.compute_coorindex(numk)
                fout.write(str(self.theta[i][tuple(coords)]) + " ")
            fout.write('\n')

        fout.close()
        return 0

    # @tested
    def save_model_phi(self, filename):
        fout = open(filename, 'w')
        if not fout:
            print("Cannot open file ", filename, " to save!\n")
            return 1

        # write docs with topic assignments for words
        for kn in range(self.KN):
            fout.write('the probability of '+str(kn)+'-th topic-word' )
            fout.write('\n')
            for i in range(self.K[kn]):
                for j in range(self.V[kn]):
                    fout.write(str(self.phi[kn][i][j]) + " ")
                fout.write('\n')

        fout.close()
        return 0

    # @tested
    def save_model_twords(self,filename):
        fout = open(filename, 'w')
        if not fout:
            print("Cannot open file ", filename, " to save!\n")
            return 1
        for kn in range(self.KN):
            if(self.twords[kn] > self.V[kn]) :
                self.twords[kn] = self.V[kn]
            tmp = "the " + str(kn) + "-th properity:\n"
            fout.write(tmp)
            for k in range(self.K[kn]) :
                words_probs = []
                for w in range(self.V[kn]) :
                    word_prob = {w:self.phi[kn][k][w]}
                    words_probs.append(word_prob)

                # quick sort to word-topic probability
                u = Utils()
                u.quicksort(words_probs,0,len(words_probs)-1)

                tmp = "\tTopic " + str(k) + "th:\n"
                fout.write(tmp)
                for i in range(self.twords[kn]) :
                    found = False
                    for key in self.id2word[kn].keys() :
                        if list(words_probs[i].keys())[0] == int(key) :
                            found = True
                            break
                    if found :
                        tmp = "\t\t" + str(list(words_probs[i].keys())[0]) + " " + str(list(words_probs[i].values())[0]) + '\n'
                        fout.write(tmp)

        fout.close()
        return 0

    # @tested
    def save_model_others(self,filename):
        fout = open(filename,'w')
        if not fout :
            print("Cannot open file ",filename," to save!\n")
            return 1
        tmp = "alpha=" + str(self.alpha) + '\n'
        fout.write(tmp)
        tmp = "beta=" + '_'.join([str(x) for x in self.beta]) + '\n'
        fout.write(tmp)
        tmp = "ntopics=" + '_'.join([str(x) for x in self.K]) + '\n'
        fout.write(tmp)
        tmp = "ndocs=" + str(self.M) + '\n'
        fout.write(tmp)
        tmp = "nwords=" + '_'.join([str(x) for x in self.K]) + '\n'
        fout.write(tmp)
        tmp = "liter=" + str(self.liter) + '\n'
        fout.write(tmp)
        fout.close()
        return 0

    # @tested
    def save_inf_model(self, model_name):
        if self.save_inf_model_tassign(self.dir + model_name + self.tassign_suffix):
            return 1
        if (self.save_inf_model_others(self.dir + model_name + self.others_suffix)):
            return 1
        if (self.save_inf_model_newtheta(self.dir + model_name + self.theta_suffix)):
            return 1
        if (self.save_inf_model_newphi(self.dir + model_name + self.phi_suffix)):
            return 1
        if self.twords > 0:
            if (self.save_inf_model_twords(self.dir + model_name + self.twords_suffix)):
                return 1
        return 0

    # @tested
    def save_inf_model_tassign(self, filename):
        fout = open(filename, 'w')
        if not fout:
            print("Cannot open file ", filename, " to save!\n")
            return 1

        # write docs with topic assignments for words
        for i in range(self.pnewdata.M):
            for j in range(self.pnewdata.docs[i].length):
                tmp = str(self.pnewdata.docs[i].words[j]) + ":" + str(self.newz[i][j]) + " "
                fout.write(tmp)
            fout.write('\n')

        fout.close()
        return 0

    # @tested
    def save_inf_model_newtheta(self, filename):
        fout = open(filename, 'w')
        if not fout:
            print("Cannot open file ", filename, " to save!\n")
            return 1

        # write docs with topic assignments for words
        for i in range(self.newM):
            for j in range(self.K):
                fout.write(str(self.newtheta[i][j]) + " ")
            fout.write('\n')

        fout.close()
        return 0

    # @tested
    def save_inf_model_newphi(self, filename):
        fout = open(filename, 'w')
        if not fout:
            print("Cannot open file ", filename, " to save!\n")
            return 1

        # write docs with topic assignments for words
        for i in range(self.K):
            for j in range(self.newV):
                fout.write(str(self.newphi[i][j]) + " ")
            fout.write('\n')

        fout.close()
        return 0

    # @tested
    def save_inf_model_twords(self, filename):
        fout = open(filename, 'w')
        if not fout:
            print("Cannot open file ", filename, " to save!\n")
            return 1

        if (self.twords > self.newV):
            self.twords = self.newV

        for k in range(self.K):
            words_probs = []
            for w in range(self.newV):
                word_prob = {w: self.newphi[k][w]}
                words_probs.append(word_prob)

            # quick sort to word-topic probability
            u = Utils()
            u.quicksort(words_probs, 0, len(words_probs) - 1)

            tmp = "Topic " + str(k) + "th:\n"
            fout.write(tmp)
            for i in range(self.twords):
                found = False
                for key in self.pnewdata._id2id.keys() :
                    found2 = False
                    if list(words_probs[i].keys())[0] == key :
                        found2 = True
                        break
                if not found2 :
                    continue
                else :
                    for i in self.id2word :
                        if self.pnewdata._id2id(list(words_probs[i].keys())[0]) == key:
                            found = True
                            break
                    if found :
                        tmp = "\t" + list(words_probs[i].keys())[0] + " " + list(words_probs[i].values())[0] + '\n'
                        fout.write(tmp)

        fout.close()
        return 0

    # @tested
    def save_inf_model_others(self,filename):
        fout = open(filename,'w')
        if not fout :
            print("Cannot open file ",filename," to save!\n")
            return 1
        tmp = "alpha=" + str(self.alpha) + '\n'
        fout.write(tmp)
        tmp = "beta=" + str(self.beta) + '\n'
        fout.write(tmp)
        tmp = "ntopics=" + str(self.K) + '\n'
        fout.write(tmp)
        tmp = "ndocs=" + str(self.newM) + '\n'
        fout.write(tmp)
        tmp = "nwords=" + str(self.newV) + '\n'
        fout.write(tmp)
        tmp = "liter=" + str(self.inf_liter) + '\n'
        fout.write(tmp)
        fout.close()
        return 0

    # @tested
    def estimate(self):
        if self.twords[0] > 0 :
            da = DataSet()
            #为了读取id2word
            da.read_wordmapdict(self.dir + self.wordmapfile,self.id2word,self.KN)

        #print("Sampling ",self.niters," iterations!\n")

        last_iter = self.liter
        for self.liter in range(last_iter+1,self.niters+last_iter) :
            print("Iteration ",self.liter," ...\n")

            # for all z_i
            for m in range(self.M) :
                for n in range(self.ptrndata.docs[m].length) :
                    # (z_i) = z[m][n]
                    # sample from p(z_i|z_-i,w)
                    #topic=[j,k,l]
                    topic = self.sampling(m,n)
                    self.z[m][n] = topic

            if self.savestep > 0 :
                if self.liter % self.savestep == 0 :
                    # saving the model
                    print("Saving the model at iteration ",self.liter," ...\n")
                    self.compute_theta()
                    self.compute_phi()
                    u = Utils()
                    self.save_model(u.generate_model_name(self.model_name,self.liter))

        print("Gibbs sampling completed!\n")
        print("Saving the final model!\n")
        self.compute_theta()
        self.compute_phi()
        self.computer_perplexity()
        self.liter -= 1
        u = Utils()
        self.save_model(u.generate_model_name(self.model_name,-1))

    # @tested
    def sampling(self,m,n):
        # remove z_i from the count variables
        topic = self.z[m][n]
        ws=[]

        for kn in range(self.KN):
            w = self.ptrndata.docs[m].words[kn][n]
            ws.append(w)
            self.nw[kn][w][topic[kn]] -= 1
            self.nwsum[kn][topic[kn]] -= 1

        self.nd[m][tuple(topic)] -= 1

        self.ndsum[m] -= 1

        prodK = np.prod(self.K)
        Vbeta = (np.array(self.V) * np.array(self.beta)).tolist()
        Kalpha = prodK * self.alpha

        # do multinomial sampling via cumulative method
        #p[j,k,l]

        for numk in range(prodK):
            coords=self.compute_coorindex(numk)
            partone=parttwo=1.0
            for kn in range(self.KN):
                partone*=(self.nw[kn][ws[kn]][coords[kn]] + self.beta[kn]) / (self.nwsum[kn][coords[kn]] + Vbeta[kn])
            parttwo=(self.nd[m][tuple(coords)] + self.alpha) / (self.ndsum[m] + Kalpha)
            self.p[tuple(coords)]=partone*parttwo
                 
        # cumulate multinomial parameters
        for numk in range(1,prodK):
            pcoords=self.compute_coorindex(numk-1)
            coords=self.compute_coorindex(numk)
            self.p[tuple(coords)] += self.p[tuple(pcoords)]


        # scaled sample because of unnormalized p[]      
        lastcoord=self.compute_coorindex(prodK-1)
        u = np.random.rand() * self.p[tuple(lastcoord)]



        topic = []
        for numk in range(prodK):
            coords=self.compute_coorindex(numk)
            if self.p[tuple(coords)] > u :
                topic = coords
                break


        # add newly estimated z_i to count variables
        for kn in range(self.KN):
            self.nw[kn][ws[kn]][topic[kn]] += 1
            self.nwsum[kn][topic[kn]] += 1
        self.nd[m][tuple(topic)] += 1
        self.ndsum[m] += 1

        return topic
        
    # @tested
    def compute_coorindex(self,number):
        if self.KN==1:
            return [number]
        coors=[]
        for kn in range(self.KN-1):
            coors.append(number//np.prod(self.K[kn+1:]))
            number= number%np.prod(self.K[kn+1:])
            if kn==self.KN-2:
                coors.append(number)
        return coors

    # @tested
    def compute_theta(self):
        prodK = np.prod(self.K)
        for numk in range(prodK):
            coords=self.compute_coorindex(numk)
            for m in range(self.M) :
                self.theta[m][tuple(coords)] = (self.nd[m][tuple(coords)] + self.alpha) / (self.ndsum[m] + prodK * self.alpha)

         


    # @tested
    def compute_phi(self):
        for kn in range(self.KN):
            for k in range(self.K[kn]):
                for w in range(self.V[kn]):
                    self.phi[kn][k][w] = (self.nw[kn][w][k] + self.beta[kn]) / (self.nwsum[kn][k] + self.V[kn] * self.beta[kn])

    # @tested
    def computer_perplexity(self):
        prep = 0.0
        prob_doc_sum = 0.0
        docset_word_num = 0
        for m in range(self.M) :
            prob_doc = 0.0 # the probablity of the doc
            for n in range(self.ptrndata.docs[m].length) :
                prob_word = 0.0 # the probablity of the word
                prodK = np.prod(self.K)
                for numk in range(prodK):
                    coords=self.compute_coorindex(numk)
                    prob_word_s = self.theta[m][tuple(coords)]
                    for kn in range(self.KN):
                        w = self.ptrndata.docs[m].words[kn][n]
                        prob_word_s *= self.phi[kn][coords[kn]][w]
                    prob_word += prob_word_s
                prob_doc += math.log(prob_word) # p(d) = sum(log(p(w)))
            prob_doc_sum += prob_doc
            docset_word_num += self.ptrndata.docs[m].length
        prep = math.exp(-prob_doc_sum/docset_word_num) # perplexity = exp(-sum(p(d)/sum(Nd))
        self.perplexity=prep
        print ("the perplexity of this ldamodel is : %s"%prep)


    # @not tested
    def inference(self):
        if self.twords > 0 :
            self.pnewdata.read_wordmap2(self.dir + self.wordmapfile,self.id2word)

        print("Sampling ",self.niters,' iterations for inference!\n')

        for self.inf_liter in range(1,self.niters+1) :
            print("Iteration ",self.inf_liter," ...\n")

            # for all newz_i
            for m in range(self.newM) :
                for n in range(self.pnewdata.docs[m].length) :
                    # newz_i = newz[m][n]
                    # sample from p(z_i|z_-i,w)
                    topic = self.inf_sampling(m,n)
                    self.newz[m][n] = topic

        print("Gibbs sampling for inference completed!\n")
        print("Saving the inference outputs!\n")
        self.compute_newtheta()
        self.compute_newphi()
        self.inf_liter -= 1
        self.save_inf_model(self.dfile)

    # @not tested
    def inf_sampling(self,m,n):
        # remove z_i from the count variables
        topic = self.newz[m][n]
        w = self.pnewdata.docs[m].words[n]
        _w = self.pnewdata._docs[m].words[n]
        self.newnw[_w][topic] -= 1
        self.newnd[m][topic] -= 1
        self.newnwsum[topic] -= 1
        self.newndsum[m] -= 1

        Vbeta = self.V * self.beta
        Kalpha = self.K * self.alpha

        # do multinomial sampling via cumulative method
        for k in range(self.K) :
            self.p[k] = (self.nw[w][k] + self.newnw[_w][k] + self.beta) / (self.nwsum[k] + self.newnwsum[k] + Vbeta) * (self.newnd[m][k] + self.alpha) / (self.newndsum[m] + Kalpha)

        # cumulate multinomial parameters
        for k in range(self.K) :
            self.p[k] += self.p[k-1]

        # scaled sample because of unnormalized p[]
        u = np.random.rand() * self.p[self.K-1]

        for topic in range(self.K) :
            if self.p[topic] > u :
                break

        # add newly estimated z_i to count variables
        self.nw[_w][topic] += 1
        self.nd[m][topic] += 1
        self.nwsum[topic] += 1
        self.ndsum[m] += 1

        return topic

    # @not tested
    def compute_newtheta(self):
        for m in range(self.newM) :
            for k in range(self.K) :
                self.newtheta[m][k] = (self.newnd[m][k] + self.alpha) / (self.newndsum[m] + self.K * self.alpha)

    # @not tested
    def compute_newphi(self):
        for k in range(self.K):
            for w in range(self.newV):
                found = False
                for key in self.pnewdata._id2id.keys() :
                    if key == w :
                        found = True
                if found:
                    self.newphi[k][w] = (self.nw[self.pnewdata._id2id[w]][k] + self.newnw[w][k] + self.beta) / (self.nwsum[k] + self.newnwsum[k] + self.V * self.beta)