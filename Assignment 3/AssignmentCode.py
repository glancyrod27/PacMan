import pandas as pd
import numpy as np
import time
import math

#Data with features and target values
#Tutorial for Pandas is here - https://pandas.pydata.org/pandas-docs/stable/tutorials.html
#Helper functions are provided so you shouldn't need to learn Pandas
dataset = pd.read_csv("data.csv")

#========================================== Data Helper Functions ==========================================

#Normalize values between 0 and 1
#dataset: Pandas dataframe
#categories: list of columns to normalize, e.g. ["column A", "column C"]
#Return: full dataset with normalized values
def normalizeData(dataset, categories):
    normData = dataset.copy()
    col = dataset[categories]
    col_norm = (col - col.min()) / (col.max() - col.min())
    normData[categories] = col_norm
    return normData

#Encode categorical values as mutliple columns (One Hot Encoding)
#dataset: Pandas dataframe
#categories: list of columns to encode, e.g. ["column A", "column C"]
#Return: full dataset with categorical columns replaced with 1 column per category
def encodeData(dataset, categories):
    return pd.get_dummies(dataset, columns=categories)

#Split data between training and testing data
#dataset: Pandas dataframe
#ratio: number [0, 1] that determines percentage of data used for training
#Return: (Training Data, Testing Data)
def trainingTestData(dataset, ratio):
    tr = int(len(dataset)*ratio)
    return dataset[:tr], dataset[tr:]

#Convenience function to extract Numpy data from dataset
#dataset: Pandas dataframe
#Return: features numpy array and corresponding labels as numpy array
def getNumpy(dataset):
    features = dataset.drop(["can_id", "can_nam","winner"], axis=1).values
    labels = dataset["winner"].astype(int).values
    return features, labels

#Convenience function to extract data from dataset (if you prefer not to use Numpy)
#dataset: Pandas dataframe
#Return: features list and corresponding labels as a list
def getPythonList(dataset):
    f, l = getNumpy(dataset)
    return f.tolist(), l.tolist()

#Calculates accuracy of your models output.
#solutions: model predictions as a list or numpy array
#real: model labels as a list or numpy array
#Return: number between 0 and 1 representing your model's accuracy
def evaluate(solutions, real):
    predictions = np.array(solutions)
    labels = np.array(real)
    return (predictions == labels).sum() / float(labels.size)

#===========================================================================================================

class KNN:
    def __init__(self):
        return
        #KNN state here
        #Feel free to add methods

    def preprocess(self,dataset):
        category=["net_ope_exp","net_con","tot_loa"]
        data1=normalizeData(dataset,category)
        category2=["can_off","can_inc_cha_ope_sea"]
        data2=encodeData(data1,category2)
        trainset,testset=trainingTestData(data2,0.67)
        trainfeature,trainlabel=getNumpy(trainset)
        testfeature,testlabel=getNumpy(testset)
        return trainfeature,trainlabel,testfeature,testlabel

    def train(self, features, labels):
        self.trainfeature = features
        self.trainlabels = labels
        return

    def predict(self, features):


        def euclidean(obj1, obj2, length):
            distance = 0
            for i in range(length):
                distance = distance + pow((obj1[i] - obj2[i]), 2)
            E_distance = pow(distance, 0.5)
            return E_distance

        sol = []
        k = 10
        for i in range(len(features)):
            dist = []
            votes = {}

            #Finding all the closest vectors
            for j in range(len(self.trainfeature)):
                dist1 = euclidean(self.trainfeature[j], features[i], len(features[i]))
                dist.append((self.trainfeature[j], self.trainlabels[j], dist1))
            dist.sort(key=lambda x: x[2])
            closed = []
            for j in range(k):
                closed.append(dist[j][1])

            #Storing labels of closest one so that we can take vote lateron
            for j in range(k):
                if closed[j] in votes:
                    votes[closed[j]] = votes[closed[j]] + 1
                else:
                    votes[closed[j]] = 1

            #sorting to find best one to return
            best = sorted(votes.items(), key=lambda x: x[1], reverse=True)

            answer = best[0][0]
            sol.append(answer)
        return sol


class Perceptron:
    def __init__(self):
        #Perceptron state here
        #Feel free to add methods
        pass

    def preprocess(self,dataset):
        category=["net_ope_exp","net_con","tot_loa"]
        data1=normalizeData(dataset,category)
        category2=["can_off","can_inc_cha_ope_sea"]
        data2=encodeData(data1,category2)
        trainset,testset=trainingTestData(data2,0.67)
        trainfeature,trainlabel=getNumpy(trainset)
        testfeature,testlabel=getNumpy(testset)
        return trainfeature,trainlabel,testfeature,testlabel

    def train(self, features, labels):

        def binary_to_bipolar(labels):
            for i in range(len(labels)):
                if (labels[i] == 0):
                    labels[i] = -1
            return labels

        self.wt=np.random.uniform(low=-0.1, high=0.1, size=9)
        self.bias=np.random.uniform(-0.1,0.1)

        alpha=0.01
        theta=0

        labels1=binary_to_bipolar(labels)

        #training for 55
        t = time.time()+55
        while(time.time() < t ):
            for i in range(len(features)):

                #calculating output using step function
                yin=self.bias+np.dot(self.wt, features[i])
                if (yin >= theta):
                    y=1
                else:
                    y=-1

                #weight and bias updation
                if(labels1[i]!=y):
                    a=alpha*(labels1[i]-y)
                    self.bias=self.bias+a
                    for k in range(0,9):
                        self.wt[k]=self.wt[k]+a*features[i][k]

        return self.wt


    def predict(self, features):
        sol=[]
        theta=0
        for i in range(len(features)):
            yin=self.bias+np.dot(self.wt, features[i])
           #step function
            if (yin >= theta):
                sol.append(1)
            else:
                sol.append(0)
        return sol

class MLP:
    def __init__(self):
        #Multilayer perceptron state here
        #Feel free to add methods
        return

    def preprocess(self,dataset):
        category=["net_ope_exp","net_con","tot_loa"]
        data1=normalizeData(dataset,category)
        category2=["can_off","can_inc_cha_ope_sea"]
        data2=encodeData(data1,category2)
        trainset,testset=trainingTestData(data2,0.67)
        trainfeature,trainlabel=getNumpy(trainset)
        testfeature,testlabel=getNumpy(testset)
        return trainfeature,trainlabel,testfeature,testlabel




    def train(self, features, labels):

        def transfer_function(x):
            return (1 / float(1 + np.exp(- x)))

        def derivative_transfer_function(x):
            func = transfer_function(x)
            return (func * (1 - func))

        def binary_to_bipolar(labels):
            for i in range(len(labels)):
                if (labels[i] == 0):
                    labels[i] = -1
            return labels

        labels1 = binary_to_bipolar(labels)

        input_neurons = 9
        hidden_neurons = 9
        self.wh = np.random.uniform(low=-0.1, high=0.1, size=(input_neurons * hidden_neurons)).reshape(input_neurons, hidden_neurons)
        self.bh = np.random.uniform(low=-0.1, high=0.1, size=(hidden_neurons)).reshape(1, hidden_neurons)
        self.wo = np.random.uniform(low=-0.1, high=0.1, size=(hidden_neurons)).reshape(hidden_neurons, 1)
        self.bo = np.random.uniform(-0.1, 0.1)

        alpha = 0.01
        theta=0.5
        t = time.time() + 55
        while (time.time() < t):
            for i in range(len(features)):
                #calculating output for hidden layer
                n_hidden = (features[i].reshape(1, 9)).dot(self.wh) + self.bh
                a_hidden = np.zeros(hidden_neurons, float).reshape(1, hidden_neurons)
                for j in range(hidden_neurons):
                    a_hidden[0][j] = transfer_function(n_hidden[0][j])

                #calculating output of output layer
                n_output = a_hidden.dot(self.wo) + self.bo
                a_output = transfer_function(n_output)

                #calculating sensitivity of output layer
                so = (a_output - labels1[i]) * derivative_transfer_function(n_output[0][0])

                # calculating sensitivity of hidden layer
                sh = np.zeros(hidden_neurons, float).reshape(1, hidden_neurons)
                for j in range(hidden_neurons):
                    sh[0][j] = derivative_transfer_function(n_hidden[0][j]) * self.wo[j][0] * so

                #hidden layer weight updation
                self.wh = self.wh - (alpha * (features[i].reshape(1, 9).T.dot(sh)))
                self.bh = self.bh - (alpha * sh)

                #output layer weight updation
                self.wo = self.wo - alpha * (a_hidden.T * so)
                self.bo = self.bo - alpha * so

        return

    def predict(self, features):


        def transfer_function(x):
            return (1 / float(1 + np.exp(- x)))

        sol = []
        input_neurons = 9
        hidden_neurons = 9
        theta = 0.5


        for i in range(len(features)):
            # calculating output for hidden layer
            n_hidden = (features[i].reshape(1, 9)).dot(self.wh) + self.bh
            a_hidden = np.zeros(hidden_neurons, float).reshape(1, hidden_neurons)
            for j in range(hidden_neurons):
                a_hidden[0][j] = transfer_function(n_hidden[0][j])

            # calculating output of output layer
            n_output = a_hidden.dot(self.wo) + self.bo
            y = transfer_function(n_output)

            if (y>= theta):
                sol.append(1)
            else:
                sol.append(0)
        return sol

class ID3:
    def __init__(self):
        #Decision tree state here
        #Feel free to add methods
        return

    def preprocess(self,dataset):
        category=["net_ope_exp","net_con","tot_loa"]
        data1=normalizeData(dataset,category)
        data1['net_ope_exp1'] = pd.qcut(data1['net_ope_exp'].rank(method='first'), 5, labels=["i", "ii", "iii", "iv", "v"])
        data1['net_con1'] = pd.qcut(data1['net_con'].rank(method='first'), 5, labels=["a", "b", "c", "d", "e"])
        data1['tot_loa1'] = pd.qcut(data1['tot_loa'].rank(method='first'), 5, labels=["1", "2", "3", "4", "5"])
        data1 = data1.loc[:, ~data1.columns.isin(['net_ope_exp', 'net_con', 'tot_loa'])]
        trainset,testset=trainingTestData(data1,0.67)
        trainfeature,trainlabel=getNumpy(trainset)
        testfeature,testlabel=getNumpy(testset)
        return trainfeature,trainlabel,testfeature,testlabel

    def train(self, features, labels):

        def BuildTree(dataSet, att, labels):

            # if all classes are same dont split further
            if labels[0] == np.all(labels):
                return labels[0]
            #if dataset has no feature dont split further
            if len(att) == 1:
                count_c = {}
                for v in labels:
                    if v not in count_c.keys():
                        count_c[v] = 0
                    count_c[v] =count_c+ 1
                sortclass = sorted(count_c.iteritems(), key=lambda x: x[1], reverse=True)
                return sortclass[0][0]

            # Finding information gain
            features_n = len(dataSet[0])

            No_of_Entries = len(labels)
            counts_l = {}
            for l in labels:
                cur = l
                if cur not in  counts_l.keys():  counts_l[cur] = 0
                counts_l[cur] =counts_l[cur]+ 1
            Entropy = 0.0
            for key in  counts_l:
                probability = float( counts_l[key]) / No_of_Entries
                Entropy =Entropy- probability * math.log(probability, 2)

            bestGain = 0.0
            bestFeature = -1
            for i in range(features_n):
                featureList = [ex[i] for ex in dataSet]
                uniqueV = set(featureList)
                Entropy_n = 0.0
                for value in uniqueV:
                    self.subDataSet = []
                    self.newLables = []
                    for k, featV in enumerate(dataSet):
                        if featV[i] == value:
                            reducedFeatVec = featV[:i]
                            reducedFeatVec = np.append(reducedFeatVec, featV[i + 1:])
                            self.subDataSet.append(reducedFeatVec)
                            self.newLables.append(labels[k])

                    probability = len(self.subDataSet) / float(len(dataSet))
                    numEntries = len(labels)
                    labelCounts = {}
                    for label in labels:
                        current = label
                        if current not in labelCounts.keys(): labelCounts[current] = 0
                        labelCounts[current] =labelCounts[current]+ 1
                    shannonEntropy = 0.0
                    for key in labelCounts:
                        prob = float(labelCounts[key]) / numEntries
                        shannonEntropy -= probability * math.log(prob, 2)

                    Entropy_n = Entropy_n+ probability * shannonEntropy

                infoGain = Entropy - Entropy_n
                if (infoGain > bestGain):
                    bestGain = infoGain
                    bestFeature = i

            bestFeatLabel = att[bestFeature]

            Tree = {bestFeatLabel: {}}
            featValues = [ex[bestFeature] for ex in dataSet]
            uniqueVals = set(featValues)

            del att[bestFeature]
            for value in uniqueVals:
                new_data = self.subDataSet
                new_labels=self.newLables
                Tree[bestFeatLabel][value] = BuildTree(new_data, att, new_labels)

            return Tree

        att = ["0", "1", "2", "3", "4"]
        self.DecisionTree = BuildTree(features, att, labels)

    def predict(self, features):
        def traverse(Tree,ins):
            str1 = Tree.keys()[0]
            sec = Tree[str1]
            Index=str1
            key = ins[int (Index)]
            value = sec[key]
            if isinstance(value, dict):
                Label1 = traverse(value, ins)
            else:
                Label1 = value
            return Label1

        sol = []
        for i in features:
            sol.append(traverse(self.DecisionTree,i))
        return sol

k = KNN()
ktrainf,ktrainl,ktestf,ktestl=k.preprocess(dataset)
k.train(ktrainf,ktrainl)
ksol=k.predict(ktestf)
k_accuracy=evaluate(ksol,ktestl)
print k_accuracy


p = Perceptron()
ptrainf,ptrainl,ptestf,ptestl=p.preprocess(dataset)
p.train(ptrainf,ptrainl)
psol=p.predict(ptestf)
p_accuracy=evaluate(psol,ptestl)
print p_accuracy

m = MLP()
mtrainf,mtrainl,mtestf,mtestl=m.preprocess(dataset)
m.train(mtrainf,mtrainl)
msol=m.predict(mtestf)
m_accuracy=evaluate(msol,mtestl)
print (m_accuracy)

id = ID3()
itrainf,itrainl,itestf,itestl=id.preprocess(dataset)
id.train(itrainf,itrainl)
isol=id.predict(itestf)
i_accuracy=evaluate(isol,itestl)
print (i_accuracy)