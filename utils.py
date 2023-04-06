from sklearn import svm
import numpy as np
import time

def load_data(data_path):
    train_data_path = data_path+"/train_data.npy"
    train_label_path = data_path+"/train_label.npy"
    test_data_path = data_path+"/test_data.npy"
    test_label_path = data_path+"/test_label.npy"
    train_data = np.load(train_data_path)
    train_label = np.load(train_label_path)
    test_data = np.load(test_data_path)
    test_label = np.load(test_label_path)
    return train_data, train_label, test_data, test_label

class quad_svm():
    def __init__(self, train_data, train_label, C, kernel, seed, scaler, pca):
        self.train_data = train_data
        self.train_label = train_label
        self.C = C
        self.kernel = kernel
        self.seed = seed
        self.clf_list = []
        self.scaler = scaler
        self.pca = pca

    def train_clfs(self):
        start = time.time()
        self.train_data = self.train_data.reshape(self.train_data.shape[0], -1)
        if self.pca:
            self.train_data = self.pca.fit_transform(self.train_data)
        if self.scaler:
            self.scaler.fit(self.train_data)
            self.train_data = self.scaler.transform(self.train_data)
        for i in range(4):
            label_k = np.where(self.train_label==i, 1, 0)
            clf = svm.SVC(C=self.C, 
                          kernel=self.kernel,
                          random_state=self.seed,
                          max_iter=int(1e7))
            clf.fit(self.train_data, label_k)
            self.clf_list.append(clf)
        print("time:", time.time()-start)
    
    def test_clfs(self, test_data, test_label):
        acc = 0
        correct = 0
        test_data = test_data.reshape(test_data.shape[0], -1)
        if self.pca:
            test_data = self.pca.transform(test_data)
        if self.scaler:
            test_data = self.scaler.transform(test_data)
        datasize = len(test_data)
        preds = []

        for i in range(datasize):
            decision_list = []
            for clf in self.clf_list:
                decision = clf.decision_function(test_data[i].reshape(1,-1))
                decision_list.append(decision)
            pred = np.argmax(decision_list)
            preds.append(pred)

        for i in range(datasize):
            if test_label[i] == preds[i]:
                correct += 1
        acc = correct/float(datasize)
        return acc, correct
