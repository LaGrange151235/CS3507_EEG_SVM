from sklearn import svm
import numpy as np
import os
import copy
import utils


class svm_model():
    def __init__(self, train_data, train_label, test_data, test_label):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.clf_list = []

    def generate_clf(self):
        train_data = self.train_data
        test_data = self.test_data

        train_data = train_data.reshape(train_data.shape[0], -1)
        test_data = test_data.reshape(test_data.shape[0], -1)
        
        for k in range(4):
            print("Training %d clf" %(k))
            label_k = np.where(train_label==k, 1, 0)
            clf = svm.SVC(C=10, probability=True, random_state=0)
            print(train_data.shape, label_k.shape)
            clf.fit(train_data, label_k)
            self.clf_list.append(clf)

    def test_clfs(self, test_data, test_label):
        test_data = test_data.reshape(test_data.shape[0], -1)
        acc = 0
        correct = 0
        preds = []
        for k in range(len(test_data)):
            proba_list = []
            for clf in self.clf_list:
                proba = clf.predict_proba(test_data[k].reshape(1,-1))
                proba_list.append(proba[0][1])
            pred = np.argmax(proba_list)
            preds.append(pred)
        for k in range(len(test_label)):
            if test_label[k] == preds[k]:
                correct += 1
        acc = correct / float(len(test_label))
        return acc, correct
            

if __name__=="__main__":
    data_path = "./SEED-IV"
        
    model_list = []
    acc_list = []
    
    for experiment in os.listdir(data_path):
        experiment_path = data_path+"/"+experiment
        print("Experiment ID: ", experiment)

        train_data_list = []
        train_label_list = []
        test_data_list = []
        test_label_list = []

        for session in os.listdir(experiment_path):
            session_path = experiment_path+"/"+session
            print("Session ID: ", session)
            train_data = np.load(session_path+"/train_data.npy")
            train_label = np.load(session_path+"/train_label.npy")
            test_data = np.load(session_path+"/test_data.npy")
            test_label = np.load(session_path+"/test_label.npy")

            train_data_list.append(train_data)
            train_label_list.append(train_label)
            test_data_list.append(test_data)
            test_label_list.append(test_label)

        for k in range(len(train_data_list)):
            print("Take %d as test" %(k))
            train_input = copy.deepcopy(train_data_list)
            train_label = copy.deepcopy(train_label_list)
            test_input = copy.deepcopy(test_data_list)
            test_label = copy.deepcopy(test_label_list)

            #print(len(train_label))
            model_test_data = test_input[k]
            model_test_label = test_label[k]
            del train_input[k]
            del train_label[k]
            del test_input[k]
            del test_label[k]
            
            train_input = np.concatenate(train_input)
            train_label = np.concatenate(train_label)
            test_input = np.concatenate(test_input)
            test_label = np.concatenate(test_label)
            
            model = svm_model(train_input, train_label, test_input, test_label)
            model.generate_clf()
            
            acc, correct = model.test_clfs(test_data=model_test_data, test_label=model_test_label)
            print("acc: %f, correct: %d" %(acc, correct))

            acc_list.append(acc)

        acc_sum = sum(acc_list)
        print(acc_sum / float(len(acc_list)))
