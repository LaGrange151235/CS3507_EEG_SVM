from operator import mod
from scipy.sparse import data
from utils import load_data, quad_svm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import os
import numpy as np
import argparse
import time

def generate_model(data_path, C, kernel, seed, scaler, pca, model_list, acc_list):
    train_data, train_label, test_data, test_label = load_data(data_path)
    model = quad_svm(train_data,
                     train_label,
                     C=C,
                     kernel=kernel,
                     seed=seed,
                     scaler=scaler,
                     pca=pca)
    model.train_clfs()
    acc, correct = model.test_clfs(test_data,
                                   test_label)
    model_list.append(model)
    acc_list.append(acc)
    print("%s: Accuracy %f, Correct: %d" %(data_path, acc, correct))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", default=1, type=float)
    parser.add_argument("--kernel", default='rbf', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--scaler", default='', type=str)
    parser.add_argument("--pca", default=0, type=int)
    args = parser.parse_args()
    print(args)
    print("===start_trainning==")
    C = args.C
    kernel = args.kernel
    seed = args.seed
    if args.scaler == 'standard':
        scaler = StandardScaler()
    elif args.scaler == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = args.scaler
    if args.pca > 0:
        pca = PCA(n_components=args.pca)
    else:
        pca = None


    dataset_path = "./SEED-IV"
    data_path_list = []
    model_list = []
    acc_list = []


    for dir in os.listdir(dataset_path):
        experiment_path = dataset_path+"/"+dir
        experiment_data_path_list = []
        for serial in os.listdir(experiment_path):
            serial_path = experiment_path+"/"+serial
            experiment_data_path_list.append(serial_path)
        data_path_list.append(experiment_data_path_list)
    
    start = time.time()
    for i in range(len(data_path_list)):
        for j in range(len(data_path_list[i])):
            data_path = data_path_list[i][j]
            generate_model(data_path,
                           C,
                           kernel,
                           seed,
                           scaler,
                           pca,
                           model_list,
                           acc_list)
    print("===final_result===")
    print("Training time: %f" %(time.time()-start))
    print("Average accuracy: %f" %(sum(acc_list)/float(len(acc_list))))
