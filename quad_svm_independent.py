from operator import mod
from scipy.sparse import data
from utils import load_data, quad_svm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import os
import numpy as np
import argparse
import time

def generate_model(data_paths, test_path, C, kernel, seed, scaler, pca, model_list, acc_list):
    test_data, test_label, _, _ = load_data(test_path)
    train_data = []
    train_label = []
    for data_path in data_paths:
        train_data_, train_label_, _, _ = load_data(data_path)
        train_data.append(train_data_)
        train_label.append(train_label_)
    train_data = np.concatenate(train_data)
    train_label = np.concatenate(train_label)

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
    print("Test_data: %s, Accuracy %f, Correct: %d" %(test_path, acc, correct))


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
        print("Experiment_id: ", i)
        for j in range(len(data_path_list[i])):
            print("Test_session_id: ", j)
            test_data_path = data_path_list[i][j]
            train_data_paths = []
            for k in range(len(data_path_list[i])):
                if k == j:
                    continue
                else:
                    train_data_paths.append(data_path_list[i][k])
            generate_model(train_data_paths,
                           test_data_path,
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
