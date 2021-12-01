# -*- coding: utf-8 -*-

import os
import sys
import time

os.chdir("/home/user/deeplog")
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from Params import Hyperparameters
from log_key_detection_models import DeepLog
from log_key_train import *
import pickle
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import pandas as pd
import numpy as np
import sys

#오차행렬, 정확도, 정밀도, 재현율 구하는 함수
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average='weighted')
    recall = recall_score(y_test, pred, average='weighted')
    f1 = f1_score(y_test, pred, average='weighted')
    print('오차 행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}'.format(accuracy, precision, recall))
    print('f1 스코어: {0: .4f}'.format(f1))

# window_size에 맞추어서 input과 label의 라벨값을 리턴하는 함수
def generate_pred(name, window_size):
    hdfs = list()
    hhhh = list()

    with open("data/" + name, "rb") as f:
        file = pickle.load(f)

    uri = []
    label = []

    for line in file:
        uri_tmp = []
        label_tmp = []
        for i in line:
            uri_tmp.append(i[0])
            label_tmp.append(i[1])
        uri.append(uri_tmp)
        label.append(label_tmp)
    # 길이가 windowsize보다 짧을 경우 패딩을 시킴
    for ln in uri:
        line = list(map(lambda n: n - 1, ln))
        ln = line + [-2] * (window_size + 1 - len(line))
        hdfs.append(tuple(ln))

    for ll in label:
        line = list(ll)
        ll = line + [-2] * (window_size + 1 - len(line))
        hhhh.append(tuple(ll))

    return hdfs, hhhh

# 다음에 올거라고 예측하는 event 라벨이 모델의 candidates 안에 존재하면 0, 존재하지않으면 1을 가지는 배열 생성
# 추후 룰 기반 결과와 비교해서 성능평가
def get_list(labeled, input_pred, total):
    pred_result = np.zeros((total, 1))
    idx = 0
    for i, j in zip(labeled, input_pred):
        if i in j:
            pred_result[idx] = 0
        else:
            pred_result[idx] = 1
        idx += 1
    return pred_result

def main():
    import numpy as np
    sys.stdout = open('model_result.txt', 'w')
    device = DeepLog.device
    hparams = Hyperparameters()
    # 모델을 gpu로 불러온다
    model_path = "/home/user/deeplog"
    model_name = hparams.best_model
    bestmodel = DeepLog.load_from_checkpoint(os.path.join(model_path, model_name))
    bestmodel.to(device)
    print("model_path: {}".format(os.path.join(model_path, model_name)))
    test_normal_loader, y_test = generate_pred(hparams.validset, hparams.window_size)

    total = 0
    correct = 0
    proba = []
    y_labeled = []
    inputed = []
    labeled = []
    input_pred = []
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 모델에 sequence를 window_size별로 나누어서 학습시킨다
    f = open(os.path.join("/home/user/deeplog", "result_" + hparams.validset + ".txt"), "w")
    with torch.no_grad():
        for line, y in tqdm(zip(test_normal_loader, y_test)):
            for i in range(len(line) - hparams.window_size):
                seq = line[i:i + hparams.window_size]
                label = line[i + hparams.window_size]
                y_label = y[i + hparams.window_size]
                if label == -2:
                    continue

                seq = (
                    torch.tensor(seq, dtype=torch.float, device=device).view(
                        -1, hparams.window_size, hparams.input_size
                    )
                )
                label = torch.tensor(label, device=device).view(-1)
                y_label = torch.tensor(y_label, device=device).view(-1)
                output = bestmodel(seq)
                # 모델 output을 이용해서 상위 n개의 라벨을 predicted배열에 추가한다
                predicted = torch.argsort(output, 1)[0][-hparams.num_candidates:]
                # 모델 output을 softmax함수에 적용시켜서 각각의 클래스의 확률을 담는 proba배열을 만든다.
                y_pred = F.softmax(output, dim=1)
                proba.append(y_pred.cpu().numpy()[0].tolist())

                inputed.append(np.array(seq.cpu()))
                labeled.append(np.array(label.cpu())[0])
                y_labeled.append(np.array(y_label.cpu()))
                input_pred.append(np.array(predicted.cpu()))
                total += 1
                f.write(str(seq))
                f.write("\n")
                f.write(str(predicted))
                f.write("\n")
                # 후보군안에 실제 라벨이 존재할 경우 correct를 1 증가시킨다.
                if label not in predicted:
                    correct += 1

    f.close()
    elapsed_time = time.time() - start_time

    print("elapsed_time: {:.3f}s".format(elapsed_time))
    print("correct_number : %d" % correct)
    print("total : %d" % total)
    accu = (total - correct) / total
    # 전체 개수중에서 제대로 예측한 개수
    print("accuracy : %f" % accu)
    # Compute precision, recall and F1-measure
    pred_list = get_list(labeled, input_pred, total)
    y_labeled = np.array(y_labeled)
    # 룰기반으로 생성된 정답데이터 셋을 이용해서 모델의 예측값과 성능평가
    get_clf_eval(y_labeled, pred_list)

    # PR-AUC구하기
    y = np.array(labeled)
    y_score = np.array(proba)
    class_y = np.arange(hparams.num_classes).tolist()

    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import label_binarize

    # Binarize the output
    y_test = label_binarize(y, classes=class_y)
    n_classes = y_test.shape[1]

    from sklearn.metrics import auc
    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # Compute micro-average ROC curve and ROC area
    precision["weighted"], recall["weighted"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel(), )
    average_precision["weighted"] = average_precision_score(y_test, y_score, average="weighted")
    print('PR-AUC: {0: .4f}'.format(average_precision["weighted"]))
    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall["weighted"], precision["weighted"], label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision["weighted"]))
    plt.legend(loc="lower left")
    plt.show()

    print("Finished Predicting")
    sys.stdout.close()