# -*- coding: utf-8 -*-

import os
import sys
import time

os.chdir("C:/Users/admin/DeepLog")
# os.environ['CUDA_LAUNCH_BLOCKING']="1"
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
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 시퀀스를 담고있는 배열을 불러와서 input(앞서 나오는 n개의 로그)과 output(다음으로 나올 실제라벨값)으로 분류
def generate(name, window_size, num_classes):
    num_sessions = 0
    inputs = []
    outputs = []
    # groupandevent.py에서 저장한 train dataset 불러오기
    with open("/home/user/deeplog/data/" + name, "rb") as f:
        seq_train = pickle.load(f)
    # 각 sequence단위별로 반복
    for line in seq_train:
        num_sessions += 1
        for i in range(len(line) - window_size):
            inputs_tmp = []

            for j in line[i:i + window_size]:
                inputs_tmp.append(j[0])
            inputs.append(inputs_tmp)

            outputs.append(line[i + window_size][0])
    print("Number of sessions({}): {}".format(name, num_classes))
    print("Number of seqs({}): {}".format(name, len(inputs)))

    dataset = TensorDataset(
        torch.tensor(inputs, dtype=torch.float, device=device),
        torch.tensor(outputs, device=device)
    )
    return dataset

def main():
    hparams = Hyperparameters()
    pl.seed_everything(hparams.seed)
    train_dset = generate(hparams.trainset, hparams.window_size, hparams.num_classes)
    # dataloader를 이용해서 batch별로 불러옴
    train_loader = DataLoader(
        train_dset, batch_size=hparams.batch_size, shuffle=True, pin_memory=True, num_workers=hparams.num_workers
    )
    valid_dset = generate(hparams.validset, hparams.window_size, hparams.num_classes)
    valid_loader = DataLoader(
        valid_dset, batch_size=hparams.batch_size, shuffle=False, pin_memory=True, num_workers=hparams.num_workers
    )
    # 모델 초기화
    model = DeepLog(
        input_size=hparams.input_size,
        hidden_size=hparams.hidden_size,
        window_size=hparams.window_size,
        num_layers=hparams.num_layers,
        num_classes=hparams.num_classes,
        lr=hparams.lr
    )
    # early_stopping 조건
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, strict=False, verbose=True, mode="min"
    )
    logger = TensorBoardLogger("logs", name="deeplog")
    # 모델에 대한 체크포인트 생성
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="deeplog/",
        filename="checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min"
    )
    # 트레이너 정의
    trainer = pl.Trainer(
        gpus=hparams.gpus,
        deterministic=True,
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback],
        max_epochs=hparams.epoch
    )
    # 모델 학습
    trainer.fit(model, train_loader, valid_loader)

if __name__ == "__main__":
    main()