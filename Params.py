# -*- coding: utf-8 -*-

class Hyperparameters:
    """ Hyperparameters for DeepLog """

    seed = 711

    gpus = 1
    epoch = 200
    batch_size = 2048
    lr = 0.001
    # train, test 데이터셋의 이름
    trainset = "uri_train_dept_code"
    validset = "uri_test_dept_code"
    input_size = 1
    num_workers = 6
    
    # val_loss가 가장 낮은 모델의 체크포인트
    best_model = "checkpoint-epoch=88-val_loss=1.13.ckpt"
    
    # 정의된 이밴트의 클래스 개수(30분 단위별 -> 48개의 클래스)
    num_classes = 48
    num_layers = 2
    hidden_size = 64
    window_size = 10

    # for prediction
    num_candidates = 9
