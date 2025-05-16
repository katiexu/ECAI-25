import copy
import numpy as np
import torch
import torch.nn as nn
import time
from sklearn.metrics import accuracy_score, f1_score
from datasets import get_mnist_numpy
from FusionModel_jax import DressedQuantumCircuitClassifier as QNet, translator
from Arguments import Arguments
import random


def get_param_num(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total:', total_num, 'trainable:', trainable_num)


def display(metrics):
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'

    print(YELLOW + "\nTest Accuracy: {}".format(metrics) + RESET)


def train(model, datasets):
    X,y=datasets
    model.fit(X,y)
    return model.loss_fn(X,y),model.score(X,y)


def test(model, datasets):
    X, y = datasets
    return model.loss_fn(X,y),model.score(X,y)



def Scheme_eval(design, task, weight):
    result = {}
    args = Arguments(**task)
    path = 'weights/'
    datasets = get_mnist_numpy(args, task['task'])

    train_datasets, val_datasets, test_datasets = datasets
    model = QNet(args, design)
    model.load_params(path + weight)
    # model.load_state_dict(torch.load(path + weight), strict=False)
    result['mae'] = test(model, test_datasets)
    return model, result


def Scheme(design, task, weight='base', epochs=None, verbs=None, save=None):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    args = Arguments(**task)
    epochs = args.epochs

    datasets = get_mnist_numpy(args, task['task'])

    train_datasets, val_datasets, test_datasets = datasets
    model = QNet(args, design)


    train_loss_list, val_loss_list = [], []
    best_val_acc = 0
    # print('val_loss: ', evaluate(model, val_loader, args))
    start = time.time()
    for epoch in range(epochs):
        train_loss,train_acc = train(model, train_datasets)
        train_loss_list.append(train_loss)
        val_loss, val_acc = test(model, val_datasets)
        val_loss_list.append(val_loss)
        test_loss,test_acc = test(model, test_datasets)
        val_acc = 0.5 * (val_acc + train_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if not verbs: print(epoch, train_loss, val_loss_list[-1], test_acc, 'saving model')
            best_model = copy.deepcopy(model)
        else:
            if not verbs: print(epoch, train_loss, val_loss_list[-1], test_acc)

    end = time.time()
    test_loss, test_acc=test(model, test_datasets)
    display(test_acc)
    print("Running time: %s seconds" % (end - start))
    report = {'train_loss_list': train_loss_list, 'val_loss_list': val_loss_list,
              'best_val_acc': best_val_acc, 'mae': test_acc}

    if save:
        best_model.save_params('weights/init_weight')
    return model, report


def pretrain(design, task, weight):
    args = Arguments(**task)

    datasets = get_mnist_numpy(args, task['task'])
    train_datasets, val_datasets, test_datasets = datasets
    model = QNet(args, design)
    model.load_params(weight)

    val_loss,val_acc = test(model, val_datasets)
    display(val_loss)

    return val_loss



if __name__ == '__main__':
    single = None
    enta = None

    arch_code = [10, 4]
    n_layers = 4
    n_qubits = 5
    single = [[i] + [1] * 2 * n_layers for i in range(1, n_qubits + 1)]
    enta = [[i] + [i + 1] * n_layers for i in range(1, n_qubits)] + [[n_qubits] + [1] * n_layers]

    task = 'MNIST-10'
    args = Arguments(task)

    design = translator(single, enta, 'full', arch_code, args.fold)

    # best_model, report = Scheme(design, 'MNIST', 'init', 30)
    weight = torch.load('weights/init_weight_half_10')
    report = pretrain(design, task, weight)

    # torch.save(best_model.state_dict(), 'weights/base_fashion')

