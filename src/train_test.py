import torch
import numpy as np
import torch.nn.functional as F
from src.utils import *
from src.methods import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_distillation(model, agent_models, train_loader, optimizer, obj='MSE', scheme='AvgKD'):
    """
    Function fot the KD training
    Parameters:
        - model: model for the training(student)
        - model2: model to produce logits(teacher)
        - device ['cpu', 'cuda']: computational module
        - train_loader: pytorch loader for the training data
        - optimizer: pytorch optimizer (ex. Adam)
        - epoch: Number of training epochs
        - obj['MSE', 'CE']: objective function
        - scheme['Iter KD', 'Averaged', 'Averaged GT']: KD scheme
        - model1(only for 'Averaged'): previous version of the student model
    Return:
        - History of the training loss
        - History of the training accuracy
    """
    model.train()
    Acc = []
    Loss = []
    if obj=='CE':
        LOSS = torch.nn.CrossEntropyLoss()
    if obj=='MSE':
        LOSS = torch.nn.MSELoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        if scheme=='PKD':
            loss = PKD(output, agent_models, data, target, obj)
        if scheme=='AvgKD':
            loss = AvgKD(output, agent_models, data, target, obj)
        if scheme=='AKD':
            loss = AKD(output, agent_models, data, obj)
        if scheme=='FedProx':
            loss = FedProx(output, [model.parameters(), agent_models[0].parameters()], data, target, agent_models[1], obj)
        if scheme=='FedMA':
            loss = FedMA(output, [model.parameters(), agent_models[0].parameters()], data, target, agent_models[1], obj)
        if scheme=='FedAvg':
            if obj=='CE':
                loss = LOSS(output, target)
            if obj=='MSE':
                loss = LOSS(F.softmax(output,dim=-1), make_onehot(target))

        loss.backward()
        Loss.append(loss.item())
        optimizer.step()
        acc = (output.argmax(dim=1) == target).float().mean().item()
        Acc.append(acc)
    accuracy = sum(Acc) / len(Acc)
    train_loss = sum(Loss) / len(Loss)
    return train_loss, accuracy

def train(model, train_loader, optimizer, obj='MSE'):
    """
    Function for the supervised (Ground truth) training
    Parameters:
        - model: model for the optimization
        - device ['cpu', 'cuda']: computational module
        - train_loader: pytorch loader for the training data
        - optimizer: pytorch optimizer (ex. Adam)
        - epoch: Number of training epochs
        - obj['MSE', 'CE']: objective function
    Return:
        - History of the train loss
        - History of the train accuracy
    """
    model.train()
    Acc = []
    Loss = []
    if obj=='CE':
        LOSS = torch.nn.CrossEntropyLoss()
    if obj=='MSE':
        LOSS = torch.nn.MSELoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if obj=='CE':
            loss = LOSS(output, target)
        if obj=='MSE':
            loss = LOSS(F.softmax(output,dim=-1), make_onehot(target))
        loss.backward()
        
        optimizer.step()
        Loss.append(loss.item())
        acc = (output.argmax(dim=1) == target).float().mean().item()
        Acc.append(acc)
    accuracy = sum(Acc) / len(Acc)
    train_loss = sum(Loss) / len(Loss)
    return train_loss, accuracy

def test(model, test_loader, obj='MSE'):
    """
    Function for the model evaluation
    Parameters:
        - model: model for the evaluation
        - device ['cpu', 'cuda']: computational module
        - test_loader: pytorch loader for the testing data
        - obj['MSE', 'CE']: objective function
    Return:
        - History of the test loss
        - History of the test accuracy
    """
    model.eval()
    test_loss = []
    if obj=='CE':
        LOSS = torch.nn.CrossEntropyLoss()
    if obj=='MSE':
        LOSS = torch.nn.MSELoss()
    Acc = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        if obj=='CE':
            test_loss.append(LOSS(output, target).item())
        if obj=='MSE':
            test_loss.append(LOSS(F.softmax(output,dim=-1), make_onehot(target)).item())
        Acc.append((output.argmax(dim=1) == target).float().mean().item())
    test_loss = sum(test_loss) / len(test_loss)
    accuracy = sum(Acc) / len(Acc)
    return test_loss, accuracy

def test_EKD(models, test_loader, output, k):
    """
    Function for the model evaluation
    Parameters:
        - model: model for the evaluation
        - device ['cpu', 'cuda']: computational module
        - test_loader: pytorch loader for the testing data
        - obj['MSE', 'CE']: objective function
    Return:
        - History of the test loss
        - History of the test accuracy
    """
    test_loss = []
    LOSS = torch.nn.MSELoss()
    Acc = []
    flag = False
    if output == None:
        flag = True
        output = []
    with torch.no_grad():
        for t, (data, target) in enumerate(test_loader):
            y = make_onehot(target)
            data, target = data.to(device), target.to(device)
            if flag:
                output.append(torch.zeros_like(make_onehot(target)))
            
            output[t] += k*(F.softmax(models[0].to(device)(data), dim=-1) + F.softmax(models[1].to(device)(data), dim=-1))
            test_loss.append(LOSS(output[t], y).item())
            Acc.append((output[t].argmax(dim=1) == target).float().mean().item())
    k *= -1
    test_loss = sum(test_loss) / len(test_loss)
    accuracy = sum(Acc) / len(Acc)
    return test_loss, accuracy, output, k

def train_ideal(model, models, train_loader, optimizer, obj='MSE', Alpha=0.5):
    model.train()
    Acc = []
    Loss = []
    LOSS = torch.nn.MSELoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        y = make_onehot(target)
        with torch.no_grad():
            s1 = F.softmax(models[0](data), dim=-1)
            s2 = F.softmax(models[1](data), dim=-1)
        
        if obj=='MSE':
            # loss = LOSS(F.softmax(output, dim=-1), Alpha * y + (1 - Alpha) * s1)
            loss = LOSS(F.softmax(output, dim=-1), Alpha * s1 + (1 - Alpha) * s2)
        if obj=='CE':
            # loss = soft_loss(output, Alpha * y + (1 - Alpha) * s1)
            loss = soft_loss(output, Alpha * s1 + (1 - Alpha) * s2)

        loss.backward()
        Loss.append(loss.item())
        optimizer.step()
        acc = (output.argmax(dim=1) == target).float().mean().item()
        Acc.append(acc)
    accuracy = sum(Acc) / len(Acc)
    train_loss = sum(Loss) / len(Loss)
    return train_loss, accuracy