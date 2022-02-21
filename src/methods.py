import torch
import numpy as np
import torch.nn.functional as F
from src.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def AvgKD(prediction, agent_models, x, y, obj):
    'NOTICE: Here you need to pass M-1 models from other agents'
    with torch.no_grad():
        agents_pred = (torch.sum(torch.stack([F.softmax(md.to(device)(x), dim=-1) for md in agent_models]), 0) + make_onehot(y)) / (len(agent_models) + 1)
        # agents_pred = (torch.sum(torch.stack([F.softmax(md.to(device)(x), dim=-1) for md in agent_models]), 0) / len(agent_models)  + make_onehot(y)) / 2
    if obj=='MSE':
        LOSS = torch.nn.MSELoss()
        loss = LOSS(F.softmax(prediction, dim=-1), agents_pred)

    if obj=='CE':
        loss = soft_loss(prediction, agents_pred)
    
    return loss

def PKD(prediction, agent_models, x, y, obj):
    'NOTICE: Here you need to pass all M agent models'
    with torch.no_grad():
        agents_pred = torch.sum(torch.stack([F.softmax(md.to(device)(x), dim=-1) for md in agent_models]), 0) / len(agent_models)

    if obj=='MSE':
        LOSS = torch.nn.MSELoss()
        loss = LOSS(F.softmax(prediction, dim=-1), agents_pred)

    if obj=='CE':
        loss = soft_loss(prediction, agents_pred)
    return loss

def AKD(prediction, prev_agent, x, obj):
    with torch.no_grad():
        agents_pred = F.softmax(prev_agent.to(device)(x), dim=-1)

    if obj=='MSE':
        LOSS = torch.nn.MSELoss()
        loss = LOSS(F.softmax(prediction, dim=-1), agents_pred)

    if obj=='CE':
        loss = soft_loss(prediction, agents_pred)
    return loss

def FedProx(prediction, parameters, x, y, lambd, obj):
    reg = torch.tensor(0.)
    for param1, param2 in zip(parameters[0], parameters[1]):
        reg += torch.norm(param1 - param2)**2
    if obj=='CE':
        LOSS = torch.nn.CrossEntropyLoss()
        loss = LOSS(prediction, y) + lambd * reg
    if obj=='MSE':
        LOSS = torch.nn.MSELoss()
        loss = LOSS(F.softmax(prediction, dim=-1), make_onehot(y)) + lambd * reg
    
    return loss

def FedMA(prediction, parameters, x, y, lambd, obj):
    pass