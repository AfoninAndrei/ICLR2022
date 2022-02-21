import setGPU
import torch
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import json
import copy

from src.Data import *
from src.Models import *
from src.utils import *
from src.train_test import *
# /opt/conda/bin/python /mlodata1/afonin/centralized.py
# nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
scheme = 'Centralized' 
dataset = 'CIFAR10' # ['CIFAR10', 'CIFAR100', 'MNIST']
path = 'config/data-heterog/' + dataset

with open(path + '/' + scheme + '.json') as jsonFile:
    config = json.load(jsonFile)
    jsonFile.close()

seed = config['Seed']
batch_size = config['Data']['Batch_size']
test_batch_size = config['Data']['Eval_batch_size']

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Log = {}
lr = config['Optmizer']['lr']
weight_decay = config['Optmizer']['weight_decay']
obj = config['Optmizer']['Loss'] #'MSE'#
Epochs = config['Optmizer']['Epochs']
if dataset in ['CIFAR10', 'MNIST']:
    num_classes = 10
if dataset in ['CIFAR100']:
    num_classes = 100

try:
    lambd = config['Optmizer']['lambd']
except:
    pass

train_loader_agents, test_loader, train_loader = data_prepare(dataset, batch_size=batch_size, test_batch_size=test_batch_size, alpha=0, num_clients=1)

# train
agent_models_init = model_prepare(config['Model'], num_classes)

Log = {}
Train_loss, Test_loss = 1e3, 1e3
Train_acc, Test_acc = 0, 0
model = agent_models_init.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
for epoch in range(1, Epochs + 1):
    train_loss, train_acc = train(model, train_loader, optimizer, obj)
    test_loss, test_acc = test(model, test_loader, obj=obj)
    if test_acc > Test_acc:
        Test_loss = test_loss
        Test_acc = test_acc
        Train_loss = train_loss
        Train_acc = train_acc
Log['Train_loss'], Log['Test_loss'], Log['Train_acc'], Log['Test_acc'] = Train_loss, Test_loss, Train_acc, Test_acc
print('Train acc = {:.3f}'.format(Train_acc), 'Test acc = {:.3f}'.format(Test_acc))

with open(path + '/'+ scheme + '/' + 'result'+'.json', 'w') as jsonFile:
    json.dump(Log, jsonFile)
    jsonFile.close()

print('The Experiment is finished. Log has been written to ' + path)