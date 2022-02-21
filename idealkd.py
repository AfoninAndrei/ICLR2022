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
# works only for 2 agent case
# /opt/conda/bin/python /mlodata1/afonin/ekd.py
# nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
scheme = 'AKD' # ['FedAvg', 'AvgKD', 'AKD', 'PKD']
alpha = 1
dataset = 'CIFAR10' # ['CIFAR10', 'CIFAR100', 'MNIST']
num_clients = 2 #[2, 5, None]
partitions = [0.7, 0.3]

assert len(partitions) == num_clients, 'Number of data partitions is not equal to number of clients'
assert sum(partitions) == 1, 'Sum of data partitions if not equal to 1'
path = 'config/data-heterog-partit/' + dataset + '/' + str(num_clients) + ' agents'

with open(path + '/' + scheme + '_' + str(alpha) + '.json') as jsonFile:
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

lr = config['Optmizer']['lr']
weight_decay = config['Optmizer']['weight_decay']
obj = 'CE'#config['Optmizer']['Loss'] #'MSE'#
Epochs = config['Optmizer']['Epochs']
if dataset in ['CIFAR10', 'MNIST']:
    num_classes = 10
if dataset in ['CIFAR100']:
    num_classes = 100

train_loader_agents, test_loader, train_loader = data_prepare(dataset, batch_size=batch_size, test_batch_size=test_batch_size, alpha=alpha, num_clients=num_clients, partitions=partitions)

md = {}
for start_from in [1,2]:
    model_path = path + '/EKD/models_' + str(start_from) + '_' + str(alpha)
    md[str(start_from)] = torch.load(model_path)

Log = {}
agent_models_init = [model_prepare(config['Model'], num_classes) for _ in range(num_clients)]

clients = range(num_clients)
denom = sum(len(train_loader_agents[client].dataset.idx) for client in range(num_clients))

for c, client in enumerate(clients):
    Log[str(client)] = {}
    Train_loss, Test_loss = 1e3, 1e3
    Train_acc, Test_acc = 0, 0
    model = agent_models_init[client].to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Alpha = len(train_loader_agents[client].dataset.idx) / denom
    
    if c == 0:
        model_1, model_2 = md['1'][0].to(device), md['2'][0].to(device)
    if c == 1:
        model_1, model_2 = md['2'][0].to(device), md['1'][0].to(device)

    for epoch in range(1, Epochs + 1):
        train_loss, train_acc = train_ideal(model, [model_1, model_2], train_loader_agents[client], optimizer, obj, Alpha)
                
        test_loss, test_acc = test(model, test_loader, obj=obj)
        if test_acc > Test_acc:
            Test_loss = test_loss
            Test_acc = test_acc
            Train_loss = train_loss
            Train_acc = train_acc
        print('Test_acc = ', test_acc)
    Log[str(client)]['Train_loss'], Log[str(client)]['Test_loss'], Log[str(client)]['Train_acc'], Log[str(client)]['Test_acc'] = Train_loss, Test_loss, Train_acc, Test_acc
    print('Client ', client, ': Train acc = {:.3f}'.format(Train_acc), 'Test acc = {:.3f}'.format(Test_acc))
    
with open(path + '/IdealKD/result_' + str(alpha) + 'PKD.json', 'w') as jsonFile:
    json.dump(Log, jsonFile)
    jsonFile.close()
print('The Eval is finished. Log has been written to ' + path)