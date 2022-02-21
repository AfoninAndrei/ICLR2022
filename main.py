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
# /opt/conda/bin/python /mlodata1/afonin/main.py
# nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
scheme = 'PKD' # ['FedAvg', 'AvgKD', 'AKD', 'PKD']
regime = 'Same' #['Same', 'Diff']
alpha = 0
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

distill_schemes = ['AKD', 'PKD', 'AvgKD']
baselines = ['FedAvg', 'FedProx', 'FedMA']

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

train_loader_agents, test_loader, train_loader = data_prepare(dataset, batch_size=batch_size, test_batch_size=test_batch_size, alpha=alpha, num_clients=num_clients, partitions=partitions)

# train
agent_models_init = [model_prepare(config['Model'], num_classes) for _ in range(num_clients)]

for i in range(config['Communications']):
    Log[str(i)] = {}
    print('Iteration = ', i)
    if scheme in distill_schemes:
        agent_models_init = [model_prepare(config['Model'], num_classes) for _ in range(num_clients)]
        # agent_models_init = [model_prepare('Net', num_classes), model_prepare('MLP', num_classes)]
        # agent_models_init = [model_prepare('ResNet', num_classes), model_prepare(config['Model'], num_classes)]
    if scheme in baselines:
        if scheme == 'FedAvg':
            agent_models_init = average_params(agent_models_init)
        # if scheme == 'FedProx':
        #     agent_models = average_params(agent_models_init)[0]
        #     agent_models_init = [model_prepare(config['Model'], num_classes) for _ in range(num_clients)]
            
    for client in range(num_clients):
        Log[str(i)][str(client)] = {}
        Train_loss, Test_loss = 1e3, 1e3
        Train_acc, Test_acc = 0, 0
        model = agent_models_init[client].to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(1, Epochs + 1):
            if (i >= 1) and (scheme in distill_schemes):
                if scheme == 'AvgKD':
                    other_agents = agent_models[:client] + agent_models[client+1:]
                    train_loss, train_acc = train_distillation(model, other_agents, train_loader_agents[client], optimizer, obj=obj, scheme=scheme)
                if scheme == 'PKD':
                    train_loss, train_acc = train_distillation(model, agent_models, train_loader_agents[client], optimizer, obj=obj, scheme=scheme)
                if scheme == 'AKD':
                    train_loss, train_acc = train_distillation(model, prev_model, train_loader_agents[client], optimizer, obj=obj, scheme=scheme)
                if scheme == 'FedProx':
                    agent_models.eval()
                    train_loss, train_acc = train_distillation(model, [agent_models, lambd], train_loader_agents[client], optimizer, obj=obj, scheme=scheme)
                if scheme == 'FedMA':
                    pass
            else:
                if (scheme == 'AKD') and (client > 0):
                    train_loss, train_acc = train_distillation(model, prev_model, train_loader_agents[client], optimizer, obj=obj, scheme=scheme)
                else:
                    train_loss, train_acc = train(model, train_loader_agents[client], optimizer, obj)
            
            test_loss, test_acc = test(model, test_loader, obj=obj)
            if test_acc > Test_acc:
                Test_loss = test_loss
                Test_acc = test_acc
                Train_loss = train_loss
                Train_acc = train_acc
        agent_models_init[client] = copy.deepcopy(model.cpu())
        Log[str(i)][str(client)]['Train_loss'], Log[str(i)][str(client)]['Test_loss'], Log[str(i)][str(client)]['Train_acc'], Log[str(i)][str(client)]['Test_acc'] = Train_loss, Test_loss, Train_acc, Test_acc
        if scheme == 'AKD':
            prev_model = copy.deepcopy(model.cpu())
    print('Client ', client, ': Train acc = {:.3f}'.format(Train_acc), 'Test acc = {:.3f}'.format(Test_acc))
    if scheme in distill_schemes:
        agent_models = [copy.deepcopy(md) for md in agent_models_init]

with open(path + '/'+ scheme + '/' + 'result'+'_' + str(alpha) + '.json', 'w') as jsonFile:
    json.dump(Log, jsonFile)
    jsonFile.close()

print('The Experiment is finished. Log has been written to ' + path)