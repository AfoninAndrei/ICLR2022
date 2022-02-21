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
start_from = 2 #[1, 2]
alpha = 0.01
dataset = 'CIFAR10' # ['CIFAR10', 'CIFAR100', 'MNIST']
num_clients = 2 #[2, 5, None]
mode = 'Eval' #['Train', 'Eval']
partitions = [0.5, 0.5]

assert len(partitions) == num_clients, 'Number of data partitions is not equal to number of clients'
assert sum(partitions) == 1, 'Sum of data partitions if not equal to 1'

path = 'config/data-heterog/' + dataset + '/' + str(num_clients) + ' agents'

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
# config['Communications'] = 1
if mode == 'Train':
    Log = {}
    # train
    agent_models_init = [model_prepare(config['Model'], num_classes) for _ in range(num_clients)]
    Models = []
    for i in range(config['Communications']):
        Log[str(i)] = {}
        print('Iteration = ', i)
        agent_models_init = [model_prepare(config['Model'], num_classes) for _ in range(num_clients)]
        # agent_models_init = [model_prepare('ResNet', num_classes), model_prepare(config['Model'], num_classes)]
        # agent_models_init = [model_prepare('Net', num_classes), model_prepare('MLP', num_classes)]
        clients = range(num_clients)
        if start_from == 2:
            clients = clients[::-1]
        for c, client in enumerate(clients):
            Log[str(i)][str(client)] = {}
            Train_loss, Test_loss = 1e3, 1e3
            Train_acc, Test_acc = 0, 0
            model = agent_models_init[client].to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            for epoch in range(1, Epochs + 1):
                if (i >= 1):
                    train_loss, train_acc = train_distillation(model, prev_model, train_loader_agents[client], optimizer, obj=obj, scheme=scheme)
                else:
                    if (c > 0):
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
            Models.append(copy.deepcopy(model.cpu()))
            prev_model = copy.deepcopy(model.cpu())
        print('Client ', client, ': Train acc = {:.3f}'.format(Train_acc), 'Test acc = {:.3f}'.format(Test_acc))

    with open(path + '/EKD/result_' + str(start_from) + '_' + str(alpha) + '.json', 'w') as jsonFile:
        json.dump(Log, jsonFile)
        jsonFile.close()
    print('The Experiment is finished. Log has been written to ' + path)

    model_path = path + '/EKD/models_' + str(start_from) + '_' + str(alpha)
    torch.save(Models, model_path)
    print('Models are saved to ' + model_path)

if mode == 'Eval':
    md = {}
    Log = {}
    for start_from in [1,2]:
        model_path = path + '/EKD/models_' + str(start_from) + '_' + str(alpha) + '_Diff'
        md[str(start_from)] = torch.load(model_path)
    test_loss, test_accuracy = [], []
    k = 1
    output_test, output_train = None, None
    for it in range(len(md['1'])):
        Log[str(it)] = {}
        Models = [md['1'][it], md['2'][it]]
        Log[str(it)]['Test_loss'], Log[str(it)]['Test_acc'], output_test, k = test_EKD(Models, test_loader, output_test, k)
        Log[str(it)]['Train_loss'], Log[str(it)]['Train_acc'], output_train, _ = test_EKD(Models, train_loader, output_train, k)
        print(Log[str(it)]['Test_acc'])
    
    with open(path + '/EKD/result_' + str(alpha) + '_Diff' + '.json', 'w') as jsonFile:
        json.dump(Log, jsonFile)
        jsonFile.close()
    print('The Eval is finished. Log has been written to ' + path)