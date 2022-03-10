import json
import os

seed = 0
dataset = 'CIFAR10' # ['MNIST', 'CIFAR10', 'CIFAR100']
alpha = 0.01

scheme = 'FedAvg'

path = 'config/data-heterog/' + dataset + '/' + scheme

try:
    os.mkdir('config/data-heterog/' + dataset)
except FileExistsError:
    pass

try:
    os.mkdir(path)
except FileExistsError:
    pass

model = 'VGG-9' # ['Net', 'VGG-9', 'ResNet-18', 'ResNet-34', 'ShuffleNet-V2']

lr = 0.001
weight_decay = 0
epochs = 40
loss = 'CE'

# scheme = ['AvgKD', 'FedAvg', 'FedProx', 'FedMA']
clients = 20

data = {
    'Dataset' : dataset,
    'Alpha': alpha,
    'Batch_size': 256,
    'Eval_batch_size': 2048
}

optimizer = {
    'Name' : 'Adam',
    'lr': lr,
    'weight_decay' : weight_decay,
    'Loss' : loss,
    'Epochs' : epochs
}

json_data = {
    'Seed' : seed,
    'Data' : data,
    'Model' : model,
    'Optmizer' : optimizer,
    'Scheme' : scheme,
    'Clients': clients,
    'Communications' : 10
}

with open(path + '_' + str(alpha) + '.json', 'w') as jsonFile:
    json.dump(json_data, jsonFile)
    jsonFile.close()

print('Config has been written to ' + path)