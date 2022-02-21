import torch
from PIL import Image
import numpy as np
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch.distributions.dirichlet import Dirichlet as Dirichlet
from src.Data import *
from src.Models import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Ditrichlet_sampler(numbered_list_indices, size, q):
    'For each agent returns left indices and the indices of datapoints for this agent'
    # q = sample distribution of classes for the agent from Dir(alpha*p), where p is uniform
    # indices = sample w/o return list of indices from numbered_list_indices according to q
    indices = []
    p = q.sample()
    # print(p.shape, len(numbered_list_indices), torch.argmax(p))
    if (len(numbered_list_indices[torch.argmax(p)]) < p[torch.argmax(p)]*size):
        p = q.sample()
    cat_distr = torch.distributions.Categorical(p)
    i = 0
    while len(indices) != size:
        i+=1
        c = cat_distr.sample()
        idxs = numbered_list_indices[c]
        if len(idxs) != 0:
            k = torch.randint(0, len(idxs), (1,))
            idx = idxs[k]
            indices.append(idx)
            numbered_list_indices[c] = torch.cat([idxs[:k], idxs[k+1:]])
        if i >= 30*size:
            indices = []
            i = 0
            p = q.sample()
            cat_distr = torch.distributions.Categorical(p)

    return numbered_list_indices, torch.tensor(indices)

def Sampler(numbered_list_indices, alpha, num_agents, partitions):
    'For each agent returns left indices and the indices of datapoints for this agent'
    # shuffled_indices_train = torch.randperm(len(dataset1.targets))
    # chunk_length = int(len(shuffled_indices_train) / num_clients)
    # indices = [shuffled_indices_train[i*chunk_length:(i+1)*chunk_length] for i in range(num_clients)]
    # for client in range(num_clients):
    #     dataset_t = Data(dataset1.data, dataset1.targets, transform_train, idx=indices[client])
    #     train_loader_agents.append(torch.utils.data.DataLoader(dataset_t, batch_size=batch_size, shuffle=True))
    indices = []
    taken = []
    # num_of_classes = int(len(numbered_list_indices) / num_agents)
    
    used = 0
    for client in range(num_agents):
        num_of_classes = int(len(numbered_list_indices) * partitions[client])
        indices.append(torch.cat([numbered_list_indices[c+used] for c in range(num_of_classes)]))
        taken.append(torch.randperm(len(indices[client]))[:int(len(indices[client])*alpha)])
        used += num_of_classes
    
    common = torch.cat([indices[client][taken[client]] for client in range(num_agents)])
    permutation = torch.randperm(len(common))
    cumulate = 0
    
    for client in range(num_agents):
        tens = indices[client]
        tens[taken[client]] = common[permutation[cumulate:cumulate+len(taken[client])]]
        indices[client] = tens
        cumulate += len(taken[client])

    return indices

def data_prepare(data, batch_size, test_batch_size, alpha, num_clients, partitions):
    """
    Function returns data loaders for the 2 actors(Teacher, Student) and test loader
    Parameters:
        - batch_size: train batch size
        - test_batch_size: test batch size
        - alpha: portion size of data for the student model(actor 2)
        - regime ['Same', 'Diff']:
    Return:
        - Train loader for the teacher(actor 1)
        - Train loader for the student(actor 2)
        - Test loader
    """
    if data == 'MNIST':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transform_train=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset1 = datasets.MNIST('data', train=True, download=True, transform=transform)
        dataset2 = datasets.MNIST('data', train=False, transform=transform)
    
    if data == 'CIFAR10':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_train=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((32, 32), 4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        dataset1 = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        dataset2 = datasets.CIFAR10('data', train=False, transform=transform)
    
    if data == 'CIFAR100':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        transform_train=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((32, 32), 4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        dataset1 = datasets.CIFAR100('data', train=True, download=True, transform=transform)
        dataset2 = datasets.CIFAR100('data', train=False, transform=transform)
    
    # dataset = Data(dataset1.data, dataset1.targets, transform_train, idx=None)
    dataset = Data(dataset1.data, torch.tensor(dataset1.targets), transform_train, idx=None)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=test_batch_size)

    indices = []
    classes = np.unique(np.array(dataset1.targets))
    num_classes = len(classes)
    train_loader_agents = []
    
    for i in classes:
        # idx = torch.where((dataset1.targets == i) == 1)[0]
        idx = torch.where((torch.tensor(dataset1.targets) == i) == 1)[0]
        indices.append(idx)

    #Dirichlet
    # chunk_length = int(len(dataset1.targets) / num_clients)

    # p = torch.tensor([1 / num_classes for i in range(num_classes)])
    # q = Dirichlet(alpha * p)

    # for client in range(num_clients - 1):
    #     indices, index_client = Ditrichlet_sampler(indices, chunk_length, q)
    #     dataset_t = Data(dataset1.data, torch.tensor(dataset1.targets), transform_train, idx=index_client)
    #     train_loader_agents.append(torch.utils.data.DataLoader(dataset_t, batch_size=batch_size, shuffle=True))
    
    # index_client = torch.cat([indices[i] for i in range(len(indices))])
    # index_client = index_client[torch.randperm(len(index_client))]
    # dataset_t = Data(dataset1.data, torch.tensor(dataset1.targets), transform_train, idx=index_client)
    # train_loader_agents.append(torch.utils.data.DataLoader(dataset_t, batch_size=batch_size, shuffle=True))
    
    #Usual
    indices = Sampler(indices, alpha, num_clients, partitions)
    for client in range(num_clients):
        # dataset_t = Data(dataset1.data, dataset1.targets, transform_train, idx=indices[client])
        dataset_t = Data(dataset1.data, torch.tensor(dataset1.targets), transform_train, idx=indices[client])
        train_loader_agents.append(torch.utils.data.DataLoader(dataset_t, batch_size=batch_size, shuffle=True))

    print('The separation of data is done.')
    return train_loader_agents, test_loader, train_loader

def model_prepare(model, num_classes):
    if model == 'VGG-9':
        return VGG(nn_arch="O", num_classes=num_classes, width=8, use_bn=False)
    
    if model == 'Net':
        return Net()
    
    if model == 'MLP':
        return MLP()

    if model == 'ResNet':
        return ResNet(resnet_size=8, num_classes=num_classes)

    print('MODEL ERROR: There is no such a model')

def make_onehot(label, N=10):
    'Returns one-hot vector from the given int'
    y = torch.zeros((label.shape[0], N), device=device)
    y[torch.arange(label.shape[0]), label] = 1
    return y

def soft_loss(preds, y):
    'Function for the soft cross entrohpy objective'
    logprobs = torch.nn.functional.log_softmax(preds, dim = -1)
    loss = -(y * logprobs).sum() / preds.shape[0]
    return loss

def average_params(agent_models):
    params = [md.state_dict() for md in agent_models]
    for key in params[0]:
          params[0][key] = sum([sd[key] for sd in params]) / len(params)

    agent_models[0].load_state_dict(params[0])
    for md in agent_models[1:]:
        md.load_state_dict(agent_models[0].state_dict())
    
    return agent_models