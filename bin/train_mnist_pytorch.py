#!/home/user/miniconda/envs/pytorch-1.5-cuda-10.1/bin/python
    
from __future__ import print_function
import argparse
import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
import os
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from alive_progress import alive_bar


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output


def train(log_interval, model, device, train_loader, optimizer, epoch):
    train_loss = 0
    loss_list = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

    #    if batch_idx % log_interval == 0:
    #        print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
    #              f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    loss_list = np.array(loss_list)
    train_loss = np.mean(loss_list)
    #print('Epoch ' + str(epoch) + ' - loss avg: ' + str(np.mean(loss_list)))

    return train_loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    #print(f'\nTest set: Average loss: {test_loss:.4f}, '
    #      f'Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

    return test_loss
    

@click.command()
@click.option('--seed', type=int, default=0)
@click.option('--epochs', type=int, default=3)
@click.option('--no-cuda', type=bool, default=False)
@click.option('--log-interval', type=int, default=10)
@click.option('--mode', type=str, default='det', help='Training mode: rand/seed/det')
@click.option('--out-path', type=str, default='data')
def start_training(epochs, no_cuda, seed, log_interval, mode, out_path):

    print('==================================')
    print('==============Training parameters:')
    print('seed: ' + str(seed))
    print('epochs: ' + str(epochs))
    print('mode: ' + mode)
    print('out-path: ' + out_path)
    #time.sleep(1.0)
    print('==================================')
    
    model_tag = str(random.randint(0, 10000))
    model_ouput_path = os.path.join(out_path, 'output_models', mode)

    if(mode == 'rand'):
        print("setting RANDOM mode...")
        # Noting to do, calling dummy function
        set_random_mode()
    elif(mode == 'seed'):
        print("setting SEED mode...")
        set_seed_mode(seed, True)
    else:
        print("setting DETERMINISTIC mode...")
        # Set all random seeds and possibly turn of GPU non-determinism
        set_deterministic_mode(seed, True)
    
    # Set GPU settings
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Load training and testing data
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(out_path, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
    batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(out_path, train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
    batch_size=64, shuffle=True, **kwargs)

    # Define model, device and optimizer
    model = Net()
    if torch.cuda.device_count() > 1:
      print(f'Using {torch.cuda.device_count()} GPUs!')
      model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7) decaying LR 
    optimizer.step()

    # Start training
    gpu_runtime = time.time()
    with alive_bar(epochs, title=f'Training:') as bar:
        for epoch in range(1, epochs + 1):
            train_loss = train(log_interval, model, device, train_loader, optimizer, epoch)
            test_loss = test(model, device, test_loader)
            # scheduler.step()
            optimizer.step()

            bar.text('[Epoch ' + str(epoch) + ']-[train loss: ' + str(train_loss) + ']-[test loss: ' + str(test_loss) + ']')
            bar()

    torch.save(model.state_dict(), os.path.join(model_ouput_path, model_tag + '_model.pth'))
    print("saving model to: " + os.path.join(model_ouput_path, model_tag + '_model.pth'))

    print(f'GPU Run Time: {str(time.time() - gpu_runtime)} seconds')


def set_deterministic_mode(seed, use_cuda):
    os.environ['PYTHONHASHSEED'] = str(seed) # Python general
    np.random.seed(seed) 
    random.seed(seed) # Python random
    torch.manual_seed(seed)
    if use_cuda: 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multiGPU
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False # Disable

def set_seed_mode(seed, use_cuda):
    os.environ['PYTHONHASHSEED'] = str(seed) # Python general
    np.random.seed(seed) 
    random.seed(seed) # Python random
    torch.manual_seed(seed)
    if use_cuda: 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multiGPU

def set_random_mode():
    return 0


if __name__ == '__main__':

    start_training()
