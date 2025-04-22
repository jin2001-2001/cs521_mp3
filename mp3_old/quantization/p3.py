from torch.quantization.observer import MovingAverageMinMaxObserver
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
from resnet_torch import ResNet18
import os
import time

###copy form ipynb###
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target):
    """ Computes the top 1 accuracy """
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct_one = correct[:1].view(-1).float().sum(0, keepdim=True)
        return correct_one.mul_(100.0 / batch_size).item()


def print_size_of_model(model):
    """ Prints the real size of the model """
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def load_model(quantized_model, model):
    """ Loads in the weights into an object meant for quantization """
    state_dict = model.state_dict()
    model = model.to('cpu')
    quantized_model.load_state_dict(state_dict)


def fuse_modules(model):
    """ Fuse together convolutions/linear layers and ReLU """
    torch.quantization.fuse_modules(model, [['conv1', 'relu1'],
                                            ['conv2', 'relu2'],
                                            ['fc1', 'relu3'],
                                            ['fc2', 'relu4']], inplace=True)
    
def test(model: nn.Module, dataloader: DataLoader, cuda=False) -> float:
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data

            if cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total
##end of copy

###code begin for quantization part


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

##code form reference of network
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

#trainset = torchvision.datasets.CIFAR10(root='../dataset', train=True,
#                                        download=True, transform=transform)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
#                                          shuffle=True, num_workers=16, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='../dataset', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=16, pin_memory=True)

net = ResNet18()
model_weights_path = '../resnet18_cifar10.pth'
state_dict = torch.load(model_weights_path, map_location='cpu')
net.load_state_dict(state_dict)
net.to(device)
net.eval()


##3-1 print size:
print_size_of_model(net) 
time_start = time.time()
score = test(net, testloader, cuda=True)
time_end = time.time()
print('Accuracy of the network on the test images: {}% - FP32'.format(score))
print('Time cost of the network on the test images: {} seconds '.format(time_end - time_start))


print("for quants")
quanti_net = ResNet18(q=True)
load_model(quanti_net, net)

####begin of calibration
quanti_net.qconfig = torch.quantization.default_qconfig
#print(quanti_net.qconfig)
torch.quantization.prepare(quanti_net, inplace=True)
test(quanti_net, testloader, cuda=False)
torch.quantization.convert(quanti_net, inplace=True)
####end of calibration

print_size_of_model(quanti_net)
quanti_net.to('cpu')
time_start = time.time()
score = test(quanti_net, testloader, cuda=False) # goes fault if running on gpu
time_end = time.time()
print('Accuracy of the fused and quantized network on the test images: {}% - INT8'.format(score))
print('Time cost of the network on the test images: {} seconds '.format(time_end - time_start))
