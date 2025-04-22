import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from resnet_torch import ResNet18
import torchvision
import torchvision.transforms as transforms
from torch.quantization.observer import MovingAverageMinMaxObserver
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

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

#########4.1
print("4.1:\n")
conv_layers = []
linear_layers = []
for name, module in net.named_modules():
    if isinstance(module, nn.Conv2d):
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        conv_layers.append((name, num_params))

    elif isinstance(module, nn.Linear):
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        linear_layers.append((name, num_params))
        

# Summary
total_conv = sum(p for _, p in conv_layers)
total_linear = sum(p for _, p in linear_layers)

print(f"Total Conv2d layers  : {len(conv_layers)}")
print(f"Total Linear layers  : {len(linear_layers)}")
print(f"Total Conv2d params  : {total_conv:,}")
print(f"Total Linear params  : {total_linear:,}")





########4.2
print("4.2:\n")
layer_names = []
accuracies = []

# Target layers to test pruning on
target_layer_types = (nn.Linear, nn.Conv2d)

# Loop over all modules
for name, module in net.named_modules():
    if isinstance(module, target_layer_types):
        # Deepcopy to avoid modifying original model
        model_copy = copy.deepcopy(net)
        model_copy.to(device)
        model_copy.eval()

        # Find the same module in the copied model
        # You must re-traverse to get the prunable layer
        prunable_layer = dict(model_copy.named_modules())[name]

        # Apply 90% L1 pruning to this layer only
        prune.l1_unstructured(prunable_layer, name='weight', amount=0.9)

        # Evaluate the model
        acc = test(model_copy, testloader, cuda=True)
        #print(f"Pruned {name}: {acc:.2f}%")

        # Store results
        layer_names.append(name)
        accuracies.append(acc)

# Plot the results
plt.figure(figsize=(12, 5))
plt.bar(layer_names, accuracies, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy After Pruning Individual Layers (90%)')
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("pruning_accuracy_per_layer.png", dpi=300)
plt.show()


###########4.3
print("4.3:\n")
def check_k(k):
    model_copy = copy.deepcopy(net)
    model_copy.to(device)
    model_copy.eval()
    for name, module in model_copy.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=k)
    score = test(model_copy, testloader, cuda=True)
    #print(f"  Accuracy after pruning {k}: {score}%")
    return score


best_k = 0
L, R = 0, 100

threshold = test(net, testloader, cuda=True) - 2
##binary searching:
while L <= R:
    mid = (L + R) // 2
    k = mid / 100
    score = check_k(k)
    if score >= threshold:
        best_k = k
        L = mid + 1
    else:
        R = mid - 1

print(f"4.3: Best k: {best_k*100} %")
