import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer
from nni.compression.pytorch.quantization.settings import change_default_quant_settings

import sys
sys.path.append('../models')
from mnist.naive import NaiveModel


def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('{:2.0f}%  Loss {}'.format(100 * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('Loss: {}  Accuracy: {}%)\n'.format(
        test_loss, 100 * correct / len(test_loader.dataset)))

def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=trans),
        batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=trans),
        batch_size=1000, shuffle=True)

    model = NaiveModel()
    '''you can change this to DoReFaQuantizer to implement it
    DoReFaQuantizer(configure_list).compress(model)
    '''
    configure_list = [{
            'quant_types': ['weight', 'input'],
            'quant_bits': {'weight': 8, 'input': 8},
            'op_names': ['conv1']
        }, {
            'quant_types': ['output'],
            'quant_bits': {'output': 8, },
            'op_names': ['relu1']
        }, {
            'quant_types': ['weight', 'input'],
            'quant_bits': {'weight': 8, 'input': 8},
            'op_names': ['conv2']
        }, {
            'quant_types': ['output'],
            'quant_bits': {'output': 8},
            'op_names': ['relu2']
        }, {
            'quant_types': ['output', 'weight', 'input'],
            'quant_bits': {'output': 8, 'weight': 8, 'input': 8},
            'op_names': ['fc1'],
        }, {
            'quant_types': ['output', 'weight', ],
            'quant_bits': {'output': 8, 'weight': 8, 'input': 8},
            'op_names': ['fc2'],
        }]
    change_default_quant_settings('weight', 'per_channel_symmetric', 'int')
    change_default_quant_settings('output', 'per_tensor_symmetric', 'int')
    change_default_quant_settings('input', 'per_tensor_symmetric', 'int')

    model = NaiveModel().to(device)
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    quantizer = QAT_Quantizer(model, configure_list, optimizer)
    quantizer.compress()

    model.to(device)
    for epoch in range(40):
        print('# Epoch {} #'.format(epoch))
        train(model, quantizer, device, train_loader, optimizer)
        test(model, device, test_loader)

    model_path = "mnist_model.pth"
    calibration_path = "mnist_calibration.pth"
    onnx_path = "mnist_model.onnx"
    input_shape = (1, 1, 28, 28)
    device = torch.device("cuda")

    calibration_config = quantizer.export_model(model_path, calibration_path, onnx_path, input_shape, device)
    print("Generated calibration config is: ", calibration_config)

if __name__ == '__main__':
    main()
