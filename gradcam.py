import os
import argparse
from PIL import Image
import numpy
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as utils
from torchvision import models

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURR_DIR, 'output')

ELEMENT = 0
MAXDEPTH = 1
AVGDEPTH = 2
MAXPOOL = 3
AVGPOOL = 4
GMP = 5
GAP = 6
MODE_NUM = 7

MODE_STR = {
    ELEMENT: 'element',
    MAXDEPTH: 'maxdepth',
    AVGDEPTH: 'avgdepth',
    MAXPOOL: 'maxpool',
    AVGPOOL: 'avgpool',
    GMP: 'gmp',
    GAP: 'gap',
}

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Path to the dataset')
    parser.add_argument('--mode', type=int, default=0, help='Choose GradCAM mode: 0(element) | 1(maxdepth) | 2(avgdepth) | 3(maxpool) | 4(avgpool) | 5(gmp) | 6(gap)')
    parser.add_argument('--all', action='store_true', default=False, help='Execute all CAM modes')
    parser.add_argument('--layer', type=str, help='Network layer to extract CAM from')
    parser.add_argument('--index', type=int, help='Class index')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use CUDA for computation')
    opt = parser.parse_args()

    opt.cuda = opt.cuda and torch.cuda.is_available()
    if opt.cuda:
        print('Using CUDA for computation')
    else:
        print('Using CPU for computation')

    return opt

def get_activation(activation_list):
    def forward_hook(module, input, output):
        activation_list.append(output.cpu().detach())
    return forward_hook

def get_gradient(gradient_list):
    def backward_hook(module, grad_input, grad_output):
        gradient_list.append(grad_output[0].cpu().detach())
    return backward_hook

def extract_cam(activation, gradient, mode, size):
    if mode == ELEMENT:
        weight = gradient
    elif mode == MAXDEPTH:
        weight, _ = torch.max(gradient, dim=0, keepdim=True)
    elif mode == AVGDEPTH:
        weight = torch.mean(gradient, dim=0, keepdim=True)
    elif mode == MAXPOOL:
        weight = nn.functional.max_pool2d(gradient, 4, stride=4).unsqueeze(0)
        print(weight.shape)
        weight = nn.functional.interpolate(weight, gradient.shape[1:], mode='bicubic').squeeze(0)
    elif mode == AVGPOOL:
        weight = nn.functional.avg_pool2d(gradient, 3, stride=1, padding=1)
    elif mode == GMP:
        weight, _ = torch.max(gradient, dim=1, keepdim=True)
        weight, _ = torch.max(weight, dim=2, keepdim=True)
    elif mode == GAP:
        weight = torch.mean(gradient, dim=(1, 2), keepdim=True)

    cam = torch.sum(weight * activation, dim=0, keepdim=True)
    cam = nn.functional.relu(cam)

    # cam = torch.mean(weight * activation, dim=0, keepdim=True)
    # cam = torch.exp(cam)

    cam = cam - torch.min(cam)
    cam = cam / torch.max(cam)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size, interpolation=Image.BICUBIC)
    ])
    cam = cv2.applyColorMap(numpy.array(transform(cam)), cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    cam = transforms.functional.to_tensor(cam)

    return cam

if __name__ == '__main__':
    opt = get_options()

    model = models.resnext101_32x8d(pretrained=True)
    model.eval()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if opt.layer != 'input':
        activation_list = []
        gradient_list = []
        for name, module in model.named_modules():
            if name == opt.layer:
                print('Registering hooks on ' + name)
                print(module)
                forward_handle = module.register_forward_hook(get_activation(activation_list))
                backward_handle = module.register_backward_hook(get_gradient(gradient_list))
                break
        else:
            print('Layer not found')
            exit()

    image = Image.open(opt.dataset).convert('RGB')
    image_tensor = transforms.functional.to_tensor(image)
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    image = transform(image).unsqueeze(0)
    image.requires_grad_(True)

    if opt.cuda:
        model.cuda()
        image = image.cuda()
        image.retain_grad()

    score = model.forward(image)

    if opt.index == None:
        opt.index = torch.argmax(score)

    if opt.cuda:
        one_hot = torch.zeros(score.size()).cuda()
    else:
        one_hot = torch.zeros(score.size())
    one_hot[0][opt.index] = 1

    model.zero_grad()
    score.backward(one_hot)

    if opt.layer == 'input' :
        activation = image.cpu().detach().squeeze(0)
        gradient = image.grad.cpu().detach().squeeze(0)
    else:
        activation = activation_list[0].squeeze(0)
        gradient = gradient_list[0].squeeze(0)

    print(activation.shape)
    print(gradient.shape)

    if opt.all:
        for mode in range(MODE_NUM):
            cam = extract_cam(activation, gradient, mode, image_tensor.shape[1:])

            heatmap = image_tensor + cam
            heatmap = heatmap / torch.max(heatmap)

            utils.save_image(cam, os.path.join(OUTPUT_DIR, str(mode) + MODE_STR[mode] + '-cam.jpg'))
            utils.save_image(heatmap, os.path.join(OUTPUT_DIR, str(mode) + MODE_STR[mode] + '-heatmap.jpg'))
    else:
        cam = extract_cam(activation, gradient, opt.mode, image_tensor.shape[1:])

        heatmap = image_tensor + cam
        heatmap = heatmap / torch.max(heatmap)

        utils.save_image(cam, os.path.join(OUTPUT_DIR, str(opt.mode) + MODE_STR[opt.mode] + '-cam.jpg'))
        utils.save_image(heatmap, os.path.join(OUTPUT_DIR, str(opt.mode) + MODE_STR[opt.mode] + '-heatmap.jpg'))
