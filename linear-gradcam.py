import argparse
from PIL import Image
import numpy
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as utils
from torchvision import models

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='File path to an image')
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

def get_activations(activation_list):
    def forward_hook(module, input, output):
        print(module)
        activation_list.append(output.cpu().detach())
    return forward_hook

def get_gradients(gradient_list):
    def backward_hook(module, grad_input, grad_output):
        print(module)
        gradient_list.append(grad_output[0].cpu().detach())
    return backward_hook

if __name__ == '__main__':
    opt = get_options()

    model = models.resnet50(pretrained=True)
    model.eval()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    layer = opt.layer
    index = opt.index
    activation_list = []
    gradient_list = []

    for name, module in model.named_modules():
        if layer == name:
            print('Registering hooks on ' + layer)
            forward_handle = module.register_forward_hook(get_activations(activation_list))
            backward_handle = module.register_backward_hook(get_gradients(gradient_list))
            break
    else:
        print('Layer not found')
        exit()

    image = Image.open(opt.file).convert('RGB')
    image_tensor = transforms.functional.to_tensor(image)
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    image = transform(image).unsqueeze_(0).requires_grad_(True)

    if opt.cuda:
        model.cuda()
        score = model.forward(image.cuda())
    else:
        score = model.forward(image)

    if index == None:
        index = torch.argmax(score)
    one_hot = torch.zeros(score.size()).cuda()
    one_hot[0][index] = 1

    model.zero_grad()
    score.backward(one_hot)

    activations = activation_list[0].squeeze(0)
    gradients = gradient_list[0].squeeze(0)
    print(activations.__class__)
    print(activations.shape)
    print(activations)
    print(gradients.__class__)
    print(gradients.shape)
    print(gradients)

    weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
    cam = nn.functional.relu(torch.sum(weights * activations, dim=0, keepdim=True))
    cam = cam - torch.min(cam)
    cam = cam / torch.max(cam)
    print(cam.__class__)
    print(cam.shape)
    print(cam)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_tensor.shape[1:], interpolation=Image.BICUBIC)
    ])
    cam = cv2.applyColorMap(numpy.array(transform(cam)), cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    cam = transforms.functional.to_tensor(cam)
    utils.save_image(cam, 'cam.jpg')

    heatmap = image_tensor + cam
    heatmap = heatmap / torch.max(heatmap)
    utils.save_image(heatmap, 'heatmap.jpg')
