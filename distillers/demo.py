import timm
import torch
import torch.nn.functional as F
import torch.nn as nn


if __name__ == '__main__':

    all_mobilenetv2_models = timm.list_models("*mobilenetv2*")
    print(all_mobilenetv2_models)

    model = timm.create_model('efficientnet_b0', num_classes=100)
    model2 = timm.create_model('mobilenetv2_100', num_classes=100)
    print(model, model2)
    # model.head = torch.nn.Linear(model.head.in_features, 100)
    checkpoint = torch.load('../ckpt/teacher_checkpoint/swin_tiny_patch4_window7_224.pth')

    # 从状态字典中移除分类头的权重
    # 注意：这里假设分类头的权重名为`fc.weight`和`fc.bias`
    if 'head.weight' in checkpoint['model']:
        del checkpoint['model']['head.weight']
    if 'head.bias' in checkpoint['model']:
        del checkpoint['model']['head.bias']

    # 加载修改后的状态字典
    model.load_state_dict(checkpoint['model'], strict=False)

    # 这时你可以根据需要修改分类头
    model.head = torch.nn.Linear(model.head.in_features, 100)

    print(model)
