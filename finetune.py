import torch
import torchvision
import torchvision.transforms as transforms
import timm
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn, optim


# 1. 加载CIFAR100数据集
def load_cifar100(batch_size=128):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小以匹配预训练模型的输入
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = torchvision.datasets.CIFAR100(root='data/CIFAR100', train=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=8)

    test_set = torchvision.datasets.CIFAR100(root='data/CIFAR100', train=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=8)
    return train_loader, test_loader


# 2. 使用timm加载ImageNet1k上的预训练模型
def load_pretrained_model(model_name='deit_tiny_patch16_224', num_classes=100):
    model = timm.create_model(model_name, num_classes=num_classes)

    checkpoint = torch.load('ckpt/teacher_checkpoint/deit_tiny_patch16_224-a1311bcf.pth')
    # 从状态字典中移除分类头的权重
    # 分类头的权重名为`head.weight`和`head.bias`
    if 'head.weight' in checkpoint['model']:
        del checkpoint['model']['head.weight']
    if 'head.bias' in checkpoint['model']:
        del checkpoint['model']['head.bias']

    # 加载修改后的状态字典
    model.load_state_dict(checkpoint['model'], strict=False)

    return model


# 3. 微调模型
def train_model(model, train_loader, test_loader, epochs=100, patience=10, model_name='deit_tiny_patch16_224'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # 学习率调度器

    # best model info
    best_state_dict = None
    best_metric = 0.0
    best_epoch = 0

    # 早停机制相关变量
    best_accuracy = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data in tqdm(train_loader, desc=f'Epoch {epoch + 1} Training'):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        scheduler.step()  # 更新学习率

        # 在测试集上评估模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total

        print(f'Epoch {epoch + 1}: Average Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy}%')

        # 早停逻辑
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_no_improve = 0
            # 保存最好模型的权重信息
            torch.save(model.state_dict(), f'ckpt/teacher_checkpoint/best_model.pth')
            best_state_dict = model.state_dict()
            best_epoch = epoch + 1
            best_metric = accuracy
        else:
            epochs_no_improve += 1
            print(f"No improvement in accuracy for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs.')
            break

    # 4. 保存模型权重
    torch.save(best_state_dict, f'ckpt/teacher_checkpoint/{model_name}_cifar100_epoch{best_epoch}_{best_metric}.pth')


# 主函数
if __name__ == '__main__':
    batch_size = 128
    epochs = 100
    patience = 10
    model_name = 'deit_tiny_patch16_224'

    train_loader, test_loader = load_cifar100(batch_size)
    model = load_pretrained_model(model_name, 100)
    train_model(model, train_loader, test_loader, epochs, patience, model_name)
