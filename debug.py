import timm
import torchvision
from timm.data import create_loader
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import Subset

if __name__ == '__main__' :
    # 定义数据转换
    transforms = Compose([
        Resize((224, 224)),
        ToTensor()
    ])

    # 加载数据集
    dataset_train = torchvision.datasets.CIFAR100('../data/CIFAR100', train=True, transform=transforms)

    # 划分训练集和验证集的索引
    train_indices, val_indices = train_test_split(range(len(dataset_train)), test_size=0.2, random_state=42)
    print(len(train_indices), len(val_indices))

    # 使用 Subset 创建训练集和验证集的子集
    train_dataset = Subset(dataset_train, train_indices)
    val_dataset = Subset(dataset_train, val_indices)
    print(len(train_dataset), len(val_dataset))

    # 创建数据加载器
    train_loader = create_loader(train_dataset, input_size=(3, 224, 224), batch_size=32, is_training=True)
    val_loader = create_loader(val_dataset, input_size=(3, 224, 224), batch_size=32, is_training=False)

    for idx in range(len(val_dataset)):
        sample = val_dataset[idx]
        print(f"Sample {idx}: {sample}")

    for batch_idx, (input, target) in enumerate(val_loader):
        break
