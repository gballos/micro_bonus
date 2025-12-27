import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, Subset, DataLoader

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar100_mean, cifar100_std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar100_mean, cifar100_std),
])

def load_cifar100_data(batch_size = 128, val_ratio = 0.1, test_size = 500):
    
    full_trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                 download=True, transform=transform_train)

    total_size = len(full_trainset)
    v_size = int(total_size * val_ratio)
    t_size = total_size - v_size
    
    trainset, valset = random_split(full_trainset, [t_size, v_size])
    valset.dataset.transform = transform_test

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform_test)

    subset_indices = list(range(test_size))
    test_subset = Subset(testset, subset_indices)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader