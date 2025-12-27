import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, Subset, DataLoader

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


def load_cifar10_data(batch_size = 128, val_ratio = 0.1, test_size = 500):

    full_trainset = torchvision.datasets.CIFAR10(root = './data', train = True,
                                                download = True, transform = transform_train)

    total_size = len(full_trainset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    trainset, valset = random_split(full_trainset, [train_size, val_size])

    valset.dataset.transform = transform_test

    train_loader = DataLoader(trainset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(valset, batch_size = batch_size, shuffle = False)

    testset = torchvision.datasets.CIFAR10(root='./data', train = False,
                                        download = True, transform = transform_test)

    full_test_loader = DataLoader(testset, batch_size = 100, shuffle = False)

    full_test_dataset = full_test_loader.dataset
    subset_indices = list(range(test_size))
    subset = Subset(full_test_dataset, subset_indices)
    test_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader
