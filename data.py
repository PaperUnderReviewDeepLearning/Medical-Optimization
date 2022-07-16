import os
import scipy.io
import numpy as np
import random
import torchvision
import torch
import torch.utils.data as data
from torchvision import transforms, datasets
from os import walk
from PIL import Image
import os, sys
from torch.utils.data import random_split
import medmnist
from medmnist import INFO, Evaluator


def load_file(path):
  return np.load(path).astype(np.float32)


def resize(path):
  dirs = os.listdir(path)
  for item in dirs:
    if os.path.isfile(path + item):
      images = np.uint8(path + item)
      im = Image.fromarray(images)
      f, e = os.path.splitext(path + item)
      imResize = im.resize((32, 32), Image.ANTIALIAS)
      imResize.save(f)

def get_data(args):
  if args.dataset == 'mnist':
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))]
    )

    train_data = datasets.MNIST(
      root=args.data,
      download=True,
      train=True,
      transform=transform
    )
    test_data = datasets.MNIST(
      root=args.data,
      download=True,
      train=False,
      transform=transform
    )
    args.num_classes = 10
    args.in_dim = 28 * 28
  elif args.dataset == 'cifar10':
    transform = transforms.Compose([
      transforms.Scale(32),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
    )

    train_data = datasets.CIFAR10(
      root=args.data,
      download=True,
      train=True,
      transform=transform
    )
    test_data = datasets.CIFAR10(
      root=args.data,
      download=True,
      train=False,
      transform=transform
    )
    args.num_classes = 10
    args.in_dim = 3
  elif args.dataset == 'cifar100':
    transform = transforms.Compose([
      transforms.Scale(32),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
    )

    train_data = datasets.CIFAR100(
      root=args.data,
      download=True,
      train=True,
      transform=transform
    )
    test_data = datasets.CIFAR100(
      root=args.data,
      download=True,
      train=False,
      transform=transform
    )
    args.num_classes = 100
    args.in_dim = 3
  elif args.dataset == 'imagenet':
    transform = transforms.Compose([
      transforms.Scale(64),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
    )
    train_data = datasets.ImageFolder(
      os.path.join(args.data, 'tiny-imagenet-200', 'train'),
      transform=transform,
    )
    test_data = datasets.ImageFolder(
      os.path.join(args.data, 'tiny-imagenet-200', 'val'),
      transform=transform,
    )
    args.num_classes = 200
    args.in_dim = 3
  elif args.dataset == 'svhn':
    transform = transforms.Compose([
      transforms.Scale(32),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
    )

    train_data = datasets.SVHN(
      root=args.data,
      download=True,
      split='train',
      transform=transform
    )
    test_data = datasets.SVHN(
      root=args.data,
      download=True,
      split='test',
      transform=transform
    )
    args.num_classes = 10
    args.in_dim = 3
  elif args.dataset == 'caltech':
    args.num_classes = 101
    transform = transforms.Compose([
      transforms.CenterCrop(128),
      transforms.Scale(64),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
    )
    test_transform = transforms.Compose([
      transforms.CenterCrop(128),
      transforms.Scale(64),
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ]
    )
    train_data = datasets.Caltech101(
      root=args.data,
      download=False,
      transform=transform,
    )
    test_data = datasets.Caltech101(
      root=args.data,
      download=False,
      transform=test_transform,
    )


  elif args.dataset == 'pneumonia':
    train_transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
     transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,)),
    ])

    val_transform = transforms.Compose([
     #transforms.CenterCrop(128),
     transforms.ToTensor(),
     transforms.Lambda(lambda x: x.repeat(3, 1, 1)),

    ])

    train_data = torchvision.datasets.DatasetFolder("data/Processed_Data-pneumonia/train/",
                                                   loader = load_file,
                                                   extensions = "npy",
                                                   transform = train_transform)
    test_data = torchvision.datasets.DatasetFolder("data/Processed_Data-pneumonia/val/",
                                                  loader=load_file,
                                                  extensions="npy",
                                                  transform=val_transform)
    args.num_classes = 2 #6
    args.in_dim = 3

  elif args.dataset == 'brainmri':


    train_transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
     transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,)),
    ])

    val_transform = transforms.Compose([
     #transforms.CenterCrop(128),
     transforms.ToTensor(),
     transforms.Lambda(lambda x: x.repeat(3, 1, 1)),

    ])

    train_path = 'data/Preprocessed_Brain_MRI/train/'
    val_path = 'data/Preprocessed_Brain_MRI/val/'
    train_data = torchvision.datasets.ImageFolder(
      root=train_path,
      transform=train_transform
    )
    test_data = torchvision.datasets.ImageFolder(
      root=val_path,
      transform=val_transform
    )

    args.num_classes = 2
    args.in_dim = 3


  elif args.dataset == 'pathmnist':
    data_flag = 'pathmnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist,info['python_class'])
    # preprocessing
    data_transform = transforms.Compose([
      transforms.Resize(32),
      transforms.ToTensor(),
      transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_data = DataClass(split='train', transform=data_transform, download=True)
    test_data = DataClass(split='test', transform=data_transform, download=True)
    #pil_dataset = DataClass(split='train', download=True)


    args.num_classes = len(info['label'])
    args.in_dim = info['n_channels']
    print (train_data)

  elif args.dataset == 'tissuemnist':
    data_flag = 'tissuemnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist,info['python_class'])
    # preprocessing
    data_transform = transforms.Compose([
      transforms.Resize(32),
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
      transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_data = DataClass(split='train', transform=data_transform, download=True)
    test_data = DataClass(split='test', transform=data_transform, download=True)
    #pil_dataset = DataClass(split='train', download=True)


    args.num_classes = len(info['label'])
    args.in_dim = 3 # Lambda to make 3 channel
    print (train_data)

  elif args.dataset == 'organamnist':
    data_flag = 'organamnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist,info['python_class'])
    # preprocessing
    data_transform = transforms.Compose([
      transforms.Resize(32),
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
      transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_data = DataClass(split='train', transform=data_transform, download=True)
    test_data = DataClass(split='test', transform=data_transform, download=True)
    #pil_dataset = DataClass(split='train', download=True)


    args.num_classes = len(info['label'])
    args.in_dim = 3 # Lambda to make 3 channel
    print (train_data)

  elif args.dataset == 'dermamnist':
    data_flag = 'dermamnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist,info['python_class'])
    # preprocessing
    data_transform = transforms.Compose([
      transforms.Resize(32),
      transforms.ToTensor(),
      #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
      transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_data = DataClass(split='train', transform=data_transform, download=True)
    test_data = DataClass(split='test', transform=data_transform, download=True)
    #pil_dataset = DataClass(split='train', download=True)


    args.num_classes = len(info['label'])
    args.in_dim = 3 # Lambda to make 3 channel
    print (train_data)
  elif args.dataset == 'octmnist':
    data_flag = 'octmnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist,info['python_class'])
    # preprocessing
    data_transform = transforms.Compose([
      transforms.Resize(32),
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
      transforms.Normalize(mean=[.5], std=[.5])
    ])


    # load the data
    train_data = DataClass(split='train', transform=data_transform, download=True)
    test_data = DataClass(split='test', transform=data_transform, download=True)
    #pil_dataset = DataClass(split='train', download=True)


    args.num_classes = len(info['label'])
    args.in_dim = 3 # Lambda to make 3 channel
    print (train_data)

  elif args.dataset == 'bloodmnist':
    data_flag = 'bloodmnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    # preprocessing
    data_transform = transforms.Compose([
      transforms.Resize(32),
      transforms.ToTensor(),
      # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
      transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_data = DataClass(split='train', transform=data_transform, download=True)
    test_data = DataClass(split='test', transform=data_transform, download=True)
    # pil_dataset = DataClass(split='train', download=True)

    args.num_classes = len(info['label'])
    args.in_dim = 3  # Lambda to make 3 channel
    print (train_data)

  elif args.dataset == 'chestmnist':
    data_flag = 'chestmnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    # preprocessing
    data_transform = transforms.Compose([
      transforms.Resize(32),
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
      transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_data = DataClass(split='train', transform=data_transform, download=True)
    test_data = DataClass(split='test', transform=data_transform, download=True)
    # pil_dataset = DataClass(split='train', download=True)

    args.num_classes = len(info['label'])
    args.in_dim = 3  # Lambda to make 3 channel
    print (train_data)

  elif args.dataset == 'organmnist3d':
    data_flag = 'organmnist3d'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    # preprocessing
    data_transform = transforms.Compose([
      transforms.Resize(32),
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
      transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_data = DataClass(split='train', transform=data_transform, download=True)
    test_data = DataClass(split='test', transform=data_transform, download=True)
    # pil_dataset = DataClass(split='train', download=True)

    args.num_classes = len(info['label'])
    args.in_dim = 3  # Lambda to make 3 channel
    print (train_data)

  elif args.dataset == 'organcmnist':
    data_flag = 'organcmnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    # preprocessing
    data_transform = transforms.Compose([
      transforms.Resize(32),
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
      transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_data = DataClass(split='train', transform=data_transform, download=True)
    test_data = DataClass(split='test', transform=data_transform, download=True)
    # pil_dataset = DataClass(split='train', download=True)

    args.num_classes = len(info['label'])
    args.in_dim = 3  # Lambda to make 3 channel
    print (train_data)
  elif args.dataset == 'organsmnist':
    data_flag = 'organsmnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    # preprocessing
    data_transform = transforms.Compose([
      transforms.Resize(32),
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
      transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_data = DataClass(split='train', transform=data_transform, download=True)
    test_data = DataClass(split='test', transform=data_transform, download=True)
    # pil_dataset = DataClass(split='train', download=True)

    args.num_classes = len(info['label'])
    args.in_dim = 3  # Lambda to make 3 channel
    print (train_data)

  elif args.dataset == 'retinamnist':
    data_flag = 'retinamnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    # preprocessing
    data_transform = transforms.Compose([
      transforms.Resize(32),
      transforms.ToTensor(),
      #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
      transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_data = DataClass(split='train', transform=data_transform, download=True)
    test_data = DataClass(split='test', transform=data_transform, download=True)
    # pil_dataset = DataClass(split='train', download=True)

    args.num_classes = len(info['label'])
    args.in_dim = 3  # Lambda to make 3 channel
    print (train_data)

  elif args.dataset == 'breastmnist':
    data_flag = 'breastmnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    # preprocessing
    data_transform = transforms.Compose([
      transforms.Resize(32),
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
      transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_data = DataClass(split='train', transform=data_transform, download=True)
    test_data = DataClass(split='test', transform=data_transform, download=True)
    # pil_dataset = DataClass(split='train', download=True)

    args.num_classes = len(info['label'])
    args.in_dim = 3  # Lambda to make 3 channel
    print (train_data)






  else:
    raise NotImplementedError


  if args.ssl:
    all_indices = [i for i in range(len(train_data))]
    indices = random.sample(all_indices, int(args.percentage * len(train_data) / 100))

    sampler = data.sampler.SubsetRandomSampler(indices)
    train_loader = data.DataLoader(
      train_data,
      batch_size=args.batch_size,
      pin_memory=True,
      num_workers=int(4),
      shuffle=False,
      drop_last=True,
      sampler=sampler
    )

  else:
    train_loader = data.DataLoader(
      train_data,
      batch_size=args.batch_size,
      pin_memory=True,
      num_workers=int(4),
      shuffle=True,
      drop_last=True,
    )

  test_loader = data.DataLoader(
    test_data,
    batch_size=args.batch_size,
    pin_memory=True,
    num_workers=int(4),
    shuffle=True,
    drop_last=False,
  )



  return train_loader, test_loader, args
