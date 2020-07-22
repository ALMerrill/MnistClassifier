import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from tqdm import tqdm

IMG_SIZE = 256
CROPPED_SIZE = 200
transformation = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomCrop(CROPPED_SIZE),
    transforms.ToTensor(),
])


class Mnist_Dataset(Dataset):
    def __init__(self, labels, root):
        super(Mnist_Dataset, self).__init__()
        self.labels = labels
        self.file_names = list(labels.keys())
        self.root = root

    def __getitem__(self, idx):
        img_name = self.file_names[idx]
        label = int(self.labels[img_name])
        image = Image.open(
            f'{self.root}/{label}/{img_name}')
        image = transformation(image)

        return image, label

    def __len__(self):
        return len(self.labels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(out1))
        return out2


class MnistClassifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MnistClassifier, self).__init__()

        hidden1 = 16
        hidden2 = 32
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden1,
                               kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=hidden1, out_channels=hidden2,
                               kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Based off of the shape of the image after passing through all layers
        img_dim_after = (((CROPPED_SIZE - 2) // 2) - 2) // 2
        fc = hidden2 * img_dim_after * img_dim_after
        self.fc = nn.Linear(fc, out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


if __name__ == '__main__':
    # Hyperparameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 1
    lr = 0.001

    data_root = 'Mnist_png-master/mnist_png'
    NUM_CLASSES = len(os.listdir(data_root + '/training'))

    datasets = {
        'training': {},
        'testing': {}
    }

    for dataset in ['training', 'testing']:
        dataset_root = f'{data_root}/{dataset}'
        for classname in os.listdir(dataset_root):
            class_root = f'{dataset_root}/{classname}/'
            for filename in os.listdir(class_root):
                datasets[dataset][filename] = classname

    train_dataset = Mnist_Dataset(
        datasets['training'], data_root + '/training')
    test_dataset = Mnist_Dataset(
        datasets['testing'], data_root + '/testing')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, num_workers=2)

    print('done setting up data loaders')

    model = MnistClassifier(1, NUM_CLASSES)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    print('set up loss function and optimizer')

    model.train()

    print(f'Cuda available? {torch.cuda.is_available()}')
    print('begin training')

    for epoch in range(NUM_EPOCHS):

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
                model.cuda()

            optim.zero_grad()

            outputs = model(inputs)

            loss = loss_func(outputs, labels)
            loss.backward()
            optim.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0
                print("Epoch {}/{}, Iter: {}, Loss: {:.3f}, ".format(epoch +
                                                                     1, NUM_EPOCHS, i, loss.data.item()))

    print('Finished Training')

    model_save_name = 'mnist_model_10.pt'
    path = f"models/{model_save_name}"
    torch.save(model.state_dict(), path)
    print('model saved')

    print('begin testing')

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' %
          (100 * correct / total))
