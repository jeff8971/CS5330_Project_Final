import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

BATCH_SIZE = 128
LR = 0.01
EPOCH = 60
DEVICE = torch.device('cpu')

path_train = 'face_images/vgg_train_set'
path_valid = 'face_images/vgg_valid_set'

# Transformations for training data
transforms_train = transforms.Compose([
    transforms.Grayscale(),
    # Convert images back to grayscale as ImageFolder defaults to three channels
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    # Randomly adjust brightness and contrast
    transforms.ToTensor(),
])

# Transformations for validation data
transforms_valid = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.ToTensor(),
])

data_train = torchvision.datasets.ImageFolder(root=path_train,
                                              transform=transforms_train)
data_valid = torchvision.datasets.ImageFolder(root=path_valid,
                                              transform=transforms_valid)

train_set = torch.utils.data.DataLoader(dataset=data_train,
                                        batch_size=BATCH_SIZE, shuffle=True)
valid_set = torch.utils.data.DataLoader(dataset=data_valid,
                                        batch_size=BATCH_SIZE, shuffle=False)


class VGG(nn.Module):
    def __init__(self, *args):
        super(VGG, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                 padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2,
                            stride=2))  # This reduces the width and height by half
    return nn.Sequential(*blk)


conv_arch = ((2, 1, 32), (3, 32, 64), (3, 64, 128))
# After 5 vgg_blocks, the width and height will reduce five times, resulting in 224/32 = 7
fc_features = 128 * 6 * 6  # c * w * h
fc_hidden_units = 4096  # Any large number


def vgg(conv_arch, fc_features, fc_hidden_units):
    net = nn.Sequential()
    # Convolutional layers
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        # Each vgg_block will halve the width and height
        net.add_module("vgg_block_" + str(i + 1),
                       vgg_block(num_convs, in_channels, out_channels))
    # Fully connected layers
    net.add_module("fc", nn.Sequential(
        VGG(),
        nn.Linear(fc_features, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, 7)
    ))
    return net


model = vgg(conv_arch, fc_features, fc_hidden_units)
model.to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
criterion = nn.CrossEntropyLoss()

train_loss = []
train_acc = []
valid_loss = []
valid_acc = []
y_pred = []


def train(model, device, dataset, optimizer, epoch):
    model.train()
    correct = 0
    for i, (x, y) in tqdm(enumerate(dataset)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    train_acc.append(correct / len(data_train))
    train_loss.append(loss.item())
    print("Epoch {} Loss {:.4f} Accuracy {}/{} ({:.0f}%)".format(epoch, loss,
                                                                 correct,
                                                                 len(data_train),
                                                                 100 * correct / len(
                                                                     data_train)))


def valid(model, device, dataset):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(dataset)):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            pred = output.max(1, keepdim=True)[1]
            global y_pred
            y_pred += pred.view(pred.size()[0]).cpu().numpy().tolist()
            correct += pred.eq(y.view_as(pred)).sum().item()

    valid_acc.append(correct / len(data_valid))
    valid_loss.append(loss.item())
    print("Test Loss {:.4f} Accuracy {}/{} ({:.0f}%)".format(loss, correct,
                                                             len(data_valid),
                                                             100. * correct / len(
                                                                 data_valid)))


def RUN():
    for epoch in range(1, EPOCH + 1):
        train(model, device=DEVICE, dataset=train_set, optimizer=optimizer,
              epoch=epoch)
        valid(model, device=DEVICE, dataset=valid_set)
        # Save the model
        torch.save(model, 'model/model_vgg.pkl')


if __name__ == '__main__':
    RUN()