# pylint: disable=E0401

import torchvision.datasets as ds
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils


transform = transforms.Compose([transforms.ToTensor()])

batch_size, gen_noise_size = 64, 64
noise_deviaiton = 0.2
k_d, k_g = 1,1
fixed_z = Variable(torch.randn(batch_size, gen_noise_size)).cuda()



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(gen_noise_size, 128)
        self.fc2 = nn.Linear(128, 196)
        self.conv1 = nn.ConvTranspose2d(1, 2, kernel_size=1)
        self.conv2 = nn.ConvTranspose2d(2, 4, kernel_size=1)

    def forward(self, x):
        def common_layers(y):
            y = F.relu(y)
            y = F.dropout(y, training=self.training)
            return(y)
        x = common_layers(self.fc1(x))
        x = common_layers(self.fc2(x))
        x = x.view(batch_size, 1, 14, 14)
        x = common_layers(self.conv1(x))
        x = self.conv2(x)
        return F.tanh(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(8000, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        def common_layers(y):
            y = F.relu(y)
            y = F.dropout(y, training=self.training)
            return(y)

        x = x + Variable(torch.randn((batch_size, 1, 28, 28)) * noise_deviaiton).cuda()
        x = x.view(batch_size, 1, 28, 28)
        x = common_layers(self.conv1(x))
        x = common_layers(self.conv2(x))
        x = x.view(batch_size, -1)
        x = common_layers(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x).view(batch_size)

generator = Generator()
generator.cuda()

discriminator = Discriminator()
discriminator.cuda()

criterion = nn.MSELoss()
gen_optimizer = optim.SGD(generator.parameters(), lr=0.001, momentum=0.9)
dis_optimizer = optim.SGD(discriminator.parameters(), lr=0.001, momentum=0.9)


train_mnist = ds.MNIST("./mnist", download=True, train=True, transform=transform)
test_mnist = ds.MNIST("./mnist", download=True, train=False, transform=transform)
trainset = torch.index_select(train_mnist.train_data, 0, torch.LongTensor(np.where(train_mnist.train_labels.numpy() == 2)[0])).type(torch.FloatTensor)
testset = torch.index_select(test_mnist.test_data, 0, torch.LongTensor(np.where(test_mnist.test_labels.numpy() == 2)[0])).type(torch.FloatTensor)

train_feeder = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
test_feeder = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

for epoch in range(3):  # loop over the dataset multiple times

    running_dis_loss, running_gen_loss = 0.0, 0.0
    for i, data in enumerate(train_feeder, 0):
        # inputs, labels = data
        inputs = data
        real_samples = Variable(inputs.cuda())
        # labels = Variable(labels.cuda())
        for _ in range(k_d):
            # train discriminator
            z = Variable(torch.randn((batch_size, gen_noise_size))).cuda()
            
            fake_samples = generator(z)

            # zero the parameter gradients
            dis_optimizer.zero_grad()

            fake_labels = discriminator(fake_samples.view(batch_size,1,28,28).detach())
            #train on generated samples
            fake_loss = criterion(fake_labels, Variable(torch.FloatTensor(batch_size).random_(0,3)/10).cuda())
            fake_loss.backward()
            real_labels = discriminator(real_samples.view(batch_size, 1, 28, 28))
            #train on real samples
            real_loss = criterion(real_labels, Variable(torch.FloatTensor(batch_size).random_(7,12)/10).cuda())
            real_loss.backward()
            dis_optimizer.step()
        for _ in range(k_g):
            # train generator
            gen_optimizer.zero_grad()
            dis_labels = discriminator(fake_samples)
            gen_loss = criterion(dis_labels, Variable(torch.FloatTensor(batch_size).random_(7,12)/10.0).cuda())
            gen_loss.backward()
            gen_optimizer.step()

        # print loss statistics
        running_dis_loss += real_loss.data.mean() + fake_loss.data.mean()
        running_gen_loss += gen_loss.data.mean()

        if i % 20 == 19:    # print every 2000 mini-batches
            print('[%d, %5d] discrimenator loss: %.3f, generator loss: %.3f' %
                  (epoch + 1, i + 1, running_dis_loss / 2000, running_gen_loss / 2000))
            running_dis_loss, running_gen_loss = 0.0, 0.0
        if i % 90 == 89:
            vutils.save_image(inputs[0],
                            './real_samples.png',
                            normalize=True)
            fake = generator(fixed_z)
            vutils.save_image(fake.data[0].view(28, 28),
                    './fake_samples_epoch_%03d.png' % epoch,
                    normalize=True)


print('Finished Training')

# correct = 0
# total = 0
# for data in test_feeder:
#     images, labels = data
#     outputs = generator(Variable(images.cuda()))
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels.cuda()).sum()

# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))
