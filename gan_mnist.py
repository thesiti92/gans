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


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

batch_size = 200
gen_noise_size = 100
epochs = 30
noise_deviaiton = 0.2
k_d, k_g = 1,1
image_size = 28
fixed_z = Variable(torch.randn(batch_size, gen_noise_size)).cuda()
sw = 4
num_weights = 64
nc = 1
model_out_folder = "./checkpoints"
pic_out_folder = "./sample_images"

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(gen_noise_size, 4*num_weights*sw*sw)
        self.norm1 = nn.BatchNorm2d(4*num_weights)
        self.dc1 = nn.ConvTranspose2d(num_weights*4, num_weights*2, 4, stride=1, padding=0)
        self.norm2 = nn.BatchNorm2d(2*num_weights)
        self.dc2 = nn.ConvTranspose2d(num_weights*2, num_weights, 4, stride=2, padding=1)
        self.norm3 = nn.BatchNorm2d(num_weights)
        self.dc3 = nn.ConvTranspose2d(num_weights, nc, 4, stride=2, padding=1)
        

    def forward(self, x):
        
        x = self.norm1(F.relu(self.fc1(x)).view(batch_size, num_weights*4, sw, sw))
        x = self.norm2(F.relu(self.dc1(x)))
        x = self.norm3(F.relu(self.dc2(x)))
        x = self.dc3(x)
        return F.tanh(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(nc, num_weights, kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(num_weights)
        self.conv2 = nn.Conv2d(num_weights, num_weights*2, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(2*num_weights)
        self.conv3 = nn.Conv2d(num_weights*2, num_weights*4, kernel_size=4, stride=1, padding=0)
        self.norm3 = nn.BatchNorm2d(4*num_weights)
        self.fc1 = nn.Linear(4*num_weights*sw*sw, 1)


    def forward(self, x):
        x = self.norm1(F.leaky_relu(self.conv1(x)))
        x = self.norm2(F.leaky_relu(self.conv2(x)))
        x = self.norm3(F.leaky_relu(self.conv3(x)))
        x = self.fc1(x.view(batch_size, -1))
        return F.sigmoid(x)

generator = Generator()
generator = generator.cuda()

discriminator = Discriminator()
discriminator = discriminator.cuda()

criterion = nn.BCELoss()
gen_optimizer = optim.Adam(generator.parameters(), lr=0.001)
dis_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

trainset = ds.MNIST("./mnist", download=True, train=True, transform=transform)
testset = ds.MNIST("./mnist", download=True, train=False, transform=transform)
# trainset = torch.index_select(train_mnist.train_data, 0, torch.LongTensor(np.where(train_mnist.train_labels.numpy() == 2)[0])).type(torch.FloatTensor)
# testset = torch.index_select(test_mnist.test_data, 0, torch.LongTensor(np.where(test_mnist.test_labels.numpy() == 2)[0])).type(torch.FloatTensor)

train_feeder = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
test_feeder = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

for epoch in range(epochs):  # loop over the dataset multiple times
    running_dis_loss, running_gen_loss = 0.0, 0.0
    for i, data in enumerate(train_feeder, 0):
        inputs, labels = data
        real_samples = Variable(inputs.cuda())
        # labels = Variable(labels.cuda())
        # train discriminator
        z = Variable(torch.randn((batch_size, gen_noise_size))).cuda()
        
        fake_samples = generator(z)

        # zero the parameter gradients
        dis_optimizer.zero_grad()

        fake_labels = discriminator(fake_samples.view(batch_size,1,28,28).detach())
        #train on generated samples
        fake_loss = criterion(fake_labels, Variable(torch.FloatTensor(batch_size).fill_(0)).cuda())
        fake_loss.backward()
        real_labels = discriminator(real_samples)
        #train on real samples
        real_loss = criterion(real_labels, Variable(torch.FloatTensor(batch_size).fill_(1)).cuda())
        real_loss.backward()
        dis_optimizer.step()
        # train generator
        gen_optimizer.zero_grad()
        dis_labels = discriminator(fake_samples)
        gen_loss = criterion(dis_labels, Variable(torch.FloatTensor(batch_size).fill_(1)).cuda())
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
                            '%s/real_samples.png' % pic_out_folder,
                            normalize=True)
            fake = generator(fixed_z)
            vutils.save_image(fake.data,
                    '%s/epoch_%d.png' % (pic_out_folder, epoch),
                    normalize=True)
    torch.save(generator.state_dict(), '%s/netG_epoch_%d.pth' % (model_out_folder, epoch))
    torch.save(discriminator.state_dict(), '%s/netD_epoch_%d.pth' % (model_out_folder, epoch))


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
