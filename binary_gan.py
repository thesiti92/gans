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
d_iters_target = 5
image_size = 28
clamp_lower = -0.01
clamp_upper = 0.01
nc = 1
sw = 4
noise = torch.FloatTensor(batch_size, gen_noise_size).cuda()
fixed_noise = torch.FloatTensor(batch_size, gen_noise_size).normal_(0, 1).cuda()
num_weights = 64
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

# discriminator.load_state_dict(torch.load("%s/discriminator_epoch_39.pth" % model_out_folder))
# generator.load_state_dict(torch.load("%s/generator_epoch_39.pth" % model_out_folder))


gen_optimizer = optim.Adam(generator.parameters(), lr=0.001, betas=(.1, .999))
dis_optimizer = optim.Adam(discriminator.parameters(), lr=0.001, betas=(.1, .999))

trainset = ds.MNIST("./mnist", download=True, train=True, transform=transform)
testset = ds.MNIST("./mnist", download=True, train=False, transform=transform)
# trainset = torch.index_select(train_mnist.train_data, 0, torch.LongTensor(np.where(train_mnist.train_labels.numpy() == 2)[0])).type(torch.FloatTensor)
# testset = torch.index_select(test_mnist.test_data, 0, torch.LongTensor(np.where(test_mnist.test_labels.numpy() == 2)[0])).type(torch.FloatTensor)

train_feeder = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
test_feeder = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

input = torch.FloatTensor(batch_size, 1, image_size, image_size).cuda()
one = torch.FloatTensor([1]*batch_size).cuda()
mone = one * -1
gen_iterations = 0


for epoch in range(epochs):  # loop over the dataset multiple times
    data_iter = iter(train_feeder)
    i = 0
    while i < len(train_feeder):
        ############################
        # (1) Update D network
        ###########################
        for p in discriminator.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in generator update

        # train the discriminator Diters times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            d_iters = 100
        else:
            d_iters = d_iters_target
        j = 0
        while j < d_iters and i < len(train_feeder):
            j += 1

            # clamp parameters to a cube
            for p in discriminator.parameters():
                p.data.clamp_(clamp_lower, clamp_upper)

            data = data_iter.next()
            i += 1

            # train with real
            real_cpu, _ = data
            discriminator.zero_grad()
            batch_size = real_cpu.size(0)

            real_cpu = real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)
            inputv = Variable(input)

            errD_real = discriminator(inputv)
            errD_real.backward(one)

            # train with fake
            noise.resize_(batch_size, gen_noise_size).normal_(0, 1)
            noisev = Variable(noise, volatile = True) # totally freeze generator
            fake = Variable(generator(noisev).data)
            inputv = fake
            errD_fake = discriminator(inputv)
            errD_fake.backward(mone)
            errD = errD_real - errD_fake
            dis_optimizer.step()

        ############################
        # (2) Update G network
        ###########################
        for p in discriminator.parameters():
            p.requires_grad = False # to avoid computation
        generator.zero_grad()
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        noise.resize_(batch_size, gen_noise_size).normal_(0, 1)
        noisev = Variable(noise).cuda()
        fake = generator(noisev)
        errG = discriminator(fake)
        errG.backward(one)
        gen_optimizer.step()
        gen_iterations += 1
        if i % 100 == 0:
            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                % (epoch, epochs, i, len(train_feeder), gen_iterations,
                errD.data[0][0], errG.data[0][0], errD_real.data[0][0], errD_fake.data[0][0]))
        if gen_iterations % 100 == 0:
            real_cpu = real_cpu.mul(0.5).add(0.5)
            vutils.save_image(real_cpu, '%s/real_samples.png' % pic_out_folder)
            fake = generator(Variable(fixed_noise, volatile=True))
            fake.data = fake.data.mul(0.5).add(0.5)
            vutils.save_image(fake.data, '%s/fake_samples_%d.png' % (pic_out_folder, gen_iterations))
    torch.save(generator.state_dict(), '%s/generator_epoch_%d.pth' % (model_out_folder, epoch))
    torch.save(discriminator.state_dict(), '%s/discriminator_epoch_%d.pth' % (model_out_folder, epoch))


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
