import numpy as np
from chainer import serializers
from gan_models import Generator, Discriminator, InvertLoss
import chainer.functions as F
from scipy.misc import imsave
from chainer import cuda, Function, gradient_check, report, training, utils, Variable

import matplotlib.pyplot as plt

z_dim = 1
batch_size = 200
width = 4
granularity = .1
num_samples = int(width*2/granularity)

gen = Generator(z_dim)
dis = Discriminator()

serializers.load_npz("result/gen_iter_1200", gen)
serializers.load_npz("result/dis_iter_1200", dis)

real = np.load("rotated.npy")[:batch_size].reshape(-1, 1, 28, 28)

x = np.arange(-width, width, granularity)

zs = np.array(x).astype('float32')
zs_gpu = zs.reshape(num_samples**z_dim,z_dim,1,1)
z = Variable(zs_gpu)
x_gen = gen(z)
for i, pic in enumerate(x_gen):
    imsave("generated_samples_%d.png" % i, pic.data.reshape(28,28))
# imsave("real_samples.png", real_save.data)

# print("generated loss: ", F.softplus(-y_gen).data[:10].reshape(-1))
# print("real loss: ", F.softplus(y_data).data[:10].reshape(-1))



