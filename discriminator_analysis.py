import numpy as np
from chainer import serializers
from gan_models import Generator, Discriminator, InvertLoss
import chainer.functions as F
from scipy.misc import imsave
from chainer import cuda, Function, gradient_check, report, training, utils, Variable

import matplotlib.pyplot as plt

z_dim = 1
batch_size = 200

gen = Generator(z_dim)
dis = Discriminator()

serializers.load_npz("result/gen_iter_1200", gen)
serializers.load_npz("result/dis_iter_1200", dis)

real = np.load("rotated.npy")[:batch_size].reshape(-1, 1, 28, 28)

z = Variable(np.random.normal(
    size=(batch_size, z_dim, 1, 1)).astype(np.float32))
x_gen = gen(z)
x_gen = InvertLoss()(x_gen)

x = F.concat((x_gen, real), 0)
y = dis(x)
y_gen, y_data = F.split_axis(y, 2, 0)

# sigmoid_cross_entropy(x, 0) == softplus(x)
# sigmoid_cross_entropy(x, 1) == softplus(-x)

x_gen = x_gen[:10]
real = real[:10]

gen_save  = x_gen.transpose(1,2,0,3).reshape(-1, 10*28)
real_save = real.transpose(1,2,0,3).reshape(-1, 10*28)

imsave("generated_samples.png", gen_save.data)
imsave("real_samples.png", real_save.data)

print("generated loss: ", F.softplus(-y_gen).data[:10].reshape(-1))
print("real loss: ", F.softplus(y_data).data[:10].reshape(-1))



