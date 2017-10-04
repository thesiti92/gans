import numpy as np
from chainer import serializers
from gan_models import Generator, Discriminator, InvertLoss
from chainer.training import make_extension
import chainer.functions as F
from scipy.misc import imsave
from chainer import cuda, Function, gradient_check, report, training, utils, Variable

import matplotlib.pyplot as plt

z_dim = 1
batch_size = 200
out_size = 20
width = 4
granularity = .1
num_samples = int(width*2/granularity)
nth_image = int(num_samples/out_size)

@make_extension(trigger=(1000, 'iteration'))
def plot_z(trainer, dataset = "resized.npy"):
    iter = trainer.updater.iteration

    gen = trainer.updater.gen
    dis = trainer.updater.dis

    real = np.load(dataset)[:batch_size].reshape(-1, 1, 28, 28)

    x = np.arange(-width, width, granularity)

    zs = np.array(x).astype('float32')
    zs_gpu = cuda.to_gpu(zs.reshape(num_samples**z_dim,z_dim,1,1))
    z = Variable(zs_gpu)
    x_gen = cuda.to_cpu(gen(z).data).reshape(-1, 28, 28)
    out = np.zeros((28, 28*out_size))
    for i in range(out_size):
        out[:, 28*i:28*(i+1)] = x_gen[i*nth_image]
    imsave("image_checkpoints/z_iter_%d.png" % iter, out)

def plot_z_old(trainer, dataset = "resized.npy"):
    iter = trainer.updater.iteration

    gen = Generator(z_dim)
    dis = Discriminator()

    serializers.load_npz("result/gen_iter_%d" % iter, gen)
    serializers.load_npz("result/dis_iter_%d" % iter, dis)

    real = np.load(dataset)[:batch_size].reshape(-1, 1, 28, 28)

    x = np.arange(-width, width, granularity)

    zs = np.array(x).astype('float32')
    zs_gpu = zs.reshape(num_samples**z_dim,z_dim,1,1)
    z = Variable(zs_gpu)
    x_gen = gen(z).data.reshape(-1, 28, 28)

    out = np.zeros((28, 28*out_size))
    for i in range(out_size):
        out[:, 28*i:28*(i+1)] = x_gen[i*nth_image]
    imsave("z_iter_%d.png" % iter, out)
# imsave("real_samples.png", real_save.data)

# print("generated loss: ", F.softplus(-y_gen).data[:10].reshape(-1))
# print("real loss: ", F.softplus(y_data).data[:10].reshape(-1))



