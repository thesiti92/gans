# ^-^ coding: UTF-8 ^-^
import argparse
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
import chainer
import cupy
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers, reporter
from chainer import Link, Chain, ChainList
from chainer import function
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from itertools import product
from line_labler import histogram
from digit_generator import *
from z_dim_fun import plot_z
from scipy.misc import imsave
from gan_models import Generator, Discriminator, InvertLoss

width = 4
# num_samples = 15

class GAN_Updater(training.StandardUpdater):

    def __init__(self, iterator, generator, discriminator, optimizers,
                 converter=convert.concat_examples, device=None, z_dim=2,):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        self.gen = generator
        self.dis = discriminator
        self._optimizers = optimizers
        self.converter = converter
        self.device = device

        self.iteration = 0

        self.z_dim = z_dim

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)
        x_data = in_arrays

        batchsize = x_data.shape[0]
        # z = Variable(cuda.cupy.random.normal(
        #     size=(batchsize, self.z_dim, 1, 1), dtype=np.float32))
        z = Variable(cuda.cupy.random.uniform(
           size=(batchsize, self.z_dim, 1, 1), low=-width, high=width, dtype=np.float32))
        # zs = list(product(np.linspace(-width,width,num_samples), 
        #                   np.linspace(-width,width,num_samples)))
        # zs = np.array(zs).astype('float32')
        # zs_gpu = cuda.to_gpu(zs.reshape(num_samples*num_samples,2,1,1), device=0)
        # z = Variable(zs_gpu)
        global x_gen
        x_gen = self.gen(z)
        x_gen = InvertLoss()(x_gen)

        # concatã—ãªã„ã¾ã¾disã«é€šã™ã¨ã€bnãŒæ‚ªã•ã‚’ã™ã‚‹
        x = F.concat((x_gen, x_data), 0)
        y = self.dis(x)
        y_gen, y_data = F.split_axis(y, 2, 0)

        # sigmoid_cross_entropy(x, 0) == softplus(x)
        # sigmoid_cross_entropy(x, 1) == softplus(-x)
        loss_gen = F.sum(F.softplus(-y_gen))
        loss_data = F.sum(F.softplus(y_data))
        loss = (loss_gen + loss_data) / batchsize
        # print loss.data

        for optimizer in self._optimizers.values():
            optimizer.target.cleargrads()

        # compute gradients all at once
        loss.backward()

        for optimizer in self._optimizers.values():
            optimizer.update()

        reporter.report(
            {'loss': loss,
             'loss_gen': loss_gen / batchsize,
             'loss_data': loss_data / batchsize})


def save_x(x_gen, fn):
    x_gen_img = cuda.to_cpu(x_gen.data)
    n = x_gen_img.shape[0]
    n = n // 15 * 15
    x_gen_img = x_gen_img[:n]
    x_gen_img = x_gen_img.reshape(
        15, -1, 28, 28).transpose(1, 2, 0, 3).reshape(-1, 15 * 28)
    imsave('%s.png' % fn, x_gen_img)

def plot_z_space(gen, granularity=.125, z_dim = 2):
    num_samples = int(width*2/granularity)
    x = np.arange(-width,width, granularity)

    zs = np.meshgrid(*([x]*z_dim))
    #zs = list(product(np.linspace(-width,width,num_samples), 
    #                  np.linspace(-width,width,num_samples)))
    zs = np.array(zs).astype('float32')
    zs_gpu = cuda.to_gpu(zs.reshape(num_samples**z_dim,z_dim,1,1), device=0)
    z = Variable(zs_gpu)
    y = gen(z)
    print(y.shape)
    return y


def main():


    parser = argparse.ArgumentParser(description='GAN_MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=200,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--z_dim', '-z', default=2, type=int,
                        help='Dimension of random variable')
    parser.add_argument('--loops', '-l', default=5, type=int,
                        help='number of models to train and eval')
    args = parser.parse_args()
    for i in range(args.loops):

        scale_lines()
        print('GPU: {}'.format(args.gpu))
        print('# z_dim: {}'.format(args.z_dim))
        print('# Minibatch-size: {}'.format(args.batchsize))
        print('# epoch: {}'.format(args.epoch))
        print('')

        gen = Generator(args.z_dim)
        dis = Discriminator()
        gen.to_gpu()
        dis.to_gpu()

        opt = {'gen': optimizers.Adam(alpha=0.001, beta1=0.5),  # alphaã®ç¬¦å·ãŒé‡è¦
            'dis': optimizers.Adam(alpha=0.00001, beta1=0.5)}
        opt['gen'].setup(gen)
        opt['dis'].setup(dis)

        # train, test = datasets.get_mnist(withlabel=True, ndim=3)
        # idx = np.where(train._datasets[1] == 8)
        #train_zeros = train._datasets[0][idx[0][:500]]
        # train_zeros = train._datasets[0][idx[:int(idx[0].shape[0])]]
    #    train_zeros = train._datasets[0]
        train_zeros = np.load("resized.npy").reshape(-1, 1, 28, 28)
        train_iter = iterators.SerialIterator(train_zeros, batch_size=args.batchsize)

        updater = GAN_Updater(train_iter, gen, dis, opt,
                            device=args.gpu, z_dim=args.z_dim)
        trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

        trainer.extend(extensions.dump_graph('loss'))
        trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
        trainer.extend(extensions.snapshot_object(
            gen, 'gen_iter_{.updater.iteration}'))
        trainer.extend(extensions.snapshot_object(
            dis, 'dis_iter_{.updater.iteration}'))
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(
            ['epoch', 'loss', 'loss_gen', 'loss_data']))
        trainer.extend(extensions.ProgressBar(update_interval=100))
        trainer.extend(plot_z)


        if args.resume:
            # Resume from a snapshot
            chainer.serializers.load_npz(args.resume, trainer)

        trainer.run()

        np.save('x_gen.npy', cuda.to_cpu(x_gen.data))
        save_x(x_gen, "x_gen%d" % i)

        y = plot_z_space(gen, granularity=.1, z_dim = args.z_dim)
        np.save('y_gen.npy', cuda.to_cpu(y.data))

        save_x(y, "y_gen%d" % i)
        histogram(i, "x_gen")
        histogram(i, "y_gen")




if __name__ == '__main__':
        main()
