
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import Link, Chain, ChainList
import chainer.links as L
import chainer.functions as F
from chainer import function



class Generator(Chain):

    def __init__(self, z_dim):
        super(Generator, self).__init__(
            # in_ch,out_ch,ksize,stride,pad
            l1=L.Deconvolution2D(z_dim, 128, 3, 2, 0),
            bn1=L.BatchNormalization(128),
            l2=L.Deconvolution2D(128, 128, 3, 2, 1),
            bn2=L.BatchNormalization(128),
            l3=L.Deconvolution2D(128, 128, 3, 2, 1),
            bn3=L.BatchNormalization(128),
            l4=L.Deconvolution2D(128, 128, 3, 2, 2),
            bn4=L.BatchNormalization(128),
            l5=L.Deconvolution2D(128, 1, 3, 2, 2, outsize=(28, 28)),
        )
        self.train = True

    def __call__(self, z):
        h = self.bn1(F.relu(self.l1(z)))
        h = self.bn2(F.relu(self.l2(h)))
        h = self.bn3(F.relu(self.l3(h)))
        h = self.bn4(F.relu(self.l4(h)))
        x = F.sigmoid(self.l5(h))
        #x = InvertLoss()(x)
        return x


class Discriminator(Chain):

    def __init__(self):
        super(Discriminator, self).__init__(
            # in_ch,out_ch,ksize,stride,pad
            l1=L.Convolution2D(None, 32, 3, 2, 1),
            bn1=L.BatchNormalization(32),
            l2=L.Convolution2D(None, 32, 3, 2, 2),
            bn2=L.BatchNormalization(32),
            l3=L.Convolution2D(None, 32, 3, 2, 1),
            bn3=L.BatchNormalization(32),
            l4=L.Convolution2D(None, 32, 3, 2, 1),
            bn4=L.BatchNormalization(32),
            l5=L.Convolution2D(None, 1, 3, 2, 1),
        )

    def __call__(self, x):
        h = self.bn1(F.leaky_relu(self.l1(x)))
        h = self.bn2(F.leaky_relu(self.l2(h)))
        h = self.bn3(F.leaky_relu(self.l3(h)))
        h = self.bn4(F.leaky_relu(self.l4(h)))
        y = self.l5(h)
        return y

class InvertLoss(function.Function):
    def __init__(self):
        pass

    def check_type_forward(self, in_types):
        pass
    #type_check.expect(in_types.size() == 1)
    #x_type, = in_types
    #type_check.expect(x_type.dtype.kind == 'f')

    def forward_cpu(self, x):
        return x

    def forward_gpu(self, x):
        return x

    def backward_cpu(self, x, gy):
        #gy is gradient
        return (1./np.exp(gy[0])),

    def backward_gpu(self, x, gy):
        return (-gy[0]),
        #return (1./cupy.exp(gy[0])),