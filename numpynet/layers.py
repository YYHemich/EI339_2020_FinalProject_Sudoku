import numpy as np
from numpynet.optimizer import AdamOptimizer


class Conv2DSlow(object):
    def __init__(self, kernel_size, in_channel, out_channel, stride=1):
        self.weights = np.random.random((kernel_size, kernel_size, in_channel, out_channel)).astype('float32') * 10
        self.bias = np.random.random((out_channel,)).astype('float32') * 10

        # self.weights = np.zeros((kernel_size, kernel_size, in_channel, out_channel), dtype='float32')
        # self.bias = np.zeros((out_channel,), dtype='float32')

        self.w_gradient = np.zeros((kernel_size, kernel_size, in_channel, out_channel), dtype='float32')
        self.b_gradient = np.zeros((out_channel,), dtype='float32')

        self.stride = stride
        self.input_channels = in_channel
        self.output_channels = out_channel
        self.ksize = kernel_size

        self.col_image = None
        self.input_shape = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):  # input: [batchsize, height, width, in_channels]
        self.input_shape = x.shape[1:]
        h, w = x.shape[1:3]
        col_weights = self.weights.reshape([-1, self.output_channels])  # [k*k*in_channel, out_channels]

        self.col_image = self.im2col(x, self.ksize, self.stride)  # [batchsize*height'*width', k_h*k_w*in_channels]
        conv_out = self.col_image @ col_weights + self.bias  # [batchsize*height'*width', out_channels] + [*, out_channels]
        conv_out = conv_out.reshape((-1, (h-self.ksize+1), (w-self.ksize+1), self.output_channels))
        return conv_out  # output: [batchsize, height', width', out_channels]

    def backward(self, eta):  # input: [batchsize, height', width', out_channels]
        # self.eta = eta
        col_eta = np.reshape(eta, [-1, self.output_channels])  # [batchsize*height'*width', out_channels]
        self.w_gradient = (self.col_image.T @ col_eta).reshape(self.weights.shape) / eta.shape[0]  # [k_h,k_w,in_channels,out_channels]
        self.b_gradient = np.sum(col_eta, axis=0) / eta.shape[0]  # [out_channels]

        # deconv of padded eta with flippd kernel to get next_eta
        pad_eta = np.pad(eta, (  # [batchsize, height'', width'', out_channels]
            (0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)),
                         'constant', constant_values=0)

        flip_weights = np.reshape(self.weights, (-1, self.input_channels, self.output_channels))  # [k_h*k_w,in_channels,out_channels]
        flip_weights = flip_weights[::-1, ...]  # reverse each convolution kernel
        flip_weights = flip_weights.swapaxes(1, 2)  # [k_h*k_w,in_channels,out_channels] -> [k_h*k_w,out_channels,in_channels]
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])  # [k_h*k_w*out_channels, in_channels]

        col_pad_eta = self.im2col(pad_eta, self.ksize, self.stride)  # [batchsize*height*width, k_h*k_w*out_channels]
        next_eta = col_pad_eta @ col_flip_weights  # [batchsize*height*width, in_channels]
        if self.input_shape:
            next_eta = np.reshape(next_eta, np.hstack(([-1], self.input_shape)))  # [batchsize, height, width, in_channels]
        else:
            raise Exception("Calling backward before forward!")
        return next_eta  # output: [batchsize, height, width, in_channels]

    def step(self, lr):
        self.weights = self.weights - self.w_gradient * lr
        self.bias = self.bias - self.b_gradient * lr

    @staticmethod
    def im2col(image, ksize, stride):
        # image is a 4d tensor ([batchsize, height, width ,channel])
        image_col = []
        for b in range(image.shape[0]):
            for i in range(0, image.shape[1] - ksize + 1, stride):
                for j in range(0, image.shape[2] - ksize + 1, stride):
                    col = image[b, i:i + ksize, j:j + ksize, :].reshape([-1])
                    image_col.append(col)
        image_col = np.array(image_col)  # [batchsize*height'*width', k*k*channel]

        return image_col


class Conv2D(object):
    def __init__(self, kernel_size, in_channel, out_channel, stride=1, weight_sacle=0.001, method='VALID'):
        # self.weights = np.random.random((kernel_size, kernel_size, in_channel, out_channel)).astype('float32') * weight_sacle
        # self.bias = np.random.random((out_channel,)).astype('float32') * weight_sacle

        self.weights = np.random.uniform(low=-weight_sacle, high=weight_sacle,
                                         size=(kernel_size, kernel_size, in_channel, out_channel)).astype('float32')
        self.bias = np.random.uniform(low=-weight_sacle, high=weight_sacle, size=(out_channel,)).astype('float32')
        # self.bias = np.zeros((out_channel,), dtype='float32')

        # self.weights = np.zeros((kernel_size, kernel_size, in_channel, out_channel), dtype='float32')
        # self.bias = np.zeros((out_channel,), dtype='float32')

        self.w_gradient = np.zeros((kernel_size, kernel_size, in_channel, out_channel), dtype='float32')
        self.b_gradient = np.zeros((out_channel,), dtype='float32')

        self.stride = stride
        self.input_channels = in_channel
        self.output_channels = out_channel
        self.ksize = kernel_size
        self.method = method

        self.optimizer = AdamOptimizer(self.weights.shape, self.bias.shape)

        self.split_stride = None
        self.input_shape = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):  # input: [batchsize, height, width, in_channels]
        self.input_shape = x.shape[1:]

        if self.method == 'SAME':
            x = np.pad(x, (
                (0, 0), (self.ksize // 2, self.ksize // 2), (self.ksize // 2, self.ksize // 2), (0, 0)),
                       'constant', constant_values=0)

        self.split_stride = self._split_by_strides(x)  # [batchsize, height', width', k_h, k_w, in_channels]
        conv_out = np.tensordot(self.split_stride, self.weights, axes=([3, 4, 5], [0, 1, 2]))  # [batchsize, height', width', out_channels]
        conv_out = conv_out + self.bias
        return conv_out  # output: [batchsize, height', width', out_channels]

    def backward(self, eta):  # input: [batchsize, height', width', out_channels]
        # self.eta = eta
        col_eta = np.reshape(eta, [-1, self.output_channels])  # [batchsize*height'*width', out_channels]
        shape = (self.split_stride.shape[0] * self.split_stride.shape[1] * self.split_stride.shape[2],
                 self.split_stride.shape[3] * self.split_stride.shape[4] * self.split_stride.shape[5])
        self.w_gradient = (self.split_stride.reshape(shape).T @ col_eta).reshape(self.weights.shape) / eta.shape[0]  # [k_h,k_w,in_channels,out_channels]
        self.b_gradient = np.sum(col_eta, axis=0) / eta.shape[0]  # [out_channels]

        if self.method == 'VALID':
            pad_eta = np.pad(eta, (
                (0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)),
                             'constant', constant_values=0)

        if self.method == 'SAME':
            pad_eta = np.pad(eta, (
                (0, 0), (self.ksize // 2, self.ksize // 2), (self.ksize // 2, self.ksize // 2), (0, 0)),
                             'constant', constant_values=0)

        flip_weights = np.reshape(self.weights,
                                  (-1, self.input_channels, self.output_channels))  # [k_h*k_w,in_channels,out_channels]
        flip_weights = flip_weights[::-1, ...]  # reverse each convolution kernel
        flip_weights = flip_weights.swapaxes(1,
                                             2)  # [k_h*k_w,in_channels,out_channels] -> [k_h*k_w,out_channels,in_channels]
        col_flip_weights = flip_weights.reshape([-1, self.ksize,
                                                 self.output_channels, self.input_channels])
        # [k_h, k_w, out_channels, in_channels]

        col_split_stride = self._split_by_strides(pad_eta)  # [batchsize, height, width, k_h, k_w, out_channels]
        # next_eta = col_pad_eta @ col_flip_weights  # [batchsize*height*width, in_channels]
        next_eta = np.tensordot(col_split_stride, col_flip_weights, axes=([3, 4, 5], [0, 1, 2]))  # [batchsize, height, width, in_channels]
        return next_eta  # output: [batchsize, height, width, in_channels]

    def step(self, lr):
        # self.weights = self.weights - self.w_gradient * lr
        # self.bias = self.bias - self.b_gradient * lr
        w_grad, b_grad = self.optimizer.step(self.w_gradient, self.b_gradient)
        self.weights = self.weights - w_grad * lr
        self.bias = self.bias - b_grad * lr

    def _split_by_strides(self, x):
        # 将数据按卷积步长划分为与卷积核相同大小的子集,当不能被步长整除时，不会发生越界，但是会有一部分信息数据不会被使用
        batchsize, height, width, channel = x.shape
        h = (height - self.ksize) // self.stride + 1
        w = (width - self.ksize) // self.stride + 1
        shape = (batchsize, h, w, self.ksize, self.ksize, channel)
        strides = (x.strides[0], x.strides[1] * self.stride, x.strides[2] * self.stride, *x.strides[1:])
        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


class Pool2D(object):
    def __init__(self, size, channels, weight_sacle=0.001):
        # self.a = np.random.random((channels,)).astype('float32') * weight_sacle
        # self.bias = np.random.random((channels,)).astype('float32') * weight_sacle

        self.a = np.random.uniform(low=-weight_sacle, high=weight_sacle, size=(channels,)).astype('float32')
        self.bias = np.random.uniform(low=-weight_sacle, high=weight_sacle, size=(channels,)).astype('float32')
        # self.bias = np.zeros((channels,), dtype='float32')

        # self.a = np.zeros((channels,), dtype='float32')
        # self.bias = np.zeros((channels,), dtype='float32')

        self.a_gradient = np.zeros((channels,), dtype='float32')
        self.b_gradient = np.zeros((channels,), dtype='float32')

        self.optimizer = AdamOptimizer(self.a.shape, self.bias.shape)

        self.k_size = size
        self.size = None
        self.input_shape = None
        self.subsample = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):  # input: [batchsize, height, width, in_channels]
        if x.shape[1] % self.k_size or x.shape[2] % self.k_size:
            raise Exception("Pooling size can not mod input shape!")
        self.size = x.shape[1:]
        subsample = np.reshape(self.sub_sample(x), (-1, self.size[-1]))  # [batchsize*height'*width', in_channels]
        out = subsample * self.a + self.bias
        return out.reshape(self.subsample.shape)  # output: [batchsize, height', width', in_channels]

    def backward(self, eta):  # input: [batchsize, height', weight', in_channels]
        self.a_gradient = np.sum((eta * self.subsample).reshape((-1, eta.shape[-1])), axis=0) / eta.shape[0]
        self.b_gradient = np.sum(eta.reshape((-1, eta.shape[-1])), axis=0) / eta.shape[0]

        upsample_eta = self.up_sample(eta)  # [batchsize, height, weight, in_channels]
        next_eta = np.reshape(upsample_eta, (-1, upsample_eta.shape[-1])) * self.a  # [batchsize*height*weight, in_channels]
        return next_eta.reshape(upsample_eta.shape)

    def step(self, lr):
        # self.a = self.a - self.a_gradient * lr
        # self.bias = self.bias - self.b_gradient * lr

        w_grad, b_grad = self.optimizer.step(self.a_gradient, self.b_gradient)
        self.a = self.a - w_grad * lr
        self.bias = self.bias - b_grad * lr

    def sub_sample(self, x):  # input: [batchsize, height, width, in_channels]
        self.subsample = np.zeros(np.hstack((x.shape[0], self.size[0] // self.k_size, self.size[1] // self.k_size, self.size[2])), dtype='float32')
        for batch in range(self.subsample.shape[0]):
            for i in range(self.subsample.shape[1]):
                for j in range(self.subsample.shape[2]):
                    for channel in range(self.subsample.shape[3]):
                        tmp = np.sum(x[batch, i*self.k_size:(i+1)*self.k_size, j*self.k_size:(j+1)*self.k_size, channel])
                        self.subsample[batch, i, j, channel] = tmp
        return self.subsample.copy()

    def up_sample(self, x):
        sample = np.zeros((x.shape[0], x.shape[1] * self.k_size, x.shape[2] * self.k_size, x.shape[3]), dtype='float32')
        for batch in range(x.shape[0]):
            for i in range(x.shape[1]):
                for j in range(x.shape[2]):
                    for channel in range(x.shape[3]):
                        sample[batch, i*self.k_size:(i+1)*self.k_size, j*self.k_size:(j+1)*self.k_size, channel] = x[batch, i, j, channel]
        return sample


class MaxPool2D(object):
    def __init__(self, size, channels, stride=1):
        self.mask = None
        self.k_size = size
        self.stride = stride
        self.channels = channels

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):  # [batchsize, height, width, channel]
        return self.subsample(x)

    def backward(self, eta):
        eta = self.up_sample(eta)
        return self.mask * eta

    def subsample(self, x):
        self.mask = np.zeros(x.shape, dtype='float32')
        subsample = np.zeros(np.hstack((x.shape[0], x.shape[1] // self.k_size, x.shape[2] // self.k_size, x.shape[3])),
                                  dtype='float32')
        for batch in range(subsample.shape[0]):
            for i in range(subsample.shape[1]):
                for j in range(subsample.shape[2]):
                    for channel in range(subsample.shape[3]):
                        tmp = np.max(
                            x[batch, i*self.k_size:(i+1)*self.k_size, j*self.k_size:(j+1)*self.k_size, channel]
                        )
                        x_index, y_index = np.where(x[batch, i * self.k_size:(i + 1) * self.k_size, j * self.k_size:(j + 1) * self.k_size,
                            channel] == tmp)
                        subsample[batch, i, j, channel] = tmp
                        self.mask[batch, i * self.k_size + x_index, j * self.k_size + y_index, channel] = 1
        return subsample

    def up_sample(self, x):
        batchsize, height, width, channel = x.shape
        shape = (batchsize, height, width, self.k_size, self.k_size, channel)
        strides = (*x.strides[:3], 0, 0, x.strides[3])
        new_shape = (batchsize, height*self.k_size, width*self.k_size, channel)
        up_sample = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides).swapaxes(2, 3).reshape(new_shape)
        return up_sample


class FCLayer(object):
    def __init__(self, in_features, out_features, weight_sacle=0.001):
        # self.weight = np.random.random((in_features, out_features)).astype('float32')
        # self.bias = np.random.random((out_features,)).astype('float32')

        # self.weight = np.random.random((in_features, out_features)).astype('float32') * weight_sacle
        # self.bias = np.random.random((out_features,)).astype('float32') * weight_sacle

        self.weight = np.random.uniform(low=-weight_sacle, high=weight_sacle, size=(in_features, out_features)).astype('float32')
        self.bias = np.random.uniform(low=-weight_sacle, high=weight_sacle, size=(out_features,)).astype('float32')
        # self.bias = np.zeros((out_features,), dtype='float32')

        self.w_gradient = np.zeros((in_features, out_features), dtype='float32')
        self.b_gradient = np.zeros((out_features,), dtype='float32')

        self.in_features = in_features
        self.out_feature = out_features

        self.optimizer = AdamOptimizer(self.weight.shape, self.bias.shape)

        self.inputs = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):  # input: [batchsize, in_features]
        self.inputs = x.copy()
        return x @ self.weight + self.bias  # output: [batchsize, out_features]

    def backward(self, eta):  # input: [batchsize, out_features]
        if self.inputs is None:
            raise Exception("Calling backward before forward!")
        self.w_gradient = self.inputs.T @ eta / eta.shape[0]
        self.b_gradient = np.sum(eta, axis=0) / eta.shape[0]

        next_eta = eta @ self.weight.T  # [batchsize, in_features]
        return next_eta

    def step(self, lr):
        # self.weight = self.weight - self.w_gradient * lr
        # self.bias = self.bias - self.b_gradient * lr

        w_grad, b_grad = self.optimizer.step(self.w_gradient, self.b_gradient)
        self.weight = self.weight - w_grad * lr
        self.bias = self.bias - b_grad * lr


class BatchNorm2D(object):
    def __init__(self, channels, momentum=0.9, eps=1e-8, mode=True, weight_scaled=1):
        self.channels = channels
        # self.gamma = np.ones((channels,), dtype='float32')
        # self.beta = np.zeros((channels,), dtype='float32')

        self.gamma = np.random.uniform(low=-weight_scaled, high=weight_scaled, size=(channels,))
        self.beta = np.random.uniform(low=-weight_scaled, high=weight_scaled, size=(channels,))

        self.momentum = momentum
        self.eps = eps

        self.w_gradient = np.zeros((channels,), dtype='float32')
        self.b_gradient = np.zeros((channels,), dtype='float32')

        self.mean_now = np.zeros((channels,), dtype='float32')
        self.var_now = np.zeros((channels,), dtype='float32')
        self.input = None
        self.x_hat = None

        self.running_mean = np.zeros((channels,), dtype='float32')
        self.running_var = np.zeros((channels,), dtype='float32')

        self.mode = mode
        self.optimizer = AdamOptimizer(self.gamma.shape, self.beta.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):  # input: [batchsize, height, width, channel]
        if self.mode:
            x_mean = np.mean(x, axis=(0, 1, 2))  # [channel,]
            x_var = np.var(x, axis=(0, 1, 2))  # [channel,]

            self.mean_now = x_mean  # [channel,]
            self.var_now = x_var + self.eps  # [channel,]
            self.input = x

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * x_mean  # [channel,]
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * x_var  # [channel,]

            x_hat = (x - x_mean) / np.sqrt(x_var + self.eps)  # [batchsize, height, width, channel]
            self.x_hat = x_hat

            out = self.gamma * x_hat + self.beta
        else:
            out = self.gamma / np.sqrt(self.running_var + self.eps) * x + self.beta \
                    - (self.gamma * self.running_mean / np.sqrt(self.running_var + self.eps))
        return out  # output: [batchsize, height, width, channel]

    def backward(self, eta):  # input [batchsize, height, width, channel]
        m = self.input.shape[0] * self.input.shape[1] * self.input.shape[2]

        self.w_gradient = np.sum(eta * self.x_hat, axis=(0, 1, 2)) #/ batchsize
        self.b_gradient = np.sum(eta, axis=(0, 1, 2)) #/ batchsize

        p_xhat = eta * self.gamma  # [batchsize, height, width, channel]
        p_var = -0.5 * p_xhat * (self.input - self.mean_now) * ((1 / np.sqrt(self.var_now)) ** 3)  # [batchsize, height, width, channel]
        p_var = np.sum(p_var.reshape((-1, self.channels)), axis=0)  # [channel,]
        p_mean1 = -p_xhat * (1 / np.sqrt(self.var_now))  # [batchsize, height, width, channel]
        p_mean2 = -2 * (self.input - self.mean_now)  # [batchsize, height, width, channel]
        p_mean = np.sum(p_mean1.reshape((-1, self.channels)), axis=0) + p_var * np.mean(p_mean2, axis=(0, 1, 2))  # [channel,]

        next_eta = p_xhat * (1 / np.sqrt(self.var_now)) + p_var * 2 * (self.input - self.mean_now) / m + p_mean / m
        return next_eta

    def step(self, lr):
        if self.mode:
            w_grad, b_grad = self.optimizer.step(self.w_gradient, self.b_gradient)
            self.gamma = self.gamma - w_grad * lr
            self.beta = self.beta - b_grad * lr

    def setMode(self, flag):
        self.mode = flag


class Tanh(object):
    def __init__(self, a=1.7159, s=1):
        self.A = a
        self.S = s
        self.inputs = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):  # input: [batchsize, height, width, channels]
        self.inputs = x
        return self.A * np.tanh(self.S * x)

    def backward(self, eta):
        if self.inputs is None:
            raise Exception("Call backward before forward!")
        return eta * self.A * (1 - np.square(np.tanh(self.S * self.inputs))) * self.S


class Sigmoid(object):
    def __init__(self):
        self.out = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, eta):
        if self.out is None:
            raise Exception("Call backward before forward!")
        return eta * self.out * (1 - self.out)


class ReLU(object):
    def __init__(self):
        self.mask = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.mask = np.zeros(x.shape, dtype='float32')
        self.mask[np.where(x > 0)] = 1
        # x = (x + abs(x)) / 2
        x[np.where(x <= 0)] = 0
        return x

    def backward(self, eta):
        if self.mask is None:
            raise Exception("Call backward before forward!")
        return eta * self.mask


class Softmax(object):
    def __init__(self):
        self.input = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):  # input: [batchsize, in_features]
        self.input = x.copy()
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
