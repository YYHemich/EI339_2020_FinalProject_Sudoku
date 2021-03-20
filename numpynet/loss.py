import numpy as np


class CrossEntropyLoss(object):  # softmax + cross_entropy_loss
    def __init__(self):
        self.softmax_out = None
        self.label = None

    def __call__(self, x, labels):
        return self.loss(x, labels)

    def loss(self, x, label):  # input: x:[batchsize, num_class], label:[batchsize, ]
        if not self._dim_check(x, label):
            raise Exception("Input dimension does not match!")
        max = np.max(x, axis=1, keepdims=True)
        self.softmax_out = np.exp(x - max) / np.sum(np.exp(x - max), axis=1, keepdims=True)  # [batchsize, num_class]
        self.label = label.copy()
        batchsize = x.shape[0]
        losses = 0
        for bs in range(batchsize):
            losses += -np.log(self.softmax_out[bs, label[bs]] + 1e-8)
        return losses / batchsize

    def backward(self):
        if self.softmax_out is None:
            raise Exception("Call backward before forward!")
        for i in range(self.softmax_out.shape[0]):
            self.softmax_out[i, self.label[i]] -= 1
        # print('checkpoint of loss func', self.softmax_out)
        return self.softmax_out.copy()

    def _dim_check(self, x, label):
        return (x.shape[0] == label.shape[0]) and (len(label.shape) == 1)
