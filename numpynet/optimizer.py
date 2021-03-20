import numpy as np


class AdamOptimizer(object):
    def __init__(self, w_shape, b_shape, beta1=0.9, beta2=0.999, epsilon=10e-6):
        self.w_momentum = np.zeros(w_shape, dtype='float32')
        self.b_momentum = np.zeros(b_shape, dtype='float32')
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.w_ms = np.zeros(w_shape, dtype='float32')
        self.b_ms = np.zeros(b_shape, dtype='float32')

    def step(self, grad_w, grad_b):
        self.w_momentum = self.beta1 * self.w_momentum + (1 - self.beta1) * grad_w
        self.b_momentum = self.beta1 * self.b_momentum + (1 - self.beta1) * grad_b

        self.w_ms = self.beta2 * self.w_ms + (1 - self.beta2) * np.square(grad_w)
        self.b_ms = self.beta2 * self.b_ms + (1 - self.beta2) * np.square(grad_b)

        w_momentum_hat = self.w_momentum / (1 - self.beta1)
        w_ms_hat = self.w_ms / (1 - self.beta2)

        b_momentum_hat = self.b_momentum / (1 - self.beta1)
        b_ms_hat = self.b_ms / (1 - self.beta2)

        grad_w_out = w_momentum_hat / (np.sqrt(w_ms_hat) + self.epsilon)
        grad_b_out = b_momentum_hat / (np.sqrt(b_ms_hat) + self.epsilon)

        return grad_w_out, grad_b_out
