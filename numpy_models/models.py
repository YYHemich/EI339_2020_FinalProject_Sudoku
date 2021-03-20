import numpy as np
import numpynet as nnet


class LetNet5:
    def __init__(self, debug=False, weight=0.001):
        self.c1 = nnet.Conv2D(kernel_size=5, in_channel=1, out_channel=6, weight_sacle=weight)
        self.s2 = nnet.Pool2D(size=2, channels=6, weight_sacle=weight)
        self.sigmoid1 = nnet.Tanh()
        self.c3 = nnet.Conv2D(5, 6, 16, weight_sacle=weight)
        self.s4 = nnet.Pool2D(size=2, channels=16, weight_sacle=weight)
        self.sigmoid2 = nnet.Tanh()
        self.c5 = nnet.Conv2D(kernel_size=5, in_channel=16, out_channel=120, weight_sacle=weight)
        self.f6 = nnet.FCLayer(in_features=120, out_features=84, weight_sacle=weight)
        self.sigmoid3 = nnet.Tanh()
        self.f7 = nnet.FCLayer(in_features=84, out_features=10, weight_sacle=weight)
        # self.sigmoid4 = nnet.Tanh()

        self.debug = debug

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):  # inputs: [batchsize, 32, 32, 1]
        x = self.c1(x)
        if self.debug: print('c1 outputs', x)
        x = self.s2(x)
        if self.debug: print('s2 outputs', x)
        x = self.sigmoid1(x)
        if self.debug:print('sig1 outputs', x)
        x = self.c3(x)
        if self.debug:print('c3 outputs', x)
        x = self.s4(x)
        if self.debug:print('s4 outputs', x)
        x = self.sigmoid2(x)
        if self.debug:print('sig2 outputs', x)
        x = self.c5(x)
        if self.debug:print('c5 outputs', x)
        x = x.squeeze()
        x = self.f6(x)
        if self.debug:print('f6 outputs', x)
        x = self.sigmoid3(x)
        if self.debug: print('sig3 outputs', x)
        x = self.f7(x)
        if self.debug:print('f7 outputs', x)
        # x = self.sigmoid4(x)
        # if self.debug: print('sig4 outputs', x)
        return x

    def backward(self, eta):
        # if self.debug:print('eta input', eta)
        # eta = self.sigmoid4.backward(eta)
        if self.debug: print('eta sig4', eta)
        eta = self.f7.backward(eta)
        if self.debug:print('eta f7', eta)
        eta = self.sigmoid3.backward(eta)
        if self.debug: print('eta sig3', eta)
        eta = self.f6.backward(eta)
        if self.debug:print('eta f6', eta)
        eta = np.expand_dims(eta, 1)
        eta = np.expand_dims(eta, 1)
        if self.debug:print('eta expand', eta)
        eta = self.c5.backward(eta)
        if self.debug:print('eta c5', eta)
        eta = self.sigmoid2(eta)
        if self.debug: print('eta sig2', eta)
        eta = self.s4.backward(eta)
        if self.debug: print('eta s4', eta)
        eta = self.c3.backward(eta)
        if self.debug: print('eta c3', eta)
        eta = self.sigmoid1.backward(eta)
        if self.debug: print('eta sig1', eta)
        eta = self.s2.backward(eta)
        if self.debug: print('eta s2', eta)
        eta = self.c1.backward(eta)
        if self.debug: print('eta c1', eta)
        return eta

    def step(self, lr):
        self.c1.step(lr)
        self.s2.step(lr)
        self.c3.step(lr)
        self.s4.step(lr)
        self.c5.step(lr)
        self.f6.step(lr)
        self.f7.step(lr)


class LetNet5BatchNorm:
    def __init__(self, debug=False):
        self.c1 = nnet.Conv2D(kernel_size=5, in_channel=1, out_channel=6)
        self.s2 = nnet.Pool2D(size=2, channels=6)
        self.bn1 = nnet.BatchNorm2D(channels=6)
        self.sigmoid1 = nnet.Tanh()

        self.c3 = nnet.Conv2D(5, 6, 16)
        self.s4 = nnet.Pool2D(size=2, channels=16)
        self.bn2 = nnet.BatchNorm2D(channels=16)
        self.sigmoid2 = nnet.Tanh()

        self.c5 = nnet.Conv2D(kernel_size=5, in_channel=16, out_channel=120)
        self.bn3 = nnet.BatchNorm2D(120)

        self.f6 = nnet.FCLayer(in_features=120, out_features=84)
        self.sigmoid3 = nnet.Tanh()
        self.f7 = nnet.FCLayer(in_features=84, out_features=10)
        # self.sigmoid4 = nnet.Tanh()

        self.debug = debug

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):  # inputs: [batchsize, 32, 32, 1]
        x = self.c1(x)
        if self.debug: print('c1 outputs', x)
        x = self.s2(x)
        if self.debug: print('s2 outputs', x)
        x = self.bn1(x)
        x = self.sigmoid1(x)
        if self.debug:print('sig1 outputs', x)
        x = self.c3(x)
        if self.debug:print('c3 outputs', x)
        x = self.s4(x)
        x = self.bn2(x)
        if self.debug:print('s4 outputs', x)
        x = self.sigmoid2(x)
        if self.debug:print('sig2 outputs', x)
        x = self.c5(x)
        if self.debug:print('c5 outputs', x)
        x = self.bn3(x)
        x = x.squeeze()
        x = self.f6(x)
        if self.debug:print('f6 outputs', x)
        x = self.sigmoid3(x)
        if self.debug: print('sig3 outputs', x)
        x = self.f7(x)
        if self.debug:print('f7 outputs', x)
        return x

    def backward(self, eta):
        if self.debug: print('eta sig4', eta)
        eta = self.f7.backward(eta)
        if self.debug:print('eta f7', eta)
        eta = self.sigmoid3.backward(eta)
        if self.debug: print('eta sig3', eta)
        eta = self.f6.backward(eta)
        if self.debug:print('eta f6', eta)
        eta = np.expand_dims(eta, 1)
        eta = np.expand_dims(eta, 1)
        eta = self.bn3.backward(eta)
        if self.debug:print('eta expand', eta)
        eta = self.c5.backward(eta)
        if self.debug:print('eta c5', eta)
        eta = self.sigmoid2(eta)
        if self.debug: print('eta sig2', eta)
        eta = self.bn2.backward(eta)
        eta = self.s4.backward(eta)
        if self.debug: print('eta s4', eta)
        eta = self.c3.backward(eta)
        if self.debug: print('eta c3', eta)
        eta = self.sigmoid1.backward(eta)
        if self.debug: print('eta sig1', eta)
        eta = self.bn1.backward(eta)
        eta = self.s2.backward(eta)
        if self.debug: print('eta s2', eta)
        eta = self.c1.backward(eta)
        if self.debug: print('eta c1', eta)
        return eta

    def step(self, lr):
        self.c1.step(lr)
        self.s2.step(lr)
        self.bn1.step(lr)
        self.c3.step(lr)
        self.s4.step(lr)
        self.bn2.step(lr)
        self.c5.step(lr)
        self.bn3.step(lr)
        self.f6.step(lr)
        self.f7.step(lr)

    def train(self):
        self.bn1.setMode(True)
        self.bn2.setMode(True)
        self.bn3.setMode(True)

    def eval(self):
        self.bn1.setMode(False)
        self.bn2.setMode(False)
        self.bn3.setMode(False)


class SudokuNet:
    def __init__(self):
        self.c1 = nnet.Conv2D(kernel_size=5, in_channel=1, out_channel=32, method='SAME')
        self.a1 = nnet.ReLU()
        self.s1 = nnet.MaxPool2D(size=2, channels=32)

        self.c2 = nnet.Conv2D(kernel_size=3, in_channel=32, out_channel=32, method='SAME')
        self.a2 = nnet.ReLU()
        self.s2 = nnet.MaxPool2D(size=2, channels=32)

        self.c3 = nnet.Conv2D(kernel_size=3, in_channel=32, out_channel=64, method='VALID')
        self.a3 = nnet.ReLU()
        self.s3 = nnet.MaxPool2D(size=5, channels=64)

        self.f1 = nnet.FCLayer(in_features=64, out_features=128)
        self.a4 = nnet.ReLU()

        self.f2 = nnet.FCLayer(in_features=128, out_features=10)
        self.a5 = nnet.ReLU()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.c1(x)
        x = self.a1(x)
        x = self.s1(x)

        x = self.c2(x)
        x = self.a2(x)
        x = self.s2(x)

        x = self.c3(x)
        x = self.a3(x)
        x = self.s3(x)

        x = x.squeeze()

        x = self.f1(x)
        x = self.a4(x)

        x = self.f2(x)
        x = self.a5(x)

        return x

    def backward(self, eta):
        eta = self.a5.backward(eta)
        eta = self.f2.backward(eta)
        eta = self.a4.backward(eta)
        eta = self.f1.backward(eta)

        eta = np.expand_dims(eta, 1)
        eta = np.expand_dims(eta, 1)

        eta = self.s3.backward(eta)
        eta = self.a3.backward(eta)
        eta = self.c3.backward(eta)

        eta = self.s2.backward(eta)
        eta = self.a2.backward(eta)
        eta = self.c2.backward(eta)

        eta = self.s1.backward(eta)
        eta = self.a1.backward(eta)
        eta = self.c1.backward(eta)

        return eta

    def step(self, lr):
        self.c1.step(lr)
        self.c2.step(lr)
        self.c3.step(lr)
        self.f1.step(lr)
        self.f2.step(lr)


class BaseNet:
    def __init__(self):
        self.conv1 = nnet.Conv2D(kernel_size=5, in_channel=1, out_channel=16, stride=1, method='SAME')
        self.bn1 = nnet.BatchNorm2D(channels=16)
        self.ac1 = nnet.ReLU()
        self.pool1 = nnet.MaxPool2D(size=2, channels=16)

        self.conv2 = nnet.Conv2D(kernel_size=5, in_channel=16, out_channel=32, stride=1, method='SAME')
        self.bn2 = nnet.BatchNorm2D(channels=32)
        self.ac2 = nnet.ReLU()
        self.pool2 = nnet.MaxPool2D(size=2, channels=32)

        self.fc = nnet.FCLayer(in_features=7*7*32, out_features=10, weight_sacle=0.001)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ac1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ac2(x)
        x = self.pool2(x)

        x = x.reshape((x.shape[0], -1))
        x = self.fc(x)

        return x

    def backward(self, eta):
        eta = self.fc.backward(eta)

        eta = eta.reshape((-1, 7, 7, 32))

        eta = self.pool2.backward(eta)
        eta = self.ac2.backward(eta)
        eta = self.bn2.backward(eta)
        eta = self.conv2.backward(eta)

        eta = self.pool1.backward(eta)
        eta = self.ac1.backward(eta)
        eta = self.bn1.backward(eta)
        eta = self.conv1.backward(eta)

        return eta

    def step(self, lr):
        self.conv1.step(lr)
        self.conv2.step(lr)

        self.bn1.step(lr)
        self.bn2.step(lr)

        self.fc.step(lr)

    def train(self):
        self.bn1.setMode(True)
        self.bn2.setMode(True)

    def eavl(self):
        self.bn1.setMode(False)
        self.bn2.setMode(False)


if __name__ == "__main__":
    img = np.random.random((256, 32, 32, 1))
    print('batch input', img.shape)
    labels = np.random.randint(0, 10, (256,))

