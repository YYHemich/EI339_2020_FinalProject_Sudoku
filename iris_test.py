import numpy as np
from matplotlib import pyplot as plt
import numpynet as nnet


def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]


np.random.seed(2020)

path = u'datasets/iris_data/iris.data'  # 数据文件路径
data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
np.random.shuffle(data) #打乱数据

x, y = np.split(data, (4,), axis=1)

x_train,x_test = np.split(x,(120,),axis=0)
y_train,y_test = np.split(y,(120,),axis=0)

print(type(x_train), x_train.shape)
print(type(y_train), y_train.shape, y_train[0])
y_train = y_train.astype('int32').reshape((-1,))
y_test = y_test.astype('int32').reshape((-1,))
# print(y_train)
# exit(0)
'''
网络模型
'''


class IrisNet:
    def __init__(self, d, h, c):
        self.fc1 = nnet.FCLayer(in_features=d, out_features=h, weight_sacle=0.001)
        self.fc2 = nnet.FCLayer(in_features=h, out_features=c, weight_sacle=0.001)
        self.ac1 = nnet.ReLU()
        self.ac2 = nnet.ReLU()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.fc2(x)
        # x = self.ac2(x)
        return x

    def backward(self, eta):
        # eta = self.ac2.backward(eta)
        eta = self.fc2.backward(eta)
        eta = self.ac1.backward(eta)
        eta = self.fc1.backward(eta)
        return eta

    def step(self, l_rate):
        self.fc1.step(l_rate)
        self.fc2.step(l_rate)


'''
输入d维向量，中间h个隐含神经元，输出c>1类单热向量编码,设置学习率、轮数
'''
d = 4
h = 30
c = 3
learn = 0.01
epochs = 5000

'''
创建模型对象并训练
'''

iris_net = IrisNet(d, h, c)
loss_func = nnet.CrossEntropyLoss()

EPOCH = 1000
lr = 0.001

loss1 = []
accuracy = []
for epoch in range(EPOCH):
    train_accuracy = 0

    out = iris_net(x_train)
    pred = np.argmax(out, axis=1)
    for j in range(pred.shape[0]):
        if pred[j] == y_train[j]:
            train_accuracy += 1
    loss = loss_func(out, y_train)
    # print(iris_net.fc1.bias)
    iris_net.backward(loss_func.backward())
    iris_net.step(lr)
    # for i in range(x_train.shape[0]):
    #     out = iris_net(x_train[i:i+1, ...])
    #     pred = np.argmax(out, axis=1)
    #     for j in range(pred.shape[0]):
    #         if pred[j] == y_train[j]:
    #             train_accuracy += 1
    #     loss = loss_func(out, y_train[i:i+1])
    #     iris_net.backward(loss_func.backward())
    #     iris_net.step(lr)

    loss1.append(float(loss))
    accuracy.append(float(train_accuracy/x_train.shape[0]))

out = iris_net(x_test)
print(out)
pred = np.argmax(out, axis=1)
print(pred)
print(y_test)
test_acc = 0
for i in range(pred.shape[0]):
    if pred[i] == y_test[i]:
        test_acc += 1
# print("train loss:",train_loss)
# print("train accuracy:",train_accuracy)
print("test loss:", test_acc / y_test.shape[0])
print("test accuracy:", test_acc / y_test.shape[0])

'''
绘制训练时，训练集测试和测试集测试的loss曲线
'''
epochs_ = range(1, EPOCH + 1)
plt.plot(loss1, 'r', label = 'Training loss')
plt.plot(accuracy, 'b', label = 'Training acc')
plt.title('Training And Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plt.clf()
#
# '''
# 绘制训练时，训练集测试和测试集测试的准确率曲线
# '''
# plt.plot(epochs_, train_accuracy, 'r', label = 'Training acc')
# plt.plot(epochs_, test_accuracy, 'b', label = 'Validation acc')
# plt.title('Training And Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
