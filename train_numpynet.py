from tensorflow.keras.datasets import mnist
from utils import ResultRecorder
from numpy_models import *
import torch.utils.data as data
import numpy as np
import pickle
import os


np.random.seed(2020)


class MyDataSet(data.Dataset):
    def __init__(self, x, y):
        super(MyDataSet, self).__init__()
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def test(target, pred):
    cnt = 0
    for i in range(pred.shape[0]):
        if pred[i] == target[i]:
            cnt += 1
    return cnt / pred.shape[0]


((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

split = int(trainData.shape[0] * 0.1)
validData = trainData[:split, ...]
validLabels = trainLabels[:split, ...]
trainData = trainData[split:, ...]
trainLabels = trainLabels[split:, ...]

validData = np.expand_dims(validData, 3)
validData = np.pad(validData, ((0, 0), (2, 2), (2, 2), (0, 0))).astype('float32') / 255.

testData = np.expand_dims(testData, 3)
testData = np.pad(testData, ((0, 0), (2, 2), (2, 2), (0, 0))).astype('float32') / 255.

dataset = MyDataSet(trainData, trainLabels)
dataLoader = data.DataLoader(dataset=dataset, batch_size=128, shuffle=True)

loss_func = nnet.CrossEntropyLoss()
EPOCH = 10
learning_rate = 0.001
batch_size = 128

model_name = 'BatchNormLeNet5.pkl'
model_save_path = os.path.join('outputs', 'numpy_out', model_name)
if os.path.isfile(model_save_path):
    print('Keep training from scratch.')
    with open(model_save_path, 'rb') as f:
        model = pickle.load(f)
else:
    print('No scratch found. Train from beginning.')
    model = LetNet5BatchNorm()

dict = vars(model)
print(dict)

# log_name = 'BatchNormLeNet5.rec'
# log_save_path = os.path.join('outputs', 'numpy_out', log_name)
#
# if os.path.isfile(log_save_path):
#     print('Keep recording result ...')
#     recorder = ResultRecorder.load(log_save_path)
#     n_iter = recorder.iteration
#     last_epoch = recorder.epoch
# else:
#     print("Create a new recorder.")
#     recorder = ResultRecorder()
#     n_iter = 0
#     last_epoch = 0
#
# loss = None
# inner_size = len(dataLoader)
# for epoch in range(EPOCH):
#     print('Epoch %s' % (epoch + 1))
#
#     # model.eval()
#     print('eval begin...')
#
#     # validation
#     val_out = model(validData)
#     acc_val = test(validLabels, np.argmax(val_out, axis=1))
#
#     # testing
#     test_out = model(testData)
#     acc_test = test(testLabels, np.argmax(test_out, axis=1))
#
#     recorder.addTestAcc(acc_test, n_iter)
#     recorder.addValidateAcc(acc_val, n_iter)
#     print('val acc %s test acc %s' % (acc_val, acc_test))
#
#     # model.train()
#     losses = 0
#     acces = 0
#     for i, (x, y) in enumerate(dataLoader):
#         img = x.data.numpy().astype('float32') / 255.
#         labels = y.data.numpy()
#         img = np.expand_dims(img, 3)
#
#         img = np.pad(img, ((0, 0), (2, 2), (2, 2), (0, 0))).astype('float32')
#         out = model(img)
#
#         acc_train = test(labels, np.argmax(out, axis=1))
#
#         loss = loss_func(out, labels)
#         eta = loss_func.backward()
#         model.backward(eta)
#         model.step(learning_rate)
#         n_iter += 1
#         losses += loss
#         acces += acc_train
#
#         recorder.addTrainAcc(acc_train, n_iter)
#         recorder.addTrainLoss(loss, n_iter)
#         print(i, loss)
#         print('acc: %s' % test(labels, np.argmax(out, axis=1)))
#
#     losses /= inner_size
#     acces /= inner_size
#     recorder.addAverageTrainLoss(losses, n_iter)
#     recorder.addAverageTrainAcc(acces, n_iter)
#     print('Epoch %s loss %s' % (epoch + 1, losses))
#
#
# with open(model_save_path, 'wb') as f:
#     pickle.dump(model, f)
#     print('model save successfully.')
#
# # model.eval()
# print('eval begin...')
#
# val_out = model(validData)
# acc_val = test(validLabels, np.argmax(val_out, axis=1))
# test_out = model(testData)
# acc_test = test(testLabels, np.argmax(test_out, axis=1))
#
# recorder.addTestAcc(acc_test, n_iter)
# recorder.addValidateAcc(acc_val, n_iter)
# print('val acc %s test acc %s' % (acc_val, acc_test))
#
# # Save logs
# recorder.epoch = last_epoch + EPOCH
# recorder.save(log_save_path)
# recorder.show_result()
