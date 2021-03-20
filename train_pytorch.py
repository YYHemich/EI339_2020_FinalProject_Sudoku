import torch
import numpy as np
import os
from pytorch_models import *
from dataset_wrapper.dataset_wrapper import DataSetWrapper
from utils import ResultRecorder, TestEachDigit


torch.manual_seed(2020)


def test(target, pred):
    cnt = 0
    for i in range(pred.shape[0]):
        if pred[i] == target[i]:
            cnt += 1
    return cnt / pred.shape[0]


def count_true(target, pred):
    cnt = 0
    for i in range(pred.shape[0]):
        if pred[i] == target[i]:
            cnt += 1
    return cnt, pred.shape[0]


EPOCH = 50
BATCHSIZE = 256
LR = 0.002

print('Loading data...')
dataSetWrapper = DataSetWrapper('Full', validate_rate=0.1, batchsize=BATCHSIZE)
trainLoader, devLoader, testLoader = dataSetWrapper.get_datasetLoader()
print('Data prepared.')

checkpoint_name = 'checkpoint1.dic'
rec_name = 'checkpoint1.rec'
model_save_path = os.path.join('outputs', 'pytorch_out', checkpoint_name)
rec_save_path = os.path.join('outputs', 'pytorch_out', rec_name)

model = CNSudokuNet(class_num=20)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

if os.path.isfile(model_save_path):
    print('Load scratch...')
    dic = torch.load(model_save_path)
    model.load_state_dict(dic['model'])
    optimizer.load_state_dict(dic['optimizer'])
    last_epoch = dic['EPOCH']
else:
    print('No scratch found. Train a new model.')
    last_epoch = 0

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=0)
loss_func = nn.CrossEntropyLoss()

if os.path.isfile(rec_save_path):
    recorder = ResultRecorder.load(rec_save_path)
    print('Keep recording ...')
    recorder.epoch = EPOCH + last_epoch
    n_itr = recorder.n_itr
else:
    recorder = ResultRecorder()
    recorder.epoch = EPOCH
    n_itr = 0

for epoch in range(EPOCH):

    # held out set
    model.eval()
    total = 0
    correct = 0
    losses = 0
    for i, (x, y) in enumerate(devLoader):
        x = x.float() / 255.
        y = y.long()
        x = x.unsqueeze(1)
        out = model(x)
        loss = loss_func(out, y)
        losses += loss.data.numpy()
        pred = np.argmax(out.data.numpy(), axis=1).squeeze()
        cnt, length = count_true(y, pred)
        total += length
        correct += cnt
    losses /= len(devLoader)
    print('validate acc: %s' % (correct / total))

    recorder.addValidateAcc(correct / total, n_itr)
    recorder.addValidateLoss(losses, n_itr)

    # test set
    model.eval()
    total = 0
    correct = 0
    for i, (x, y) in enumerate(testLoader):
        x = x.float() / 255.
        y = y.long()
        x = x.unsqueeze(1)
        out = model(x)
        pred = np.argmax(out.data.numpy(), axis=1).squeeze()
        cnt, length = count_true(y, pred)
        total += length
        correct += cnt
    print('test acc: %s' % (correct / total))
    recorder.test_acc.add((correct / total), n_itr)

    # Train
    model.train()
    avg_acc = 0
    avg_loss = 0
    for i, (x, y) in enumerate(trainLoader):
        x = x.float() / 255.
        y = y.long()
        optimizer.zero_grad()
        x = x.unsqueeze(1)
        out = model(x)
        loss = loss_func(out, y)
        loss.backward()
        optimizer.step()
        n_itr += 1

        recorder.addTrainLoss(loss.data.numpy(), n_itr)
        avg_loss += loss.data.numpy()

        pred = np.argmax(out.data.numpy(), axis=1).squeeze()
        # print(pred.shape)
        # exit(0)
        cnt, length = count_true(y, pred)
        acc = cnt / length

        print('%s loss %s train acc %s' % (n_itr, loss.data.numpy(), acc))
        recorder.addTrainAcc(acc, n_itr)
        avg_acc += acc

    recorder.addAverageTrainAcc(avg_acc / len(trainLoader), n_itr)
    recorder.addAverageTrainLoss(avg_loss / len(trainLoader), n_itr)
    scheduler.step()

state_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'EPOCH': EPOCH + last_epoch}
torch.save(state_dict, model_save_path)
# model.load_state_dict(torch.load('output/baseline2.pth'))

# Final Test
# held out set
model.eval()
total = 0
correct = 0
losses = 0
for i, (x, y) in enumerate(devLoader):
    x = x.float() / 255.
    y = y.long()
    x = x.unsqueeze(1)
    out = model(x)
    loss = loss_func(out, y)
    losses += loss.data.numpy()
    pred = np.argmax(out.data.numpy(), axis=1).squeeze()
    cnt, length = count_true(y, pred)
    total += length
    correct += cnt
losses /= len(devLoader)
print('validate acc: %s' % (correct / total))

recorder.addValidateAcc(correct / total, n_itr)
recorder.addValidateLoss(losses, n_itr)

# test set
model.eval()

digit_calc = TestEachDigit(class_num=20)

total = 0
correct = 0
for i, (x, y) in enumerate(testLoader):
    x = x.float() / 255.
    y = y.long()
    x = x.unsqueeze(1)
    out = model(x)
    pred = np.argmax(out.data.numpy(), axis=1).squeeze()
    cnt, length = count_true(y, pred)
    digit_calc.add(y.data.numpy().squeeze(), pred)
    total += length
    correct += cnt
print('test acc: %s' % (correct / total))
recorder.test_acc.add((correct / total), n_itr)

recorder.show_result()
recorder.iteration = n_itr
recorder.save(rec_save_path)

digit_calc.draw()
