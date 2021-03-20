import torch.utils.data as data
import os
import numpy as np


np.random.seed(2020)


class MyDataSet(data.Dataset):
    def __init__(self, x, y):
        super(MyDataSet, self).__init__()
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class DataSetWrapper:
    def __init__(self, dataset_name, validate_rate, batchsize):
        self.batchsize = batchsize
        self.validate_rate = validate_rate
        self.CN_path = os.path.join('datasets', 'CN_data')
        self.mnist_data_path = os.path.join('datasets', 'MNIST_data', 'mnist_data.npy')
        self.mnist_label_path = os.path.join('datasets', 'MNIST_data', 'mnist_labels.npy')
        self.test_rate = 0.2
        self.name = dataset_name

    def get_datasetLoader(self):
        if self.name == 'MNIST':
            loaders = self._get_MNISTdatasetLoader()
        elif self.name == 'CN':
            loaders = self._get_CNdatasetLoader()
        elif self.name == 'Full':
            loaders = self._get_FulldatasetLoader()
        else:
            raise Exception('Undefined dataset name: %s' % self.name)
        return loaders

    def _get_MNISTdatasetLoader(self):
        mnist_data = self._get_mnist_data()
        mnist_train, mnist_test = self._data_split(mnist_data, self.test_rate, shuffle=False)
        mnist_train, mnist_validate = self._data_split(mnist_train, self.validate_rate, shuffle=False)

        total_train_x, total_train_y = self._get_xy(mnist_train)
        total_validate_x, total_validate_y = self._get_xy(mnist_validate)
        total_test_x, total_test_y = self._get_xy(mnist_test)

        trainDataSet = MyDataSet(total_train_x, total_train_y)
        validateDataSet = MyDataSet(total_validate_x, total_validate_y)
        testDataSet = MyDataSet(total_test_x, total_test_y)

        tr_loader = data.DataLoader(dataset=trainDataSet, batch_size=self.batchsize, shuffle=True)
        va_loader = data.DataLoader(dataset=validateDataSet, batch_size=self.batchsize, shuffle=True)
        te_loader = data.DataLoader(dataset=testDataSet, batch_size=self.batchsize, shuffle=True)

        return tr_loader, va_loader, te_loader

    def _get_CNdatasetLoader(self):
        CN_train = self._get_CN_data('training')
        CN_train, CN_validate = self._data_split(CN_train, self.validate_rate, shuffle=True)
        CN_test = self._get_CN_data('testing')
        np.random.shuffle(CN_test)

        total_train_x, total_train_y = self._get_xy(CN_train)
        total_validate_x, total_validate_y = self._get_xy(CN_validate)
        total_test_x, total_test_y = self._get_xy(CN_test)

        trainDataSet = MyDataSet(total_train_x, total_train_y)
        validateDataSet = MyDataSet(total_validate_x, total_validate_y)
        testDataSet = MyDataSet(total_test_x, total_test_y)

        tr_loader = data.DataLoader(dataset=trainDataSet, batch_size=self.batchsize, shuffle=True)
        va_loader = data.DataLoader(dataset=validateDataSet, batch_size=self.batchsize, shuffle=True)
        te_loader = data.DataLoader(dataset=testDataSet, batch_size=self.batchsize, shuffle=True)

        return tr_loader, va_loader, te_loader

    def _get_FulldatasetLoader(self):

        # process mnist data
        mnist_data = self._get_mnist_data()
        # np.random.shuffle(mnist_data)  # you can shuffle or not
        mnist_train, mnist_test = self._data_split(mnist_data, self.test_rate, shuffle=False)
        mnist_train, mnist_validate = self._data_split(mnist_train, self.validate_rate, shuffle=False)

        # process CN data
        CN_train = self._get_CN_data('training', label_start=10)
        CN_train, CN_validate = self._data_split(CN_train, self.validate_rate, shuffle=True)

        CN_test = self._get_CN_data('testing', label_start=10)
        np.random.shuffle(CN_test)

        total_train = CN_train + mnist_train
        total_validate = CN_validate + mnist_validate
        total_test = CN_test + mnist_test

        np.random.shuffle(total_train)
        np.random.shuffle(total_validate)
        np.random.shuffle(total_test)

        total_train_x, total_train_y = self._get_xy(total_train)
        total_validate_x, total_validate_y = self._get_xy(total_validate)
        total_test_x, total_test_y = self._get_xy(total_test)

        trainDataSet = MyDataSet(total_train_x, total_train_y)
        validateDataSet = MyDataSet(total_validate_x, total_validate_y)
        testDataSet = MyDataSet(total_test_x, total_test_y)

        tr_loader = data.DataLoader(dataset=trainDataSet, batch_size=self.batchsize, shuffle=True)
        va_loader = data.DataLoader(dataset=validateDataSet, batch_size=self.batchsize, shuffle=True)
        te_loader = data.DataLoader(dataset=testDataSet, batch_size=self.batchsize, shuffle=True)

        return tr_loader, va_loader, te_loader

    def _get_xy(self, all_data):
        x = [arr[0] for arr in all_data]
        y = [arr[1] for arr in all_data]

        return np.array(x), np.array(y)

    def _get_CN_data(self, usage, label_start=0):
        data_li = []
        label_li = []
        for number in range(1, 11):
            folder = os.path.join(self.CN_path, str(number), usage)
            tmp_data = []
            for root_name, dir_li, file_li in os.walk(folder):
                if dir_li:
                    continue
                for file_name in file_li:
                    digits = np.load(os.path.join(root_name, file_name))
                    tmp_data.append(digits)
            label_li += [number % 10 + label_start] * len(tmp_data)  # CN digits has label from 10 to 19
            data_li = data_li + tmp_data
        return list(zip(data_li, label_li))

    def _get_mnist_data(self):
        x = np.load(self.mnist_data_path)
        y = np.load(self.mnist_label_path)

        data_label = []
        for i in range(x.shape[0]):
            data_label.append((x[i], y[i]))
        return data_label

    @staticmethod
    def _data_split(data, split_rate, shuffle=False):
        if shuffle:
            np.random.shuffle(data)
        split = int(len(data) * split_rate)
        held_out = data[:split]
        train = data[split:]
        return train, held_out


if __name__ == '__main__':
    import cv2
    dataSetWrapper = DataSetWrapper(dataset_name='Full', validate_rate=0.1, batchsize=128)
    dataSetWrapper.CN_path = os.path.join('..', 'datasets', 'CN_data')
    dataSetWrapper.mnist_data_path = os.path.join('..', 'datasets', 'MNIST_data', 'mnist_data.npy')
    dataSetWrapper.mnist_label_path = os.path.join('..', 'datasets', 'MNIST_data', 'mnist_labels.npy')
    trainLoader, devLoader, testLoader = dataSetWrapper.get_datasetLoader()

    print(len(trainLoader), len(devLoader), len(testLoader))

    for i, (x, y) in enumerate(testLoader):
        for j, img in enumerate(x):
            img = img.data.numpy()
            print('label is %s' % y[j].data.numpy())
            cv2.imshow('digits', img)
            cv2.waitKey(0)
        break
