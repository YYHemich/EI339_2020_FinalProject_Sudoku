from matplotlib import pyplot as plt
import pickle


class PairList:
    def __init__(self):
        self.data = []
        self.x = []

    def __len__(self):
        return len(self.x)

    def __bool__(self):
        return len(self.x) != 0

    def add(self, data, x):
        self.data.append(data)
        self.x.append(x)


class ResultRecorder:
    def __init__(self):
        self.epoch = None
        self.iteration = None

        self.train_loss = PairList()
        self.train_acc = PairList()

        self.train_loss_avg = PairList()
        self.train_acc_avg = PairList()

        self.test_loss = PairList()
        self.test_acc = PairList()

        self.validate_loss = PairList()
        self.validate_acc = PairList()

    def show_result(self):
        legend = []
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        if self.train_loss:
            legend += ['train_loss']
            plt.plot(self.train_loss.x, self.train_loss.data)
        if self.train_loss_avg:
            legend += ['Average train_loss']
            plt.plot(self.train_loss_avg.x, self.train_loss_avg.data)
        if self.validate_loss:
            legend += ['validate_loss']
            plt.plot(self.validate_loss.x, self.validate_loss.data)
        if self.test_loss:
            legend += ['test_loss']
            plt.plot(self.test_loss.x, self.test_loss.data)
        plt.legend(legend)
        plt.title('Loss Trend')
        plt.ylabel('Loss')
        plt.xlabel('Number of Iteration')

        legend = []
        plt.subplot(1, 2, 2)
        if self.train_acc:
            legend += ['train_acc']
            plt.plot(self.train_acc.x, self.train_acc.data)
        if self.train_acc_avg:
            legend += ['Average train_acc']
            plt.plot(self.train_acc_avg.x, self.train_acc_avg.data)
        if self.validate_acc:
            legend += ['validate_acc']
            plt.plot(self.validate_acc.x, self.validate_acc.data)
        if self.test_acc:
            legend += ['test_acc']
            plt.plot(self.test_acc.x, self.test_acc.data)
        plt.legend(legend)
        plt.title('Accuracy Trend')
        plt.ylabel('Accuracy')
        plt.xlabel('Number of Iteration')
        plt.show()

    def save(self, pth):
        with open(pth, 'wb') as f:
            pickle.dump(self, f)

    def addTrainLoss(self, loss, global_x):
        self.train_loss.add(loss, global_x)

    def addValidateLoss(self, loss, global_x):
        self.validate_loss.add(loss, global_x)

    def addTestLoss(self, loss, global_x):
        self.test_loss.add(loss, global_x)

    def addTrainAcc(self, acc, global_x):
        self.train_acc.add(acc, global_x)

    def addValidateAcc(self, acc, global_x):
        self.validate_acc.add(acc, global_x)

    def addTestAcc(self, acc, global_x):
        self.test_acc.add(acc, global_x)

    def addAverageTrainLoss(self, loss, global_x):
        self.train_loss_avg.add(loss, global_x)

    def addAverageTrainAcc(self, acc, global_x):
        self.train_acc_avg.add(acc, global_x)

    @staticmethod
    def load(rec_path):
        with open(rec_path, 'rb') as f:
            rec = pickle.load(f)
        return rec

    @staticmethod
    def compute_avg(iteration_res, epoch):
        iteration = len(iteration_res)
        step = iteration // epoch
        res = []
        for i in range(epoch):
            res.append(sum(iteration_res[i*step: (i+1)*step]) / step)
        return list(range(1*step, len(iteration_res), step)), res

    @property
    def n_itr(self):
        if self.iteration is None:
            return len(self.train_loss)
        else:
            return self.iteration


class TestEachDigit:
    T = 0
    F = 1

    def __init__(self, class_num):
        self.memory = []
        self.class_num = class_num
        for i in range(class_num):
            self.memory.append([0, 0])

        self.tick = [str(i) for i in range(class_num)]

    def add(self, target, pred):
        for i in range(len(target)):
            if target[i] == pred[i]:
                self.memory[target[i]][self.T] += 1
            else:
                self.memory[target[i]][self.F] += 1

    def draw(self):
        data = []
        for class_num in range(self.class_num):
            if self.memory[class_num][self.T] + self.memory[class_num][self.F]:
                data += [self.memory[class_num][self.T] / (self.memory[class_num][self.T] +
                                                           self.memory[class_num][self.F])]
            else:
                data += [0]
        plt.figure()
        x = [i for i in range(self.class_num)]
        plt.bar(x=x, height=data, tick_label=self.tick)
        for a, b in zip(x, data):
            plt.text(a, b + 0.01, '%.2f' % b, ha='center', va='bottom', fontsize=5)
        plt.title('Test Accuracy of Each Class')
        plt.ylabel('Accuracy')
        plt.xlabel('Class Number')
        plt.show()


def print_progress(now_epoch, p):
    dots = '-' * p + '>' + '.' * (19 - p)
    print("\rEPOCH %s [%s]" % (now_epoch, dots), end='')


if __name__ == '__main__':
    a = PairList()
    if a:
        print('true')
    else:
        print('false')

    a.add(1, 1)
    if a:
        print('true')
    else:
        print('false')
