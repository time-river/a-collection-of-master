import numpy
import random
import logging


def default_kernel_func(x_i, x_j):
    """
    linear kernel function:
        K(x_i, x_j) = x_i * x_j
    """
    return x_i.T * x_j

class SVM:

    def __init__(self, train_data, label_data, C,
                 toler=0.0001, kernel_func=default_kernel_func,
                 max_iter=40, accuracy=0.00001):
        self.train_data = numpy.mat(train_data)
        self.label_data = numpy.mat(label_data)
        self.C = C
        self.toler = toler
        self.kernel_func = kernel_func
        self.max_iter = max_iter
        self.accuracy = accuracy

        self.alpha = numpy.mat(
            numpy.zeros((self.train_data.shape[0], 1)))
        self.b = 0

    def train_routine(self):
        max_iter = self.max_iter
        alpha = self.alpha
        train_data = self.train_data

        iter_count = 0
        alpha_pairs_changed = 0
        examine_all = True

        logging.info("----start train----")

        while ((iter_count < max_iter) and
               (alpha_pairs_changed > 0 or examine_all)):

            alpha_pairs_changed = 0

            if examine_all:
                logging.info("----examine all----")
                for i in range(train_data.shape[0]):
                    alpha_pairs_changed += self.examine_example(i)

                logging.info("----alpha_pairs_changed {}".format(alpha_pairs_changed))
                iter_count += 1
            else:
                logging.info("----not examine all----")
                non_bound_alphas_list =                     numpy.nonzero((alpha.A > 0) * (self.alpha.A < self.C))[0]  # 返回下标
                for i in non_bound_alphas_list:
                    alpha_pairs_changed += self.examine_example(i)

                logging.info("----alpha_pairs_changed {}".format(alpha_pairs_changed))
                iter_count += 1

            if examine_all:
                examine_all = False
            elif alpha_pairs_changed == 0:
                examine_all = True

        logging.info("----end train----")

        return

    def examine_example(self, i):
        alpha_i = self.alpha[i]
        error_i = self.calculate_error(i)

        if ((self.label_data[i] * error_i < -self.toler and alpha_i < self.C) or
                (self.label_data[i] * error_i > self.toler and alpha_i > 0)):

            j = self.select_alpha_j(i)
            alpha_j = self.alpha[j]
            error_j = self.calculate_error(j)

            old_alpha_i = alpha_i.copy()
            old_alpha_j = alpha_j.copy()

            if self.label_data[i] != self.label_data[j]:
                L = max(numpy.mat(0), alpha_j - alpha_i)
                H = min(numpy.mat(self.C), self.C + alpha_j - alpha_i)
            else:
                L = max(numpy.mat(0), alpha_j + alpha_i - self.C)
                H = min(numpy.mat(self.C), alpha_j + alpha_i)

            if L == H:
                return 0

            k11 = self.kernel_func(self.train_data[i, :].T, self.train_data[i, :].T)
            k12 = self.kernel_func(self.train_data[i, :].T, self.train_data[j, :].T)
            k22 = self.kernel_func(self.train_data[j, :].T, self.train_data[j, :].T)
            eta = k11 + k22 - 2.0 * k12

            if eta > 0:
                alpha_j = old_alpha_j +                           self.label_data[j] * (error_i - error_j) / eta
                alpha_j = self.clip_alpha(alpha_j, L, H)
            else:
                # TODO
                return 0

            if abs(alpha_j - old_alpha_j) < self.accuracy:
                return 0

            alpha_i = old_alpha_i +                       self.label_data[i] * self.label_data[j] *                       (old_alpha_j - alpha_j)

            b1 = self.b - error_i -                  self.label_data[i] * (alpha_i - old_alpha_i) *                  self.kernel_func(alpha_i, alpha_i) -                  self.label_data[j] * (alpha_j - old_alpha_j) *                  self.kernel_func(alpha_i, alpha_j)
            b2 = self.b - error_j -                  self.label_data[i] * (alpha_i - old_alpha_i) *                  self.kernel_func(alpha_i, alpha_j) -                  self.label_data[j] * (alpha_j - old_alpha_j) *                  self.kernel_func(alpha_j, alpha_j)

            if (0 < alpha_i and alpha_j < self.C):
                self.b = b1
            elif (0 < alpha_j and alpha_j < self.C):
                self.b = b2
            else:
                self.b = (b1 + b2) / 2

            self.alpha[i] = alpha_i
            self.alpha[j] = alpha_j

            return 1
        else:
            return 0

    def f_xi(self, x):
        val = float(
                numpy.multiply(self.alpha, self.label_data).T *                 self.kernel_func(self.train_data.T, x.T) +                 self.b
            )
        return val

    def calculate_error(self, i):
        f_xi = self.f_xi(self.train_data[i, :])
        return f_xi - self.label_data[i]

    def select_alpha_j(self, i):
        j = i
        while j == i:
            j = int(
                random.uniform(0, self.train_data.shape[0])
            )
        return j

    def clip_alpha(self, alpha, L, H):
        if alpha > H:
            return H
        elif alpha < L:
            return L
        else:
            return alpha

    def test_sample(self, input_data, input_label):
        input_data = numpy.mat(input_data)
        input_label = numpy.mat(input_label)

        match_count = 0
        for i in range(input_data.shape[0]):
            predict = self.f_xi(input_data[i, :])
            if numpy.sign(predict) == numpy.sign(input_label[i]):
                match_count += 1
        print("accuracy: {}".format(match_count/input_data.shape[0]))

def load_dataset(filename):
    data = []
    label = []
    with open(filename) as f:
        for line in f.readlines():
            line_attr = line.strip().split('\t')
            data.append([float(x) for x in line_attr[:-1]])
            label.append([float(line_attr[-1])])
    return data, label

if __name__ == '__main__':
    data, label = load_dataset("testSet.txt")
    svm = SVM(data, label, 40)
    svm.train_routine()
    test_data, test_label = load_dataset("test_set.txt")
    svm.test_sample(test_data, test_label)

