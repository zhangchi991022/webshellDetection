# coding=utf-8
import numpy as np
import tflearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


max_document_length = 1000


def show_diffrent_max_document_length_forMLP(webshell_files_list, wp_files_list):
    global max_document_length
    a = []
    b = []
    for i in range(1, 5120, 512):
        max_document_length = i
        print "max_document_length=%d" % i

        y1 = [1] * len(webshell_files_list)

        y2 = [0] * len(wp_files_list)

        x = webshell_files_list + wp_files_list

        y = y1 + y2

        vp = tflearn.data_utils.VocabularyProcessor(max_document_length=max_document_length,
                                                    min_frequency=0,
                                                    vocabulary=None,
                                                    tokenizer_fn=None)
        x = vp.fit_transform(x, unused_y=None)
        x = np.array(list(x))

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

        clf = MLPClassifier(solver='lbfgs',
                            alpha=1e-5,
                            hidden_layer_sizes=(5, 2),
                            random_state=None)
        # clf=GaussianNB()
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)
        score = metrics.accuracy_score(y_test, y_pred)
        a.append(max_document_length)
        b.append(score)
        plt.plot(a, b, 'r')
    plt.xlabel("max_document_length")
    plt.ylabel("metrics.accuracy_score")
    plt.title("metrics.accuracy_score VS max_document_length")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    print "不同的max_document_length值对MLP准确率的影响："
    black_file_list = []
    white_file_list = []

    with open('black_opcodes.txt', 'r') as f:
        for line in f:
            black_file_list.append(line.strip('\n'))

    with open('white_opcodes.txt', 'r') as f:
        for line in f:
            white_file_list.append(line.strip('\n'))

    show_diffrent_max_document_length_forMLP(black_file_list, white_file_list)
