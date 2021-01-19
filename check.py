# coding:utf-8

import numpy as np
import tflearn
from sklearn.externals import joblib
from utils import load_php_opcode, recursion_load_php_file_opcode

max_document_length = 1000

if __name__ == '__main__':
    php_file_name = "C:\\Users\\FireFly\\Desktop\\FinalTest\\webshell\\webshell\\PHP\\tanjiti\\dama\\chat.php"
    print 'Checking the file {}'.format(php_file_name)

    # 之前的数据
    white_file_list = []
    black_file_list = []

    with open('black_opcodes.txt', 'r') as f:
        for line in f:
            black_file_list.append(line.strip('\n'))

    with open('white_opcodes.txt', 'r') as f:
        for line in f:
            white_file_list.append(line.strip('\n'))

    all_token = white_file_list + black_file_list

    # 准备数据
    token = load_php_opcode(php_file_name)
    all_token.append(token)
    x = all_token

    vp = tflearn.data_utils.VocabularyProcessor(max_document_length=max_document_length,
                                                min_frequency=0,
                                                vocabulary=None,
                                                tokenizer_fn=None)
    x = vp.fit_transform(x, unused_y=None)
    x = np.array(list(x))

    # end 准备数据

    clf = joblib.load('save/mlp.pkl')
    y_p = clf.predict(x[-1:])

    if y_p == [0]:
        print 'Not Webshell'
    elif y_p == [1]:
        print 'Webshell!'
