# -*- coding: utf-8 -*-
# @Time    : 2024/3/9 10:59
# @Author  : 张壹
# @File    :os打开文件.py
# @Software: PyCharm
# 1.使用tensorflow1.0对data3数据集，完成分类
# (1)数据处理
# ①导入相关的库，并设置随机种子数（6分）
import flask
app=flask.Flask(__name__)
@app.route("/")
def cc():
    import numpy as np
    import tensorflow as tf
    import os
    import matplotlib.pyplot as plt
    import sklearn.model_selection as ms
    tf.set_random_seed(888)
    # ②正确加载data3数据集并进行归一化（6分）
    data_path= 'data3/'
    X_data=[]
    y_data=[]
    for file_name in os.listdir(data_path):
        real_file_name=data_path+file_name
        images_data=plt.imread(real_file_name)
        X_data.append(images_data/255)
        y_data.append(file_name[0])
    print(np.array(X_data).shape)
    print(np.array(y_data).shape)
    # ③将数据集乱序切分为训练集与测试集，自定义切分比例（6分）
    X_train, X_test, y_train, y_test = ms.train_test_split(X_data,y_data,test_size=0.3)
    # ④定义批量函数，用于小批量选取数据（6分）
    def next_batch(batch_size):
        global point
        batch_x=X_train[point:point+batch_size]
        batch_y=y_train[point:point+batch_size]
        point+=batch_size
        return batch_x,batch_y
    # (2)定义模型
    # ①定义占位符（6分）
    X=tf.placeholder(tf.float32,[None,32,32,3])
    Y=tf.placeholder(tf.int32,[None,])
    Y_o=tf.one_hot(Y,len(set(y_data)))
    # ②进行两层卷积，卷积核大小均为3*3，步长为1（6分）
    w1=tf.Variable(tf.random_normal([3,3,3,16]))
    l1=tf.nn.conv2d(X,w1,[1,1],'SAME')
    l1=tf.nn.relu(l1)
    l1=tf.nn.max_pool(l1,[2,2],[2,2],'SAME')
    # ③使用relu进行激活，并使用最大池化进行降采样（6分）
    w2=tf.Variable(tf.random_normal([3,3,16,64]))
    l2=tf.nn.conv2d(l1,w2,[1,1],'SAME')
    l2=tf.nn.relu(l2)
    l2=tf.nn.max_pool(l2,[2,2],[2,2],'SAME')
    # ④卷积和池化均进行0填充（6分）
    # ⑤将卷积的特征图转换为特征向量，以便于全连接（6分）
    l2_r=tf.reshape(l2,[-1,8*8*64])
    # ⑥进行全连接分类（6分）
    W=tf.Variable(tf.random_normal([8*8*64,len(set(y_data))]))
    b=tf.Variable(tf.random_normal([len(set(y_data))]))
    h=tf.matmul(l2_r,W)+b
    # ⑦计算代价函数值（8分）
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h,labels=Y_o))
    # ⑧使用adam进行梯度下降（8分）
    op=tf.train.AdamOptimizer().minimize(loss)
    pre=tf.argmax(h,-1)
    true=tf.argmax(Y,-1)
    acc=tf.reduce_mean(tf.cast(tf.equal(pre,true),tf.float32))
    # (3)模型训练
    # ①使用小批量循环进行模型训练，10个大循环，batch=15（8分）
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    # ②输出每次大循环的平均损失值和准确率（8分）
        for i in range(10):
            point=0
            avg_list=0
            acc_list=0
            for j in range(15):
                batch_x,batch_y=next_batch(int(len(X_train)/15))
                loss_,op_,acc_=sess.run([loss,op,acc],{X:batch_x,Y:batch_y})
                avg_list+=loss_/15
                acc_list+=acc_/15
            # print(i,avg_list,acc_list)
    # ③计算测试集的准确率（8分）
    #     print(sess.run(acc,{X:X_test,Y:y_test}))
        plt.imshow(X_test[-1])
        plt.show()
        saver.save(sess,'model1.ckpt')
        saver.restore(sess,'model.ckpt')
        return sess.run(acc,{X:X_test,Y:y_test})
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8081,debug=True)
