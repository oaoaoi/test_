# -*- coding: utf-8 -*-
# @Time    : 2024/3/11 10:45
# @Author  : 张壹
# @File    :t1.py
# @Software: PyCharm
import flask
import lstm_model
from flask import request
import numpy as np
import tensorflow as tf

host = '0.0.0.0'

# 实例化 flask
app = flask.Flask(__name__)

# 加载模型
# graph=tf.reset_default_graph()

# 将预测函数定义为一个路由
@app.route("/predict")
def predict():
    with tf.Session() as sess:
        model1 = tf.train.Saver().restore(sess, 'model1.ckpt')

    # 返回Json格式的响应
        return sess.run(acc,)


# 启动Flask应用程序，允许远程连接
if __name__ == '__main__':
    app.run(debug=True, host=host, port=8888)