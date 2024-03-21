# 1.使用tensorflow2.0完成以下操作（每小题10分）
import tensorflow as tf

# (1)矩阵创建
# ①创建一个维度为3*3的全1矩阵
ones = tf.ones([3, 3])
# ②使用range，创建一个1-9的1阶张量
tf_r = tf.range(1, 10)
# ③打印上题的维度
print(tf_r.shape)

# ④将上题维度修改为3,1,3
tf_reshape = tf.reshape(tf_r, [3, 1, 3])

# ⑤使用函数，去除维度中函数1的维度
tf_squeeze = tf.squeeze(tf_reshape)
print(tf_squeeze)
# (2)切片及其他
# ①使用1-9的向量，使用切片，打印3,4,5,6  1，2，3，4，5，6，7，8，9 a[2:5]
r_ = tf_r[2:6]
print(r_)
# ②打印上题向量的均值
mean = tf.reduce_mean(r_)
print(mean)
# ③创见一个2行2列的标准正态分布矩阵
tf_normal = tf.random.normal([2, 2])
print(tf_normal)
# ④创建一个2行2列的全0矩阵
tf_zeros = tf.zeros([2, 2])
print(tf_zeros)
# ⑤将3,4问的结果拼接成一个4行2列的结果
tf_concat = tf.concat([tf_normal, tf_zeros], axis=0)
print(tf_concat)
print(tf_concat.shape)


# # 1.使用tensorflow2.0完成以下操作（每小题10分）
# import tensorflow as tf
#
# # (1)矩阵创建
# # ①创建一个维度为3*3的全1矩阵
# tf_1 = tf.ones(shape=[3, 3])
# # ②使用range，创建一个1-9的1阶张量
# tf_range = tf.range(start=1, limit=10)
# # ③打印上题的维度
# print(tf_range.shape)
# # ④将上题维度修改为3,1,3
# tf_reshape = tf.reshape(tf_range, shape=[3, 1, 3])
# # ⑤使用函数，去除维度中函数1的维度
# tf_squeeze = tf.squeeze(tf_reshape)
# # (2)切片及其他
# # ①使用1-9的向量，使用切片，打印3,4,5,6  1，2，3，4，5，6，7，8，9 a[2:5]
# tf_range_slice = tf_range[2:6]
# tf.print(tf_range_slice)
# # ②打印上题向量的均值
# mean = tf.reduce_mean(tf_range_slice)
# tf.print(mean)
# # ③创见一个2行2列的标准正态分布矩阵
# rand_norm = tf.random.normal(shape=[2, 2])
# tf.print(rand_norm)
# # ④创建一个2行2列的全0矩阵
# zero = tf.zeros(shape=[2, 2])
# tf.print(zero)
# # ⑤将3,4问的结果拼接成一个4行2列的结果
# concat = tf.concat([rand_norm, zero], axis=0)
# tf.print(concat)
# tf.print(concat.shape)