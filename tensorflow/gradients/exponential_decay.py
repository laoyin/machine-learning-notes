"""
梯度下降时，优化学习率learning_rate

其中tensorflow train.exponential_decay 函数

使用方式如下：伪代码

                learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.learning_rate,
                                                           global_step=global_step, decay_steps=FLAGS.decay_steps,
                                                           decay_rate=FLAGS.decay_rate, staircase=True)
                optimizer = tf.train.AdamOptimizer(learning_rate)
                grads, vars = zip(*optimizer.compute_gradients(loss))
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=FLAGS.norm_ratio)
                train_op = optimizer.apply_gradients(zip(grads, vars), global_step=global_step, name="****")

"""

import tensorflow as tf
import numpy as np


x = tf.placeholder(tf.float32, shape=[None, 1], name='x')
y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
w = tf.Variable(tf.constant(0.0))

global_steps = tf.Variable(0, trainable=False)


learning_rate = tf.train.exponential_decay(0.1, global_steps, 10, 0.96, staircase=True)
# 0.96 为指数衰减， global_steps 会随着训练批次进行叠加， 当staircase为true时第三个参数为每多少次进行一次更新，
loss = tf.pow(w*x-y, 2)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_steps)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        sess.run(train_step, feed_dict={x:np.linspace(1,2,10).reshape([10,1]),
            y:np.linspace(1,2,10).reshape([10,1])})
        print(sess.run(learning_rate))
        print(sess.run(global_steps))


"""
result
0.1
1
0.1
2
0.1
3
0.1
4
0.1
5
0.1
6
0.1
7
0.1
8
0.1
9
0.096
10
0.096
11
0.096
12
0.096
13
0.096
14
0.096
15
0.096
16
0.096
17
0.096
18
0.096
19
0.09216
20
0.09216
21
0.09216
22
0.09216
23
0.09216
24
0.09216
25
0.09216
26
0.09216
27
0.09216
28
0.09216
29
0.088473596
30
0.088473596
31
0.088473596
32
0.088473596
33
0.088473596
34
0.088473596
35
0.088473596
36
0.088473596
37
0.088473596
38
0.088473596
39
0.084934644
40
0.084934644
41
0.084934644
42
0.084934644
43
0.084934644
44
0.084934644
45
0.084934644
46
0.084934644
47
0.084934644
48
0.084934644
49
0.08153726
50
0.08153726
51
0.08153726
52
0.08153726
53
0.08153726
54
0.08153726
55
0.08153726
56
0.08153726
57
0.08153726
58
0.08153726
59
0.07827577
60
0.07827577
61
0.07827577
62
0.07827577
63
0.07827577
64
0.07827577
65
0.07827577
66
0.07827577
67
0.07827577
68
0.07827577
69
0.07514474
70
0.07514474
71
0.07514474
72
0.07514474
73
0.07514474
74
0.07514474
75
0.07514474
76
0.07514474
77
0.07514474
78
0.07514474
79
0.07213895
80
0.07213895
81
0.07213895
82
0.07213895
83
0.07213895
84
0.07213895
85
0.07213895
86
0.07213895
87
0.07213895
88
0.07213895
89
0.069253385
90
0.069253385
91
0.069253385
92
0.069253385
93
0.069253385
94
0.069253385
95
0.069253385
96
0.069253385
97
0.069253385
98
0.069253385
99
0.06648325
100
"""