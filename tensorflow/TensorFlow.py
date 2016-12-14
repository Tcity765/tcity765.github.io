import tensorflow as tf
#tensorflow를 사용하기 위해 import를 함.
from tensorflow.examples.tutorials.mnist import input_data

# Dataset loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)
#MNIST 데이터를 자동으로 다운로드하고 설치하는 코드를 포함한다.
# Set up model
x = tf.placeholder(tf.float32, [None, 784])
#x: 우리가 TensorFlow에게 계산을 하도록 명령할 때 입력할 값.
#none, 784: 각 이미지들은 784차원의 벡터->2차원 텐서로 표현함.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#variable의 초기값을 만듬.
y = tf.nn.softmax(tf.matmul(x, W) + b)
#위에 만든 모델을 구현.

y_ = tf.placeholder(tf.float32, [None, 10])
#새로운 placeholder 추가(교차 엔트로피를 구현하기 위해서)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#교차 엔트로피를 구현할 수 있습니다.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Session
init = tf.initialize_all_variables()
#우리가 만든 변수들을 초기화하는 작업.

sess = tf.Session()
sess.run(init)
#세션에서 모델을 시작하고 변수들을 초기화하는 작업.

# Learning
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#학습시킴(1000회.)

# Validation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#맞는 라벨을 예측했는지를 확인. tf.equal을 이용해 예측이 실제와 맞았는지 확인할 수 있음.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#얼마나 많은 미율로 밎았는지 확인함.
# Result should be approximately 91%.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#테스트 데이터를 대상으로 정확도를 확인하는 print.