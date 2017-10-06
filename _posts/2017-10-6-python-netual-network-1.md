---
title: 如何使用9行Python代码构建一个简单的神经网络
layout: post
tags: [python,neural network]
---

   <!-- ![1-4-4XkuTZopk59wOV6E-RCg.jpeg]({{site.img_path}}/1-4-4XkuTZopk59wOV6E-RCg.jpeg) -->
   <img src="{{site.img_path}}/1-4-4XkuTZopk59wOV6E-RCg.jpeg" width="50%" style="margin: 0 auto 100px;display: block;">

作为我学习AI的一部分，我给自己定的目标是用Python构建一个简单的神经网络。为了真正理解它，我没有使用神经网络库而是从原理上直接构建了它。多亏了Andrew Trask写的一篇极好的博客我实现了我的目标。下面就是这9行代码：

```python
from numpy import exp,array,random,dot
training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
training_set_outputs = array([[0,1,1,0]])
random.seed(1)
synaptic_weeights = 2 * random.random((3,1)) - 1
for interation in xrange(1000):
    output = 1 / (1 + exp(-(dot(training_set_inputs,synaptic_weights))))
    synaptic_weights += dot(training_set_inputs.T,(training_set_outpits - output) * output * (1-output))
print 1/(1+exp(-(dot(array([1,0,0]),synaptic_weeights))))
```

在这篇文章中，我将要解释我是怎样做到的，你也可以构建你自己的神经网络，我也将提供一个更长的但更美观的一个源码版本。

但是，什么是神经网络呢？人类的大脑包含10亿个神经元细胞，突触把它们连接在一起。当有足够的突触输入神经元，那么神经元也有相应的输出，我们把这个过程称之为“思考”。

通过在计算机上创建一个神经网络我们可以对“思考”进行建模。我们没有必要建立一个分子水平的大脑模型，我们可以使用数学中矩阵的方法。为了时模型足够简单，我们将建立一个3输入1输出的模型。


   <!-- ![1-HDWhvFz5t0KAjIAIzjKR1w.png]({{site.img_path}}/1-HDWhvFz5t0KAjIAIzjKR1w.png) -->
   <img src="{{site.img_path}}/1-HDWhvFz5t0KAjIAIzjKR1w.png" width="50%">

我们将训练一个解决简单问题的神经网络，下面4个Example叫做训练集，你可以计算出它的模式么？是0还是1？

   <!-- ![1-nEooKljI8XbKQh4cFbZu1Q.png]({{site.img_path}}/1-nEooKljI8XbKQh4cFbZu1Q.png) -->
   <img src="{{site.img_path}}/1-nEooKljI8XbKQh4cFbZu1Q.png" width="50%">

也许你已经注意到，输出的值总是等于最左边输入的值，因此这个答案应该等于1.

## 训练过程

我们怎样教我们的神经网络正确的回答问题呢？我们将给每一个输入一个权重，它可以是一个整数也可以是一个负数，当一个输入有一个很大的正权重或者一个很小的负权重，它将对输出有很大的影响。在我们开始之前，我们将随机设置每一个权重，然后再开始我们的训练过程：
1.从训练集中拿一个做输入，通过权重调整他们，再通过一个特殊的公式计算得到输出。
2.计算误差，计算输出值与期望值的差。
3.根据误差的方向，稍微调整权重。
4.重复这个过程10000次。


   <!-- ![1--1trgA6DUEaafJZv3k0mGw.jpeg]({{site.img_path}}/1--1trgA6DUEaafJZv3k0mGw.jpeg) -->
   <img src="{{site.img_path}}/1--1trgA6DUEaafJZv3k0mGw.jpeg" width="50%">

最终，神经网络的权重会因为训练集得到最优解，如果我们让神经网络去“思考”满足这个模式的新的情况，那么它也可以做出很好的预测。

## 计算神经网络输出的公式

也许你正在考虑，哪一个特殊的公式可以计算神经网络的输出？首先我们把每一个权重与输入的乘积的和作为神经网络的输入，公式如下：
   <!-- ![1-RV7-CFkmmByfcXKkPcbAYQ.png]({{site.img_path}}/1-RV7-CFkmmByfcXKkPcbAYQ.png) -->
   <img src="{{site.img_path}}/1-RV7-CFkmmByfcXKkPcbAYQ.png" width="50%">

下面我们需要标准化它，以至于让结果在0-1的范围内，我们使用Sinmoid函数来实现它：

   <!-- ![1-5il5GLo0gamypklQQ_z0AA.png]({{site.img_path}}/1-5il5GLo0gamypklQQ_z0AA.png) -->
   <img src="{{site.img_path}}/1-5il5GLo0gamypklQQ_z0AA.png" width="50%">

它的函数图像是一个S型曲线，如下图所示：

   <!-- ![1-sK6hjHszCwTE8GqtKNe1Yg.png]({{site.img_path}}/1-sK6hjHszCwTE8GqtKNe1Yg.png) -->
   <img src="{{site.img_path}}/1-sK6hjHszCwTE8GqtKNe1Yg.png" width="50%">

把一个公式带入到第二个公式中，我们得到的最终神经网络输出公式是：

   <!-- ![1-7YdyG6fc6f6zMmx3l0ZGsQ.png]({{site.img_path}}/1-7YdyG6fc6f6zMmx3l0ZGsQ.png) -->
   <img src="{{site.img_path}}/1-7YdyG6fc6f6zMmx3l0ZGsQ.png" width="50%">
你可能已经注意到我们没有用最小阈值，这样做是为了让事情变得更简单。

## 调整权重的公式

在循环训练期间，我们调整权重，但是我们是怎样调整权重的呢？我们使用的是“Error Weighted Derivative”公式：

   <!-- ![1-SQBjpbBcCT3lTQlPEdr1eg.png]({{site.img_path}}/1-SQBjpbBcCT3lTQlPEdr1eg.png) -->
   <img src="{{site.img_path}}/1-SQBjpbBcCT3lTQlPEdr1eg.png" width="50%">

为什么是这个公式呢？首先我们要使调整与误差的大小成比例，第二，我们乘以0或1的输入，如果输入是0，我们便不做调整。最后，我们乘以S型曲线的梯度。为了理解最后一条，考虑一下几个方面：
1.我们使用S型曲线去计算神经网络的输出。
2.如果输出是一个非常大的整数或负数，它表明神经元是非常自信的。
3.由S型函数的曲线我们可以看出，当输出值非常大时，此时的梯度非常小。
4.如果神经元确定每一个权重都是正确的，没有必要去调整它，那么神经元可以通过乘以S型权限的梯度来实现。
S型曲线的梯度可以由下面公式计算：

   <!-- ![1-HdHm9u3_wjwBPmwuLg3D3g.png]({{site.img_path}}/1-HdHm9u3_wjwBPmwuLg3D3g.png) -->
   <img src="{{site.img_path}}/1-HdHm9u3_wjwBPmwuLg3D3g.png" width="50%">
把第二个公式带入第一个公式，我们最终可以得到调整权重值的公式：

   <!-- ![1-HdHm9u3_wjwBPmwuLg3D3g.png]({{site.img_path}}/1-HdHm9u3_wjwBPmwuLg3D3g.png) -->
   <img src="{{site.img_path}}/1-HdHm9u3_wjwBPmwuLg3D3g.png" width="50%">

这是一个可选择的公式，它可以让神经元学习更快，这个公式也很简单。

## 构建Python代码

尽管我们没有使用Python的神经网络库，我们将要导入四个方法从numpy，分别是：
1.exp 指数函数
2.array 创建一个矩阵
3.dot 矩阵相乘
4.random 产生随机数

例如我们可以使用array()的方法构建上面提到的训练集：

```python
training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
training_set_outputs = array([0,1,1,0]).T
```

.T是转置，计算机存储数据如下：

   <!-- ![1-2VAykewNiKxU-gFy3BBh_w.png]({{site.img_path}}/1-2VAykewNiKxU-gFy3BBh_w.png) -->
   <img src="{{site.img_path}}/1-2VAykewNiKxU-gFy3BBh_w.png" width="50%">

好的，我想我们已经为更美观的源码做好了准备，在我给你之前，我总结了几个最终的思想。

我对每一行代码都增加了注解，值得注意的是，每一次的迭代，我们同时处理整个训练集。因此我们的变量是个矩阵。下面是用python下的完整的代码：


```python
from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
       return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)


            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print "Random starting synaptic weights: "
    print neural_network.synaptic_weights


    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print "New synaptic weights after training: "
    print neural_network.synaptic_weights

    # Test the neural network with a new situation.
    print "Considering new situation [1, 0, 0] -> ?: "
    print neural_network.think(array([1, 0, 0]))
```

## 最终的思想

使用终端命令运行神经网络：
python main.py

你将要得到的结果是：
```(python)
Random starting synaptic weights:
[[-0.16595599]
[ 0.44064899]
[-0.99977125]]

New synaptic weights after training:
[[ 9.67299303]
[-0.2078435 ]
[-4.62963669]]

Considering new situation [1, 0, 0] -> ?:
[ 0.99993704]

```

我们做到了，我们用Python构建了一个简单的神经网络！
首先神经网络指定它的权重，然后用训练集训练他自己，然后计算新的情况[1,0,0],预测结果是0.99993704，正确的答案是1，结果是如此的相近！

传统的计算机城西一般情况不能学习，神经网络最神奇的是它可以自己学习，适应，然后解答新的情况，就像人的大脑。

当然一个神经元智能完成一个简单的任务，但是如果我们把数亿个神经元连接在一起会产生什么样的效果？


原文连接： https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1
