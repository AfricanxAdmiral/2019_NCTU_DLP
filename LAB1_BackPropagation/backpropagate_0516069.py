import numpy as np
import matplotlib.pyplot as plt

## Generate Function for input x, y
def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

## Generate Function for input x, y
def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)

## Used Math Function
def MSE(y, pred_y):
    return np.mean((y - pred_y) ** 2)

## Netwowrk Class
class Neural_Network(object):
    def __init__(self):
        self.inputSize = 2
        self.hiddenSize_1 = 4
        self.hiddenSize_2 = 4
        self.outputSize = 1

        self.W1 = np.random.randn(self.inputSize, self.hiddenSize_1)
        self.W2 = np.random.randn(self.hiddenSize_1, self.hiddenSize_2)
        self.W3 = np.random.randn(self.hiddenSize_2, self.outputSize)

    def forward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        self.z4 = self.sigmoid(self.z3)
        self.z5 = np.dot(self.z4, self.W3)
        o = self.sigmoid(self.z5)
        return o

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def derivative_sigmoid(self, x):
        return np.multiply(x, 1.0-x)

    def backward(self, X, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error*self.derivative_sigmoid(o)

        self.z4_error = self.o_delta.dot(self.W3.T)
        self.z4_delta = self.z4_error*self.derivative_sigmoid(self.z4)
        
        self.z2_error = self.z4_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error*self.derivative_sigmoid(self.z2)

        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.z4_delta)
        self.W3 += self.z4.T.dot(self.o_delta)

    def train (self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

## Draw the result graph
def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.gca().set_xlim([-0.1,1.1])
    plt.gca().set_ylim([-0.1,1.1])
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] <= 0.5:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.gca().set_xlim([-0.1,1.1])
    plt.gca().set_ylim([-0.1,1.1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

## Create Netwowrk for both Data
NN_linear = Neural_Network()
NN_XOR = Neural_Network()

## Generate both Data
X_l, y_l = generate_linear()
X_x, y_x = generate_XOR_easy()

for i in range(100000):
    if i%5000 == 0:    
        print("epoch :", i, " Loss :", "%.16f" % MSE(y_l, NN_linear.forward(X_l))) # mean sum squared loss
    NN_linear.train(X_l, y_l)
print(NN_linear.forward(X_l))
show_result(X_l, y_l, NN_linear.forward(X_l))

for i in range(100000):
    if i%5000 == 0:    
        print("epoch :", i, " Loss :", "%.16f" % MSE(y_x, NN_XOR.forward(X_x))) # mean sum squared loss
    NN_XOR.train(X_x, y_x)
print(NN_linear.forward(X_x))
show_result(X_x, y_x, NN_XOR.forward(X_x))
