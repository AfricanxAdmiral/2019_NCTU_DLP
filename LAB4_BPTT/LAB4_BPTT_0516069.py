import numpy as np
import matplotlib.pyplot as plt
import random
import copy

## Hyper Parameter
learning_rate = 0.01
epochs =10000
T = 8                   # length of sequence
input_dim = 2
hidden_dim = 8         
output_dim = 1

## Create Training data
def sampledata():
    num1 = random.randint(0,127)
    num2 = random.randint(0,127)
    num3 = num1 + num2

    b_num1 = (bin(num1)[2:])
    b_num2 = (bin(num2)[2:])
    b_num3 = (bin(num3)[2:])
   

    x = ('0' * ( 8 - len(b_num1)) + b_num1)
    y = ('0' * ( 8 - len(b_num2)) + b_num2)
    z = ('0' * ( 8 - len(b_num3)) + b_num3)
    
    x = np.array(list(x), dtype = int)
    y = np.array(list(y), dtype = int)
    z = np.array(list(z), dtype = int)
    return (x, y), z
	
out_x, out_y = sampledata()
	
## Math function
def tanh_derivative(x):
    return 1 - x**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def MSE(pred, t):
    mse = np.array([(x - y)**2 for x, y in zip(pred, t)])
    return mse.sum()
	
class RNN(object):
    def __init__(self, input_dim, hidden_dim, output_dim, T, lr):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.T = T
        self.lr = lr
        
        self.U = 2*np.random.random((self.input_dim, self.hidden_dim))-1
        self.W = 2*np.random.random((self.hidden_dim, self.hidden_dim))-1
        self.V = 2*np.random.random((self.hidden_dim, self.output_dim))-1
        
    def forward(self, x, y):
        self.h = []
        self.ot_ = []
        d = np.zeros_like(y)
        k = np.zeros_like(y)
        self.h.append(np.zeros(8))
        for i in range(8):
            X = np.array([[x[0][7-i],x[1][7-i]]])
            Y = np.array([[y[7 - i]]]).T
            ht = np.tanh(np.dot(X,self.U) + np.dot(self.h[i],self.W))
            ot = sigmoid(np.dot(ht,self.V))
            error = Y - ot
            self.ot_.append(error*sigmoid_derivative(ot))
            if ot[0][0]>0.5000000000:
                d[7-i] = 1
            else:
                d[7-i] = 0
            self.h.append(ht)
        return d
    
    def backward(self, x):
        future_ht_d = np.zeros(8)
        dv = 0
        dw = 0
        du = 0
        for i in range(8):
            X = np.array([[x[0][i],x[1][i]]])
            ht = self.h[8-i]
            if i == 7:
                ht_1 = np.zeros_like(ht)
            else:
                ht_1 = self.h[7-i]
            #ht_1 = h[-i-2]
            d_ot = self.ot_[7-i]
            #print(d_ot)
            d_ht = (np.dot(future_ht_d,self.W.T) + np.dot(d_ot,self.V.T))*tanh_derivative(ht)
            future_ht_d = d_ht
            dv += np.dot(np.atleast_2d(ht).T, d_ot)
            dw += np.dot(np.atleast_2d(ht_1).T, d_ht)
            du += np.dot(X.T, d_ht)

        self.U += self.lr * du
        self.W += self.lr * dw
        self.V += self.lr * dv
		
def testing():
    correct = 0
    total = 0
    for _ in range(1000):
        x, y = sampledata()
        pred_y = net.forward(x, y)
        correct += np.equal(pred_y, y).sum()
        total += 8
        ##print(x[0], '+', x[1], '=', out, '(', y, ')')
    print('test accuracy :', 100.*correct/total, '%')
	
net = RNN(input_dim, hidden_dim, output_dim, T, learning_rate)
## net.status()

train_correct = 0
train_total = 0
accuracy = []

for epoch in range(epochs):
    x, y = sampledata()
    pred_y = net.forward(x, y)
    net.backward(x)
    train_correct += np.equal(pred_y, y).sum()
    train_total += 8
    if epoch%100==0:
        accuracy.append(train_correct/train_total)
    if epoch % 1000 == 999:
        print('epoch :', epoch+1, '| train accuracy :', 100.*train_correct/train_total, '%')
        ##pred_y = net.forward(out_x, out_y)
        ##print(out_x[0], '+', out_x[1], '=', pred_y, '(', out_y, ')')
        train_correct = 0
        train_total = 0

testing()
		
x_axix = np.arange(epochs/100)*100
plt.title('Training Accuracy')
plt.plot(x_axix, accuracy)
plt.show()