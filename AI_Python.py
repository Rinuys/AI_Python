from numpy import array, dot, random
import matplotlib.pyplot as plt
import math as mt

fx = lambda x: 0 if x<0 else 1
sig = lambda x : 1.0/(1.0+mt.exp(-x*1.0))

data = [
    (array([0,0,1]),0),
    (array([0,1,1]),1),
    (array([1,0,1]),1),
    (array([1,1,1]),1)
]

w = random.rand(3)
c = 0.2
n = 50

for i in range(n):
    for j in range(len(data)):
        x, y = data[j]
        net = dot(w,x)
        delta = y - sig(net)
        w += c * delta * x
    if i % 10 == 0:
        print('i=%d'%i)
        for x, _ in data:
            net = dot(x,w)
            print(x[:2], net, sig(net))

print("------------------------------")

for x, _ in data:
    net = dot(x,w)
    print(x[:2], net, sig(net))


