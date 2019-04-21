
import numpy as np
import chainer.links as L
import chainer.functions as F

print("START!")

# Layer 1
# W1 = np.array([[1, 10],[-1, 1]])
W1 = np.array([[1, -1], [10, 1]])
b1 = 0.0
p1 = L.Linear(2, 2, initialW=W1, initial_bias=b1)
f1 = F.relu

# Layer 2
w2 = np.array([[1, 1]])
b2 = 0.0
p2 = L.Linear(2, 1, initialW = w2, initial_bias=b2)

def XOR(x):
    h1 = f1(p1(x))
    return p2(h1)

x = np.array([[1,1],[0,1],[1,0],[0,0]], dtype=np.float32) # input data
print(XOR(x).data)


print('DONE!')