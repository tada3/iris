import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

print("START!")

# Layer 1
# w1 = np.array([[1, 10],[-1, 1]])
w1 = np.array([[1, -1], [-1, 1]])
b1 = 0.0

# Layer 2
w2 = np.array([[1, 1]])
b2 = 0.0

class Xor(chainer.Chain):
    def __init__(self):
        super().__init__()
        # https://docs.chainer.org/en/stable/reference/generated/chainer.Chain.html#chainer.Chain.init_scope
        with self.init_scope():
            self.l1 = L.Linear(2, 2, initialW=w1, initial_bias=b1)
            self.l2 = L.Linear(2, 1, initialW=w2, initial_bias=b2) 

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        return self.l2(h1)

nw = Xor()
in_data = np.array([[1,1],[0,1],[1,0],[0,0]], dtype=np.float32)
out_data = nw(in_data).data

for i in range(0, len(in_data)):
    print(in_data[i], '->', out_data[i])

print('DONE!')