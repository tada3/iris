
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F




print("START!")

class Xor(chainer.Chain):
    def __init__(self):
        super(Xor, self).__init__(
            l1 = L.Linear(2, 10),
            l2 = L.Linear(10, 1)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        return self.l2(h1)

my_xor = Xor()
chainer.serializers.load_npz('my_xor.npz', my_xor)

x1 = np.array([[1,1],[1,0],[0,1],[0,0]], dtype=np.float32)
result = my_xor(x1)
print(result.data)

print("l1.W:")
print(my_xor.l1.W.data)
print("l1.b:")
print(my_xor.l1.b.data)
print("l2.W:")
print(my_xor.l2.W.data)
print("l2.b:")
print(my_xor.l2.b.data)

print('DONE!')