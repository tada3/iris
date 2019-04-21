
import numpy as np
import chainer.functions as F

print("START!")

# Layer 1
# W1 = np.array([[1, 10],[-1, 1]])
w1 = np.array([[2, 4], [6, 8]], dtype=np.float32)
print(w1)

y = F.sum(w1)
print(y.shape)

print(y.data)


w2 = np.array([[1, 2], [3, 4]], dtype=np.float32)
print(w2)

w3 = w1 - w2
print(w3)

w4 = abs(w3)
print(w4)

w5 = w4 - 1
print(w5)

print(w5.size)
z = F.sum(w5)/w5.size
print(z.shape)
print(z.data)

