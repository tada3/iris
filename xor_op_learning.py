import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer.training import extensions

print("START!")


class Xor(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(2, 10)
            self.l2 = L.Linear(10, 1)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        return self.l2(h1)

nw = Xor()
accfun = lambda x, t: F.sum(1 - abs(x - t))/x.size
model = L.Classifier(nw, lossfun=F.mean_squared_error, accfun=accfun)

optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

indata = np.array([[0,0], [0,1], [1,0],[1,1]], dtype=np.float32)
labels = np.array([[0], [1], [1], [0]], dtype= np.float32)

dataset = chainer.datasets.TupleDataset(indata, labels)
train_iter = chainer.iterators.SerialIterator(dataset, 4)
test_iter = chainer.iterators.SerialIterator(dataset, 4, repeat=False, shuffle=False)

updater = chainer.training.StandardUpdater(train_iter, optimizer)

trainer = chainer.training.Trainer(updater, (1000, 'epoch'), out='test_result')
trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy']))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.ProgressBar())

print('### trainer start!')
trainer.run()
print('### trainer done!')

x1 = np.array([[1,1],[1,0],[0,1],[0,0]], dtype=np.float32)
result = nw(x1)
print(result.data)

print("l1.W:")
print(nw.l1.W.data)
print("l1.b:")
print(nw.l1.b.data)
print("l2.W:")
print(nw.l2.W.data)
print("l2.b:")
print(nw.l2.b.data)

chainer.serializers.save_npz('my_xor.npz', nw)

print('DONE!')