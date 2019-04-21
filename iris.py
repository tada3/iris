import numpy as np
from sklearn import datasets
import chainer
import chainer.links as L
import chainer.functions as F
from chainer.training import extensions


BATCH_SIZE = 5
MAX_EPOCH = 80

iris_data = datasets.load_iris()
#print('data', iris_data.data)
#print('target', iris_data.target)
#print('target_names', iris_data.target_names)
#print('feature_names', iris_data.feature_names)

data = iris_data.data.astype(np.float32)
label = iris_data.target.astype(np.int32)

dataset = chainer.datasets.TupleDataset(data, label)
train_count = int(len(dataset) * 0.7)
train_ds, test_ds = chainer.datasets.split_dataset_random(dataset, train_count, seed=1)
print('train_ds:', len(train_ds))
print('test_ds:', len(test_ds))

class IrisBunruiki(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(4, 10)
            self.l2 = L.Linear(None, 3)
    
    def __call__(self, x):
        h = F.relu(self.l1(x))
        return self.l2(h)


bunruiki = IrisBunruiki()
model = L.Classifier(bunruiki)

optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

train_iter = chainer.iterators.SerialIterator(train_ds, BATCH_SIZE)
test_iter = chainer.iterators.SerialIterator(test_ds, BATCH_SIZE, repeat=False, shuffle=False)

updater = chainer.training.StandardUpdater(train_iter, optimizer, device=-1)

trainer = chainer.training.Trainer(updater, (MAX_EPOCH, 'epoch'), out='train_result')
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
# trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy']))
trainer.extend(extensions.LogReport())

print('### trainer start!')
trainer.run()
print('### trainer done!')

# test
#sample_input = np.array([test_ds[0], [5.9, 3.0, 5.1, 1.8]], dtype=np.float32)
#result0 = model.predictor(sample_input)
#result = F.softmax(result0).data.argmax(axis=1)
#for i in range(0, len(sample_input)):
#    print(sample_input[i], '->', result[i])

#sample_input = chainer.datasets.SubDataset(test_ds, 0, 5)
sample_input = np.array([data[0], data[30], data[90], data[120]], dtype=np.float32)
answer = np.array([label[0], label[30], label[90], label[120]], dtype= np.int32 )
result0 = model.predictor(sample_input)
result = F.softmax(result0).data.argmax(axis=1)
for i in range(0, len(sample_input)):
    print(sample_input[i], '->', result[i], answer[i])