!pip install --pre deepchem
import deepchem as dc
from deepchem.models.layers import GraphConv, GraphPool, GraphGather
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import pandas as pd

my_learning_rate=1E-4
my_gc1_nodes=1024
my_dense_nodes=256
my_dropout_rate=0.0
my_epochs=10
my_batch_size=40

class MyGraphConvModel(tf.keras.Model):

  def __init__(self):
    super(MyGraphConvModel, self).__init__()
    self.gc1 = GraphConv(my_gc1_nodes, activation_fn=tf.nn.relu)
    self.batch_norm1 = layers.BatchNormalization()
    self.gp1 = GraphPool()

    self.dense1 = layers.Dense(my_dense_nodes, activation=tf.nn.relu)
    self.batch_norm2 = layers.BatchNormalization()
    self.dropout1 = layers.Dropout(rate=my_dropout_rate)
    self.readout = GraphGather(batch_size=my_batch_size, activation_fn=tf.nn.tanh)

    self.dense2 = layers.Dense(n_tasks*2)
    self.logits = layers.Reshape((n_tasks, 2))
    self.softmax = layers.Softmax()

  def call(self, inputs):
    gc1_output = self.gc1(inputs)
    batch_norm1_output = self.batch_norm1(gc1_output)
    gp1_output = self.gp1([batch_norm1_output] + inputs[1:])

    dense1_output = self.dense1(gp1_output)
    batch_norm2_output = self.batch_norm2(dense1_output)
    dropout1_output = self.dropout1(batch_norm2_output)
    readout_output = self.readout([dropout1_output] + inputs[1:])

    logits_output = self.logits(self.dense2(readout_output))
    return self.softmax(logits_output)

from deepchem.metrics import to_one_hot
from deepchem.feat.mol_graphs import ConvMol

def data_generator(dataset, epochs=1, predict=False):
  for ind, (X_b, y_b, w_b, ids_b) in enumerate(dataset.iterbatches(my_batch_size, epochs, deterministic=True, pad_batches=True)):
    multiConvMol = ConvMol.agglomerate_mols(X_b)
    inputs = [multiConvMol.get_atom_features(), multiConvMol.deg_slice, np.array(multiConvMol.membership)]
    for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
      inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
    labels = [to_one_hot(y_b.flatten(), 2).reshape(-1, n_tasks, 2)]
    weights = [w_b]
    yield (inputs, labels, weights)

n_tasks=1
# replace 'saved_model_rep01' with 'saved_model_rep02' to 'saved_model_rep05'
model = dc.models.KerasModel(MyGraphConvModel(), loss=dc.models.losses.SoftmaxCrossEntropy(), learning_rate=my_learning_rate, model_dir="saved_model_rep01")
model.restore()

dataset_file = "mushroom_dataset.csv"
featurizer = dc.feat.ConvMolFeaturizer()
loader = dc.data.CSVLoader(tasks=['class'], smiles_field="canonical_smiles", featurizer=featurizer)
predict_dataset = loader.featurize(dataset_file, shard_size=8192)

y=model.predict_on_generator(data_generator(predict_dataset, epochs=1))

ymod=np.empty(0)
for i in range(len(y)):
  ymod=np.append(ymod, y[i][0][1] )
df_prob = pd.DataFrame(ymod)

df = pd.read_csv('mushroom_dataset.csv', header=0)
df.drop(columns=['index', 'class'], inplace=True)
df['active_prob']=df_prob[0]
df.sort_values(by=['active_prob'], ascending=False, inplace=True)
filename='mushroom_prediction_sorted.csv'
df.to_csv(filename, index=False)

                                                                     
