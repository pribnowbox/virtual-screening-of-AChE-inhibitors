!pip install --pre deepchem
import deepchem as dc
from deepchem.models.layers import GraphConv, GraphPool, GraphGather
from deepchem.metrics import to_one_hot
from deepchem.feat.mol_graphs import ConvMol
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import pandas as pd

def read_data(fname):
	dataset_file = fname
	featurizer = dc.feat.ConvMolFeaturizer()
	loader = dc.data.CSVLoader(tasks=["class"], smiles_field="smiles", featurizer=featurizer)
	dataset = loader.featurize(dataset_file, shard_size=8192)
	transformer = dc.trans.BalancingTransformer(dataset=dataset)
	transformed_dataset = transformer.transform(dataset)
	return transformed_dataset, transformer

dataset, transformer = read_data('DATASET.csv')

p_gc_nodes = [ 256, 512, 1024 ]
p_dense_nodes = [ 256, 512, 1024 ]
p_learning_rate = [ 1E-4, 1E-3 ]
p_dropout_rate = [ 0, 0.1, 0.2 ]
p_gc_fn = [ 'relu', 'tanh' ]
p_dense_fn = [ 'relu', 'tanh' ]
p_readout_fn = [ 'relu', 'tanh' ]
my_epochs=10
my_batch_size=40
n_tasks=1

K_fold=5
splitter = dc.splits.RandomStratifiedSplitter()
data_splits = splitter.k_fold_split(dataset, K_fold)

df_out = pd.DataFrame()
df=pd.read_csv('hyperparameters.csv', header=0)
for s in range( len(df) ):
	my_gc_nodes = p_gc_nodes[ df.iloc[s,0] ]  
	my_dense_nodes = p_dense_nodes[ df.iloc[s,1] ]
	my_learning_rate = p_learning_rate[ df.iloc[s,2] ]
	my_dropout_rate = p_dropout_rate[ df.iloc[s,3] ]
	my_gc_fn = p_gc_fn[ df.iloc[s,4] ]
	my_dense_fn = p_dense_fn[ df.iloc[s,5] ]
	my_readout_fn = p_readout_fn[ df.iloc[s,6] ]
	
	train_auc=[]
	valid_auc=[]

	if my_gc_fn=='relu' and my_readout_fn=='relu':
		class MyGraphConvModel(tf.keras.Model):
			def __init__(self):
				super(MyGraphConvModel, self).__init__()
				self.gc1 = GraphConv(my_gc_nodes, activation_fn=tf.nn.relu)
				self.batch_norm1 = layers.BatchNormalization()
				self.gp1 = GraphPool()

				self.dense1 = layers.Dense(my_dense_nodes, activation=my_dense_fn)
				self.batch_norm2 = layers.BatchNormalization()
				self.dropout1 = layers.Dropout(rate=my_dropout_rate)
				self.readout = GraphGather(batch_size=my_batch_size, activation_fn=tf.nn.relu)

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
		
	if my_gc_fn=='relu' and my_readout_fn=='tanh':
		class MyGraphConvModel(tf.keras.Model):
			def __init__(self):
				super(MyGraphConvModel, self).__init__()
				self.gc1 = GraphConv(my_gc_nodes, activation_fn=tf.nn.relu)
				self.batch_norm1 = layers.BatchNormalization()
				self.gp1 = GraphPool()

				self.dense1 = layers.Dense(my_dense_nodes, activation=my_dense_fn)
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
		
	if my_gc_fn=='tanh' and my_readout_fn=='relu':
		class MyGraphConvModel(tf.keras.Model):
			def __init__(self):
				super(MyGraphConvModel, self).__init__()
				self.gc1 = GraphConv(my_gc_nodes, activation_fn=tf.nn.tanh)
				self.batch_norm1 = layers.BatchNormalization()
				self.gp1 = GraphPool()

				self.dense1 = layers.Dense(my_dense_nodes, activation=my_dense_fn)
				self.batch_norm2 = layers.BatchNormalization()
				self.dropout1 = layers.Dropout(rate=my_dropout_rate)
				self.readout = GraphGather(batch_size=my_batch_size, activation_fn=tf.nn.relu)

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
	
	if my_gc_fn=='tanh' and my_readout_fn=='tanh':
		class MyGraphConvModel(tf.keras.Model):
			def __init__(self):
				super(MyGraphConvModel, self).__init__()
				self.gc1 = GraphConv(my_gc_nodes, activation_fn=tf.nn.tanh)
				self.batch_norm1 = layers.BatchNormalization()
				self.gp1 = GraphPool()

				self.dense1 = layers.Dense(my_dense_nodes, activation=my_dense_fn)
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

	for train_folds, valid_fold in data_splits:	
			model = dc.models.KerasModel(MyGraphConvModel(), loss=dc.models.losses.SoftmaxCrossEntropy(), learning_rate=my_learning_rate)
			model.fit_generator(data_generator(train_folds, epochs=my_epochs))

			metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
			y1 = model.evaluate_generator(data_generator(train_folds), [metric])
			y2 = model.evaluate_generator(data_generator(valid_fold), [metric])

			train_auc = np.append( train_auc, y1.get('roc_auc_score') )
			valid_auc = np.append( valid_auc, y2.get('roc_auc_score') )

	print('parameters=\n')
	print( my_gc_nodes, my_dense_nodes, my_learning_rate, my_dropout_rate, my_gc_fn, my_dense_fn, my_readout_fn, sep="\n" )
	print('train auc & valid auc')
	print( train_auc, valid_auc, sep="\n")
	print('\nmean of train auc & mean of valid auc')
	print( np.mean(train_auc), np.mean(valid_auc) )
	print('\n\n')
	
	df_new = pd.Series( [ my_gc_nodes, my_dense_nodes, my_learning_rate, my_dropout_rate, my_gc_fn, my_dense_fn, my_readout_fn, train_auc[0], train_auc[1], train_auc[2], train_auc[3], train_auc[4] ] )
	df_out=df_out.append(df_new, ignore_index=True)
	df_new = pd.Series( [ my_gc_nodes, my_dense_nodes, my_learning_rate, my_dropout_rate, my_gc_fn, my_dense_fn, my_readout_fn, valid_auc[0], valid_auc[1], valid_auc[2], valid_auc[3], valid_auc[4] ] )
	df_out=df_out.append(df_new, ignore_index=True)
	
df_out.to_csv('out.csv', header=False, index=False)
