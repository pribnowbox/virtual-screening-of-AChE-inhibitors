A graph convolutional model was employed to identify potential acetylcholinesterase inhibitors through deep learning. The model was implemented with 5-fold cross-validation to determine the hyperparameter set that produced the highest validating AUCs averaged among the five folds. K-fold-cv.py runs the 5-fold cross-validation on the best-identified hyperparameter set, as shown in Table 1 in the main text. This process creates a model with the best-identified hyperparameter set using the whole dataset (DATASET.csv) to produce a final model.

As the learning algorithm is stochastic, models trained with the same data and hyperparameters may not necessarily yield identical results. Hence, we trained five instances of the models (saved_model_rep01 to 05).

The trained models were then utilized to predict the probability score of all substances in BACMUSHBASE, a database of bioactive compounds from the mushroom species found in Thailand. The code needed to execute the prediction task is provided in predict.py.


A graph convolutional model was implemented with a 5-fold cross-validation (CV) to identify the hyperparameter set that has the highest validating AUCs averaged among the five folds. The code k-fold-cv.py runs the 5-fold CV on the best-identified hyperparameter set, as shown in Table 1 in the main text. This code also creates a model with the best-identified hyperparameter set using the whole dataset (DATASET.csv) to yield a final model.

Since the learning algorithm is stochastic, the models trained with the same data and hyperparameters may not necessarily yield identical results. Thus, five instances of the models were trained and saved as 'saved_model_rep01' to 'saved_model_rep05'.

The trained models were used to predict the probability score of all substances in BACMUSHBASE (http://bacmushbase.sci.ku.ac.th/), a database of bioactive compounds from the mushroom species found in Thailand. The code to execute this task is provided in 'predict.py'.
