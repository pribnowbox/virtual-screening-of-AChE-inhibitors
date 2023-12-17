A graph convolutional model was implemented with a 5-fold cross-validation (CV) to identify the hyperparameter set with the highest validating AUCs averaged among the five folds. The code `k-fold-cv.py` runs the 5-fold CV on the 44 hyperparameter sets listed in Supplementary Table S1.

After identifying the best hyperparameter set (Table 1 in the main text), a final model was created by training on the whole dataset "DATASET.csv". The code to execute this task is provided in `train_model.py`.

Since the learning algorithm is stochastic, the models trained with the same data and hyperparameters may not necessarily yield identical results. Thus, five instances of the models were trained and saved as `saved_model_rep01` to `saved_model_rep05`.

The trained models were used to predict the probability score of all substances in BACMUSHBASE (http://bacmushbase.sci.ku.ac.th/), a database of bioactive compounds from the mushroom species found in Thailand (`mushroom_dataset.csv`). The code to execute this task is provided in `predict.py`.
