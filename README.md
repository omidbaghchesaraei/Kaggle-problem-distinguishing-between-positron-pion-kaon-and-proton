# Kaggle-problem-distinguishing-between-positron-pion-kaon-and-proton
Kaggle Problem: Using TensorFlow and classical machine learning algorithms, I identify four different types of particles, including positrons, pions, kaons, and protons with an accuracy of 95%.

The dataset contains 5,000,000 records, and the distribution of particles within the dataset is as follows:

1) The pion 56.14 % or 2,806,833
2) The proton 38.92 % or 1,945,849
3) The kaon 4.65 % or 232,471
4) The positron 0.30 % or 14,847

As can be seen, there is an unequal distribution of targets(classes) in the dataset, so data has imbalanced classes, which brings challenges to feature correlation, class separation and evaluation, and results in poor model performance. I change the dataset that I use to build my predictive model to have more balanced data. This change is called sampling my dataset. I first separate the dataset for each class, resulting in four subsets: one for the positron class with 14,847 records, one for the kaon class with 232,471 records, one for the proton class with 1,945,849 records, and one for the pion class with 2,806,833 records. For each subset, with the sample() method, I generate a random sample of rows, where the number of random rows equals the number of positrons. Then, I merge these four subsets.

My research involves two steps. The first step is to compare 6 classical machine learning algorithms using Accuracy and ROC Curves, Precision-Recall Curves, and Confusion Matrix to find the best classifier. According to my check, RandomForestClassifier has an Accuracy of 94.7%, a micro-average ROC curve(area = 0.99), a macro-average ROC curve(area = 0.99), and a micro-average Precision-recall curve(area=0.985).

Then, I train a simple Neural Network (two hidden layers, each with 128 neurons and an activation function of sigmoid, epochs = 500), and then plot accuracy and loss graphs on the training and validation datasets to find a balance between the model that is underfitting and one that is overfitting(or converging), resulting in a model with a good fit.

According to the plot of loss, validation loss is decreasing before the 100th epoch, so the model is underfitting. However, after the 100th epoch, validation loss is increasing, which indicates the model is overfitting. At the 100th epoch, when the model is either perfectly fitted or in a local minimum, the neural network model achieved an Accuracy of 95%. I found an optimum where the change in the slope of loss is around the 100th epoch. At that point, the training process can be stopped.
