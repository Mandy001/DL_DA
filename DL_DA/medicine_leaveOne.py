# coding=utf-8
import os

from sklearn.decomposition import PCA

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from scipy.stats import moment
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, accuracy_score,f1_score
# from torch import nn
# import torch
# import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random
from medicine import data_processing_medicine as data_processing
import os
from sklearn import metrics
from sklearn.model_selection import KFold, LeaveOneOut
import tensorflow as tf
import scipy
# tf.enable_eager_execution()
from tensorflow import keras
from sklearn.svm import LinearSVC


train_seed = 100
np.random.seed(train_seed)
tf.random.set_seed(train_seed)


# model = 'SVM'

# model = 'LR'

# model = 'MLP'
model = 'TextCNN'

# model = 'LSTM'


learning_rate = 1e-4

file_path = './data/data_feature.xlsx'
X, Y, treat_label = data_processing.get_data(file_path)



num_classes = len(set(Y))
print(X.shape)

def pca(X):
    pca = PCA(n_components=100)
    pca.fit(X)
    return pca.transform(X)






def evaluate(y_true, y_pred):
    if len(np.shape(y_true)) > 1:
        y_true = np.argmax(y_true, axis=-1)
    if len(np.shape(y_pred)) > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    return precision_recall_fscore_support(y_true, y_pred, average='macro')


def create_mlp_model(num_classes):
    model = keras.Sequential([
        keras.layers.Dense(100, activation=tf.nn.relu),
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dense(num_classes, activation=keras.activations.softmax)
    ])
    return model

def create_lr_model(num_classes):
    model = keras.Sequential([
        keras.layers.Dense(num_classes, activation=keras.activations.softmax)
    ])
    return model

def create_LSTM_model(num_classes):
    model = keras.Sequential([
        keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
        keras.layers.LSTM(100, return_sequences=True),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Flatten(),
        keras.layers.Dense(500, activation=tf.nn.relu),
        keras.layers.Dense(num_classes, activation=keras.activations.softmax)
    ])
    return model

def create_my_LSTM_model(num_classes):
    model = keras.Sequential([
        keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
        keras.layers.LSTM(20, return_sequences=True),
        # keras.layers.Dense(100, activation=tf.nn.relu),
        # keras.layers.Dense(10, activation=tf.nn.relu),
        # keras.layers.Flatten(),
        # keras.layers.Dense(100, activation=tf.nn.relu),
        keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=-1)),
        keras.layers.Dense(100, activation=tf.nn.relu),
        keras.layers.Dense(num_classes, activation=keras.activations.softmax)
    ])
    return model

def create_CNN_model(num_classes):
    encoder = keras.Sequential([
        keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
        keras.layers.Conv1D(512, 5, activation=tf.nn.relu),
        keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=-1)),
    ])

    cls = keras.Sequential([
        keras.layers.Dense(500, activation=tf.nn.relu),
        keras.layers.Dense(num_classes, activation=keras.activations.softmax)
    ])

    model = keras.Sequential([
        encoder,
        cls
    ])

    return model, encoder




# create_model_func = create_mlp_model
create_model_func = create_CNN_model

if model == 'LR':
    create_model_func = create_lr_model

if model == 'MLP':
    create_model_func = create_mlp_model

if model == 'TextCNN':
    create_model_func = create_CNN_model

if model == 'LSTM':
    create_model_func = create_LSTM_model

print('The model is : ', model)


def train_and_predict(create_model_func, X, Y, train_indices, test_indices, split_index=None):

    X_train = X[train_indices]
    Y_train = tf.one_hot(Y[train_indices], depth=num_classes).numpy()
    X_test = X[test_indices]
    Y_test = tf.one_hot(Y[test_indices], depth=num_classes).numpy()


    model = create_model_func(num_classes)
    if isinstance(model, tuple):
        model, encoder = model

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])
    # print(tf.one_hot(Y[train_indices], depth=num_classes).numpy())
    print(f"training split: {split_index} ...")
    model.fit(X_train, Y_train,
              epochs=500,
              batch_size=X.shape[0],
              validation_data=(X_test, Y_test), verbose=False,
              validation_batch_size=X_test.shape[0],
              validation_freq=100)
    y_pred = model(X_test).numpy()


    return np.argmax(y_pred, axis=-1)



def train_and_predict_svm(X, Y, train_indices, test_indices, split_index=None):

    X_train = X[train_indices]
    Y_train = Y[train_indices]
    X_test = X[test_indices]
    Y_test = Y[test_indices]

    cls = LinearSVC()
    cls.fit(X_train, Y_train)

    y_pred = cls.predict(X_test)
    return y_pred


loo = LeaveOneOut()



loo_y_pred = []
loo_y_true = []
for split_index, (train_indices, test_indices) in tqdm(enumerate(loo.split(X, Y))):

    if model == 'SVM':
        # svm
        y_pred = train_and_predict_svm(X, Y, train_indices, test_indices, split_index=split_index)
    else:
        y_pred = train_and_predict(create_model_func, X, Y, train_indices, test_indices, split_index=split_index)

    loo_y_pred.append(y_pred[0])
    loo_y_true.append(Y[test_indices[0]])



loo_y_pred = np.array(loo_y_pred)
loo_y_true = np.array(loo_y_true)

print(loo_y_pred.shape)
accuracy = accuracy_score(loo_y_true, loo_y_pred)
macro_f_score = f1_score(loo_y_true, loo_y_pred, average="macro")

print(f"accuracy = {accuracy:.4f}\tf1 = {macro_f_score:.4f}")
print(classification_report(loo_y_true, loo_y_pred))
print(confusion_matrix(loo_y_true, loo_y_pred))


print('\nTREATED....')
treated_mask = treat_label == "TREATED"
print(np.sum(treated_mask))
accuracy = accuracy_score(loo_y_true, loo_y_pred)
macro_f_score = f1_score(loo_y_true, loo_y_pred, average="macro")

print(f"accuracy = {accuracy:.4f}\tf1 = {macro_f_score:.4f}")
print(classification_report(loo_y_true[treated_mask], loo_y_pred[treated_mask],digits=3))
print(confusion_matrix(loo_y_true[treated_mask], loo_y_pred[treated_mask]))


print('\nUNTREATED....')
untreated_mask = treat_label != "TREATED"
print(np.sum(untreated_mask))
accuracy = accuracy_score(loo_y_true, loo_y_pred)
macro_f_score = f1_score(loo_y_true, loo_y_pred, average="macro")

print(f"accuracy = {accuracy:.4f}\tf1 = {macro_f_score:.4f}")
print(classification_report(loo_y_true[untreated_mask], loo_y_pred[untreated_mask],digits=3))
print(confusion_matrix(loo_y_true[untreated_mask], loo_y_pred[untreated_mask]))

exit()





