import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import time
import os
from sklearn.metrics import r2_score
from datetime import datetime
import types
import pickle

from keras.models import Sequential
from keras.optimizers import SGD, Adagrad, RMSprop
from keras import callbacks


def main(model=Sequential(),
         train_data=([], []), # (X, Y) or function(sample_index, **kwargs) returning (x, y)
         val_data=([], []),
         epochs=10,
         npoints=0,
         batch_size=64,
         optimizer_name='rmsprop',
         lr=0.001,
         epsilon=1e-8,
         decay=0.0, # suggested: lr / epochs
         log_dir='', # empty: saves to tb_logs/current_datetime
         tensorboard_histogram_freq=10,
         ylabels=[],
         verbose=True,
         **gen_kwds):
    nbatches = get_number_of_batches(batch_size, npoints, train_data)
    optimizer = get_optimizer(optimizer_name, lr, epsilon, decay)
    compile_model(model, optimizer)
    callbacks_list, tb_log_dir, model_filepath = setup_callbacks(
        log_dir, tensorboard_histogram_freq
    )
    history = fit_model(
        model, train_data, val_data,
        epochs, batch_size, nbatches,
        callbacks_list, verbose, **gen_kwds
    )
    evaluate_model(
        model, train_data, val_data, batch_size, nbatches,
        history.history, ylabels, verbose, **gen_kwds
    )
    save_history(history.history, tb_log_dir)
    finalize(tb_log_dir, model_filepath, verbose)


def get_number_of_batches(batch_size=16, npoints=0, train_data=None):
    if npoints <= 0:
        if type(train_data) is types.TupleType:
            if len(train_data) == 2:
                X, y = train_data
                assert np.shape(X)[0] == np.shape(y)[0]
                npoints = np.shape(X)[0]

    nbatches = int(np.ceil(float(npoints) / batch_size))

    return nbatches


def get_optimizer(optimizer_name='rmsprop',
                 lr=0.001,
                 epsilon=1e-8,
                 decay=0.0, # suggested: lr / epochs
                  ):
    if optimizer_name == 'adagrad':
        # defaults:
        # keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
        opt = Adagrad(lr=lr, epsilon=epsilon, decay=decay)
    elif optimizer_name == 'sgd':
        # defaults:
        # keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        # saw on web:
        # decay_rate = initial_learning_rate / epochs
        # momentum = 0.8
        opt = SGD(lr=lr, momentum=epsilon,
                  decay=decay,
                  nesterov=False)
    elif optimizer_name == 'rmsprop':
        # defaults:
        # keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        opt = RMSprop(lr=lr, epsilon=epsilon, decay=decay)

    return opt


def compile_model(model, optimizer):
    model.compile(loss='mean_squared_error', optimizer=optimizer)


def setup_callbacks(log_dir="", tensorboard_histogram_freq=10):
    if not log_dir:
        log_dir = get_datetime_logdir()

    tb_callback = callbacks.TensorBoard(log_dir=log_dir,
                                        histogram_freq=tensorboard_histogram_freq,
                                        write_graph=True,
                                        write_grads=True,
                                        write_images=True)
    model_filepath = os.path.join(log_dir, "keras_model.h5")  # "keras_model.hdf5"
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=model_filepath,
        monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False, mode='auto', period=1
    )

    return [tb_callback, checkpoint_callback], log_dir, model_filepath


def fit_model(model, train_data, val_data, epochs, batch_size, nbatches,
              callbacks_list, verbose, **gen_kwds):
    Xval, yval = val_data

    if type(train_data) is types.TupleType:
        Xtrain, ytrain = train_data
        history = model.fit(Xtrain, ytrain,
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=verbose,
                            validation_data=(Xval, yval),
                            callbacks=callbacks_list)
    else:
        data_gen = keras_data_generator(data_function=train_data,
                                        batch_size=batch_size,
                                        nbatches=nbatches,
                                        **gen_kwds)
        model.fit_generator(data_gen,
                            epochs=epochs,
                            steps_per_epoch=batch_size,
                            validation_data=(Xval, yval),
                            callbacks=callbacks_list,
                            verbose=verbose)

    return history


def evaluate_model(model, train_data, val_data, batch_size, nbatches,
                   history, ylabels, verbose, **gen_kwds):
    if verbose:
        Xval, yval = val_data
        val_loss = model.evaluate(Xval, yval)
        val_score = r2_score(yval, model.predict(Xval))

        if type(train_data) is types.TupleType:
            Xtrain, ytrain = train_data
            train_loss = model.evaluate(Xtrain, ytrain)
            train_score = r2_score(ytrain, model.predict(Xtrain))
        else:
            train_loss = 0
            train_score = 0

            for batch_number in range(nbatches):
                Xbatch, ybatch = train_data(batch_number, batch_size, **gen_kwds)
                train_loss += model.evaluate(Xbatch, ybatch)
                train_score += r2_score(ybatch, model.predict(Xbatch))

        plot_training_results(model, history, Xval, yval, ylabels, "training plots")

        print
        print "train loss: {}".format(train_loss)
        print "test loss: {}".format(val_loss)
        print "train score: {}".format(train_score)
        print "test score: {}".format(val_score)


def save_history(history, tb_log_dir):
    file_path = os.path.join(tb_log_dir, 'history.pkl')
    with open(file_path, 'w') as fp:
        pickle.dump(history, fp)
        print
        print "history saved to:"
        print file_path


def finalize(tb_log_dir, model_filepath, verbose):
    if verbose:
        print
        print "tensorboard-viewable logs saved to:"
        print tb_log_dir
        print
        print "keras model (with the lowest test loss) saved to:"
        print model_filepath


def keras_data_generator(data_function=object, # function(sample_index, **kwargs) returning (x, y)
                         batch_size=64,
                         nbatches=20,
                         **kwargs):
    while True:
        for k in range(nbatches):
            Xbatch, ybatch = get_kth_batch(k, batch_size,
                                           data_function, **kwargs)
            yield Xbatch, ybatch


def get_kth_batch(k, batch_size, data_function, **kwargs):
    Xbatch, ybatch = [], []

    for j in range(batch_size):
        sample_index = k * batch_size + j
        x, y = data_function(sample_index, **kwargs)
        Xbatch.append(x)
        ybatch.append(y)

    return np.array(Xbatch), np.array(ybatch)



def get_datetime_logdir():
    dt = datetime.now()
    log_dir = "tb_logs/{}".format(dt.strftime("%Y-%m-%d_%H:%M:%S"))
    return log_dir


def plot_training_results(model, history, Xtest, ytest,
                          ylabels=[],
                          figname='training results'):
    # plot the learning curves:
    plt.figure(str(figname) + 'loss')
    plt.semilogy(history['loss'], label='train loss')
    plt.semilogy(history['val_loss'], label='test loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    # plt.plot(history['loss'])
    # plt.plot(history['val_loss'])

    # predict and do a truth plot separately for each signal
    ypred = model.predict(Xtest)
    nyrows, nycols = np.shape(ypred)

    if ylabels == []:
        ylabels = ["signal {}".format(x) for x in range(nycols)]
    plt.figure(str(figname) + 'learning_curves', figsize=(15, 15))
    k = 1

    # reshape ytrain or yval if needed:
    if len(ypred.shape) == 1: ypred = ypred[:, np.newaxis]
    if len(ytest.shape) == 1: ytest = ytest[:, np.newaxis]

    for yp, yt, lbl in zip(ypred.T, ytest.T, ylabels):
        ncols = nrows = int(np.ceil(np.sqrt(nycols)))
        plt.subplot(nrows, ncols, k)
        plt.plot(yt, yp, '.')
        ax = plt.gca()
        ax.annotate(lbl, xy=(0.1, 0.9), xycoords='axes fraction')
        plt.ylabel('predicted')
        plt.xlabel('test data')
        k += 1

    plt.tight_layout()
    plt.show()


def check_conv1d_model_inputs(X_train, X_test, y_train, y_test):
    # add a new axis to all for the conv layers if needed:
    if len(np.shape(X_train)) < 3:
        Xtrain = np.copy(X_train)[:, :, np.newaxis]
    else:
        Xtrain = np.copy(X_train)

    if len(np.shape(X_test)) < 3:
        Xtest = np.copy(X_test)[:, :, np.newaxis]
    else:
        Xtest = np.copy(X_test)

    if len(np.shape(y_train)) < 2:
        ytrain = np.copy(y_train)[:, np.newaxis]
    else:
        ytrain = np.copy(y_train)

    if len(np.shape(y_test)) < 2:
        ytest = np.copy(y_test)[:, np.newaxis]
    else:
        ytest = np.copy(y_test)

    return Xtrain, Xtest, ytrain, ytest


if __name__ == "__main__":
    main()