import os
import sys
import numpy as np
from scipy.stats import logistic
from scipy.stats import spearmanr
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.base import clone
import inspect
from Shapley import ShapNN, CShapNN
from multiprocessing import dummy as multiprocessing
from sklearn.metrics import roc_auc_score, f1_score
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
        
def convergence_plots(marginals):
    
    plt.rcParams['figure.figsize'] = 15,15
    for i, idx in enumerate(np.arange(min(25, marginals.shape[-1]))):
        plt.subplot(5,5,i+1)
        plt.plot(np.cumsum(marginals[:, idx])/np.arange(1, len(marginals)+1))    
        
    
def is_integer(array):
    return (np.equal(np.mod(array, 1), 0).mean()==1)


def is_fitted(model):
        """Checks if model object has any attributes ending with an underscore"""
        return 0 < len( [k for k,v in inspect.getmembers(model) if k.endswith('_') and not k.startswith('__')] )


def return_model(mode, **kwargs):
    
    if mode=='logistic':
        solver = kwargs.get('solver', 'liblinear')
        n_jobs = kwargs.get('n_jobs', None)
        max_iter = kwargs.get('max_iter', 5000)
        model = LogisticRegression(solver=solver, n_jobs=n_jobs, 
                                 max_iter=max_iter, random_state=666)
    elif mode=='Tree':
        model = DecisionTreeClassifier(random_state=666)
    elif mode=='RandomForest':
        n_estimators = kwargs.get('n_estimators', 50)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=666)
    elif mode=='GB':
        n_estimators = kwargs.get('n_estimators', 50)
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=666)
    elif mode=='AdaBoost':
        n_estimators = kwargs.get('n_estimators', 50)
        model = AdaBoostClassifier(n_estimators=n_estimators, random_state=666)
    elif mode=='SVC':
        kernel = kwargs.get('kernel', 'rbf')
        model = SVC(kernel=kernel, random_state=666)
    elif mode=='LinearSVC':
        model = LinearSVC(loss='hinge', random_state=666)
    elif mode=='GP':
        model = GaussianProcessClassifier(random_state=666)
    elif mode=='KNN':
        n_neighbors = kwargs.get('n_neighbors', 5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif mode=='NB':
        model = MultinomialNB()
    elif mode=='linear':
        model = LinearRegression(random_state=666)
    elif mode=='ridge':
        alpha = kwargs.get('alpha', 1.0)
        model = Ridge(alpha=alpha, random_state=666)
    elif 'conv' in mode:
        tf.reset_default_graph()
        address = kwargs.get('address', 'weights/conv')
        hidden_units = kwargs.get('hidden_layer_sizes', [20])
        activation = kwargs.get('activation', 'relu')
        weight_decay = kwargs.get('weight_decay', 1e-4)
        learning_rate = kwargs.get('learning_rate', 0.001)
        max_iter = kwargs.get('max_iter', 1000)
        early_stopping= kwargs.get('early_stopping', 10)
        warm_start = kwargs.get('warm_start', False)
        batch_size = kwargs.get('batch_size', 256)
        kernel_sizes = kwargs.get('kernel_sizes', [5])
        strides = kwargs.get('strides', [5])
        channels = kwargs.get('channels', [1])
        validation_fraction = kwargs.get('validation_fraction', 0.)
        global_averaging = kwargs.get('global_averaging', 0.)
        optimizer = kwargs.get('optimizer', 'sgd')
        if mode=='conv':
            model = CShapNN(mode='classification', batch_size=batch_size, max_epochs=max_iter,
                          learning_rate=learning_rate, 
                          weight_decay=weight_decay, validation_fraction=validation_fraction,
                          early_stopping=early_stopping,
                         optimizer=optimizer, warm_start=warm_start, address=address,
                          hidden_units=hidden_units,
                          strides=strides, global_averaging=global_averaging,
                         kernel_sizes=kernel_sizes, channels=channels, random_seed=666)
        elif mode=='conv_reg':
            model = CShapNN(mode='regression', batch_size=batch_size, max_epochs=max_iter,
                          learning_rate=learning_rate, 
                          weight_decay=weight_decay, validation_fraction=validation_fraction,
                          early_stopping=early_stopping,
                         optimizer=optimizer, warm_start=warm_start, address=address,
                          hidden_units=hidden_units,
                          strides=strides, global_averaging=global_averaging,
                         kernel_sizes=kernel_sizes, channels=channels, random_seed=666)
    elif 'NN' in mode:
        solver = kwargs.get('solver', 'adam')
        hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (20,))
        if isinstance(hidden_layer_sizes, list):
            hidden_layer_sizes = list(hidden_layer_sizes)
        activation = kwargs.get('activation', 'relu')
        learning_rate_init = kwargs.get('learning_rate', 0.001)
        max_iter = kwargs.get('max_iter', 5000)
        early_stopping= kwargs.get('early_stopping', False)
        warm_start = kwargs.get('warm_start', False)
        if mode=='NN':
            model = MLPClassifier(solver=solver, hidden_layer_sizes=hidden_layer_sizes,
                                activation=activation, learning_rate_init=learning_rate_init,
                                warm_start = warm_start, max_iter=max_iter,
                                early_stopping=early_stopping)
        if mode=='NN_reg':
            model = MLPRegressor(solver=solver, hidden_layer_sizes=hidden_layer_sizes,
                                activation=activation, learning_rate_init=learning_rate_init,
                                warm_start = warm_start, max_iter=max_iter, early_stopping=early_stopping)
    else:
        raise ValueError("Invalid mode!")
    return model



def generate_features(latent, dependency):

    features = []
    n = latent.shape[0]
    exp = latent
    holder = latent
    for order in range(1,dependency+1):
        features.append(np.reshape(holder,[n,-1]))
        exp = np.expand_dims(exp,-1)
        holder = exp * np.expand_dims(holder,1)
    return np.concatenate(features,axis=-1)  


def label_generator(problem, X, param, difficulty=1, beta=None, important=None):
        
    if important is None or important > X.shape[-1]:
        important = X.shape[-1]
    dim_latent = sum([important**i for i in range(1, difficulty+1)])
    if beta is None:
        beta = np.random.normal(size=[1, dim_latent])
    important_dims = np.random.choice(X.shape[-1], important, replace=False)
    funct_init = lambda inp: np.sum(beta * generate_features(inp[:,important_dims], difficulty), -1)
    batch_size = max(100, min(len(X), 10000000//dim_latent))
    y_true = np.zeros(len(X))
    while True:
        try:
            for itr in range(int(np.ceil(len(X)/batch_size))):
                y_true[itr * batch_size: (itr+1) * batch_size] = funct_init(
                    X[itr * batch_size: (itr+1) * batch_size])
            break
        except MemoryError:
            batch_size = batch_size//2
    mean, std = np.mean(y_true), np.std(y_true)
    funct = lambda x: (np.sum(beta * generate_features(
        x[:, important_dims], difficulty), -1) - mean) / std
    y_true = (y_true - mean)/std
    if problem is 'classification':
        y_true = logistic.cdf(param * y_true)
        y = (np.random.random(X.shape[0]) < y_true).astype(int)
    elif problem is 'regression':
        y = y_true + param * np.random.normal(size=len(y_true))
    else:
        raise ValueError('Invalid problem specified!')
    return beta, y, y_true, funct


def one_iteration(clf, X, y, X_test, y_test, mean_score, tol=0.0, c=None, metric='accuracy'):
    """Runs one iteration of TMC-Shapley."""
    
    if metric == 'auc':
        def score_func(clf, a, b):
            return roc_auc_score(b, clf.predict_proba(a)[:,1])
    elif metric == 'accuracy':
        def score_func(clf, a, b):
            return clf.score(a, b)
    else:
        raise ValueError("Wrong metric!")  
    if c is None:
        c = {i:np.array([i]) for i in range(len(X))}
    idxs, marginal_contribs = np.random.permutation(len(c.keys())), np.zeros(len(X))
    new_score = np.max(np.bincount(y)) * 1./len(y) if np.mean(y//1 == y/1)==1 else 0.
    start = 0
    if start:
        X_batch, y_batch =\
        np.concatenate([X[c[idx]] for idx in idxs[:start]]), np.concatenate([y[c[idx]] for idx in idxs[:start]])
    else:
        X_batch, y_batch = np.zeros((0,) +  tuple(X.shape[1:])), np.zeros(0).astype(int)
    for n, idx in enumerate(idxs[start:]):
        try:
            clf = clone(clf)
        except:
            clf.fit(np.zeros((0,) +  X.shape[1:]), y)
        old_score = new_score
        X_batch, y_batch = np.concatenate([X_batch, X[c[idx]]]), np.concatenate([y_batch, y[c[idx]]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                clf.fit(X_batch, y_batch)
                temp_score = score_func(clf, X_test, y_test)
                if temp_score>-1 and temp_score<1.: #Removing measningless r2 scores
                    new_score = temp_score
            except:
                continue
        marginal_contribs[c[idx]] = (new_score - old_score)/len(c[idx])
        if np.abs(new_score - mean_score)/mean_score < tol:
            break
    return marginal_contribs, idxs


def marginals(clf, X, y, X_test, y_test, c=None, tol=0., trials=3000, mean_score=None, metric='accuracy'):
    
    if metric == 'auc':
        def score_func(clf, a, b):
            return roc_auc_score(b, clf.predict_proba(a)[:,1])
    elif metric == 'accuracy':
        def score_func(clf, a, b):
            return clf.score(a, b)
    else:
        raise ValueError("Wrong metric!")  
    if mean_score is None:
        accs = []
        for _ in range(100):
            bag_idxs = np.random.choice(len(y_test), len(y_test))
            accs.append(score_func(clf, X_test[bag_idxs], y_test[bag_idxs]))
        mean_score = np.mean(accs)
    marginals, idxs = [], []
    for trial in range(trials):
        if 10*(trial+1)/trials % 1 == 0:
            print('{} out of {}'.format(trial + 1, trials))
        marginal, idx = one_iteration(clf, X, y, X_test, y_test, mean_score, tol=tol, c=c, metric=metric)
        marginals.append(marginal)
        idxs.append(idx)
    return np.array(marginals), np.array(idxs)

def shapley(mode, X, y, X_test, y_test, stop=None, tol=0., trials=3000, **kwargs):
    
    try:
        vals = np.zeros(len(X))
        example_idxs = np.random.choice(len(X), min(25, len(X)), replace=False)
        example_marginals = np.zeros((trials, len(example_idxs)))
        for i in range(trials):
            print(i)
            output = one_pass(mode, X, y, X_test, y_test, tol=tol, stop=stop, **kwargs)
            example_marginals[i] = output[0][example_idxs]
            vals = vals/(i+1) + output[0]/(i+1)
        return vals, example_marginals
    except KeyboardInterrupt:
        print('Interrupted!')
        return vals, example_marginals

def early_stopping(marginals, idxs, stopping):
    
    stopped_marginals = np.zeros_like(marginals)
    for i in range(len(marginals)):
        stopped_marginals[i][idxs[i][:stopping]] = marginals[i][idxs[i][:stopping]]
    return np.mean(stopped_marginals, 0)

def error(mem):
    
    if len(mem) < 100:
        return 1.0
    all_vals = (np.cumsum(mem, 0)/np.reshape(np.arange(1, len(mem)+1), (-1,1)))[-100:]
    errors = np.mean(np.abs(all_vals[-100:] - all_vals[-1:])/(np.abs(all_vals[-1:]) + 1e-12), -1)
    return np.max(errors)

def my_accuracy_score(clf, X, y):
    
    probs = clf.predict_proba(X)
    predictions = np.argmax(probs, -1)
    return np.mean(np.equal(predictions, y))

def my_f1_score(clf, X, y):
    
    predictions = clf.predict(x)
    if len(set(y)) == 2:
        return f1_score(y, predictions)
    return f1_score(y, predictions, average='macro')

def my_auc_score(clf, X, y):
    
    probs = clf.predict_proba(X)
    true_probs = probs[np.arange(len(y)), y]
    return roc_auc_score(y, true_probs)

def my_xe_score(clf, X, y):
    
    probs = clf.predict_proba(X)
    true_probs = probs[np.arange(len(y)), y]
    true_log_probs = np.log(np.clip(true_probs, 1e-12, None))
    return np.mean(true_log_probs)
    