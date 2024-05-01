import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

class ShapNN(object):
    
    def __init__(self, mode, hidden_units=[100], learning_rate=0.001, 
                 dropout = 0., activation=None, initializer=None,
                 weight_decay=0.0001, optimizer='adam', batch_size=128,
                 warm_start=False, max_epochs=100, validation_fraction=0.1,
                 early_stopping=0, address=None, test_batch_size=1000,
                 random_seed=666):
        
        self.mode = mode
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.hidden_units = hidden_units
        self.initializer = initializer
        self.activation = activation
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.warm_start = warm_start
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.address = address
        self._extra_train_ops = []
        self.random_seed = random_seed
        self.is_built = False

    def prediction_cost(self, X_test, y_test, batch_size=None):
        
        if batch_size is None:
            batch_size = self.test_batch_size
        assert len(set(y_test)) == self.num_classes, 'Number of classes does not match!'
        with self.graph.as_default():
            losses = []
            idxs = np.arange(len(X_test))            
            batches = [idxs[k * batch_size: (k+1) * batch_size] 
                       for k in range(int(np.ceil(len(idxs)/batch_size)))]
            for batch in batches:
                losses.append(self.sess.run(self.prediction_loss, {self.input_ph:X_test[batch],
                                                                   self.labels:y_test[batch]}))
            return np.mean(losses)     
        
    def score(self, X_test, y_test, batch_size=None):
        
        if batch_size is None:
            batch_size = self.test_batch_size
        assert len(set(y_test)) == self.num_classes, 'Number of classes does not match!'
        with self.graph.as_default():
            scores = []
            idxs = np.arange(len(X_test))     
            batches = [idxs[k * batch_size: (k+1) * batch_size] 
                       for k in range(int(np.ceil(len(idxs)/batch_size)))]
            for batch in batches:
                scores.append(self.sess.run(self.prediction_score, {self.input_ph:X_test[batch],
                                                                   self.labels:y_test[batch]}))
            return np.mean(scores)
        
    def predict_proba(self, X_test, batch_size=None):
        
        if batch_size is None:
            batch_size = self.test_batch_size
        with self.graph.as_default():
            probs = []
            idxs = np.arange(len(X_test))     
            batches = [idxs[k * batch_size: (k+1) * batch_size] 
                       for k in range(int(np.ceil(len(idxs)/batch_size)))]
            for batch in batches:
                probs.append(self.sess.run(self.probs, {self.input_ph:X_test[batch]}))
            return np.concatenate(probs, axis=0)    
        
    def predict_log_proba(self, X_test, batch_size=None):
        
        if batch_size is None:
            batch_size = self.test_batch_size
        with self.graph.as_default():
            probs = []
            idxs = np.arange(len(X_test))            
            batches = [idxs[k * batch_size: (k+1) * batch_size] 
                       for k in range(int(np.ceil(len(idxs)/batch_size)))]
            for batch in batches:
                probs.append(self.sess.run(self.probs, {self.input_ph:X_test[batch]}))
            return np.log(np.clip(np.concatenate(probs), 1e-12, None))   
        
    def cost(self, X_test, y_test, batch_size=None):
        
        if batch_size is None:
            batch_size = self.batch_size
        with self.graph.as_default():
            losss = []
            idxs = np.arange(len(X_test))            
            batches = [idxs[k * batch_size: (k+1) * batch_size] 
                       for k in range(int(np.ceil(len(idxs)/batch_size)))]
            for batch in batches:
                losss.append(self.sess.run(self.prediction_loss, {self.input_ph:X_test[batch],
                                                                   self.labels:y_test[batch]}))
            return np.mean(losss)
    
    def predict(self, X_test, batch_size=None):
        
        if batch_size is None:
            batch_size = self.batch_size
        with self.graph.as_default():
            predictions = []
            idxs = np.arange(len(X_test))
            batches = [idxs[k * batch_size: (k+1) * batch_size] 
                       for k in range(int(np.ceil(len(idxs)/batch_size)))]
            for batch in batches:
                predictions.append(self.sess.run(self.predictions, {self.input_ph:X_test[batch]}))
            return np.concatenate(predictions)
        
    def fit(self, X, y, X_val=None, y_val=None, sources=None, max_epochs=None,
            batch_size=None, save=False, load=False, sample_weight=None,
            metric='accuracy'):
        
        self.num_classes = len(set(y))
        self.metric = metric
        if max_epochs is None:
            max_epochs = self.max_epochs
        if batch_size is None:
            batch_size = self.batch_size
        if not self.is_built:
            self.graph = tf.Graph() 
            with self.graph.as_default():
                config = tf.ConfigProto()
                config.gpu_options.allow_growth=True
                self.sess = tf.Session(config=config)
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            try:
                self.global_step = tf.train.create_global_step()
            except ValueError:
                self.global_step = tf.train.get_global_step()
            if not self.is_built:
                self._build_model(X, y)
                self.saver = tf.train.Saver()
            self._initialize()
            if len(X):
                if X_val is None and self.validation_fraction * len(X) > 2:
                    X_train, X_val, y_train, y_val, sample_weight, _ = train_test_split(
                        X, y, sample_weight, test_size=self.validation_fraction)
                else:
                    X_train, y_train = X, y
                self._train_model(X_train, y_train, X_val=X_val, y_val=y_val,
                                  max_epochs=max_epochs, batch_size=batch_size,
                                  sources=sources, sample_weight=sample_weight)
                if save and self.address is not None:
                    self.saver.save(self.sess, self.address)
            
    def _train_model(self, X, y, X_val, y_val, max_epochs, batch_size, 
                     sources=None, sample_weight=None):
        
        
        assert len(X)==len(y), 'Input and labels not the same size'
        self.history = {'metrics':[], 'idxs':[]}
        stop_counter = 0
        best_performance = None
        for epoch in range(max_epochs):
            vals_metrics, idxs = self._one_epoch(
                X, y, X_val, y_val, batch_size, sources=sources, sample_weight=sample_weight)
            self.history['idxs'].append(idxs)
            self.history['metrics'].append(vals_metrics)
            if self.early_stopping and X_val is not None:
                current_performance = np.mean(val_acc)
                if best_performance is None:
                    best_performance = current_performance
                if current_performance > best_performance:
                    best_performance = current_performance
                    stop_counter = 0
                else:
                    stop_counter += 1
                    if stop_counter > self.early_stopping:
                        break
        
    def _one_epoch(self, X, y, X_val, y_val, batch_size, sources=None, sample_weight=None):
        
        vals = []
        if sources is None:
            if sample_weight is None:
                idxs = np.random.permutation(len(X))
            else:
                idxs = np.random.choice(len(X), len(X), p=sample_weight/np.sum(sample_weight))    
            batches = [idxs[k*batch_size:(k+1) * batch_size]
                       for k in range(int(np.ceil(len(idxs)/batch_size)))]
            idxs = batches
        else:
            idxs = np.random.permutation(len(sources.keys()))
            batches = [sources[i] for i in idxs]
        for batch_counter, batch in enumerate(batches):
            self.sess.run(self.train_op, 
                          {self.input_ph:X[batch], self.labels:y[batch],
                           self.dropout_ph:self.dropout})
            if X_val is not None:
                if self.metric=='accuracy':
                    vals.append(self.score(X_val, y_val))
                elif self.metric=='f1':
                    vals.append(f1_score(y_val, self.predict(X_val)))
                elif self.metric=='auc':
                    vals.append(roc_auc_score(y_val, self.predict_proba(X_val)[:,1]))
                elif self.metric=='xe':
                    vals.append(-self.prediction_cost(X_val, y_val))
        return np.array(vals), np.array(idxs)
    
    def _initialize(self):
        
        uninitialized_vars = []
        if self.warm_start:
            for var in tf.global_variables():
                try:
                    self.sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninitialized_vars.append(var)
        else:
            uninitialized_vars = tf.global_variables()
        self.sess.run(tf.initializers.variables(uninitialized_vars))
        
    def _build_model(self, X, y):
        
        self.num_classes = len(set(y))
        if self.initializer is None:
            initializer = tf.initializers.variance_scaling(distribution='uniform')
        if self.activation is None:
            activation = lambda x: tf.nn.relu(x)
        self.input_ph = tf.placeholder(dtype=tf.float32, shape=(None,) + X.shape[1:], name='input')
        self.dropout_ph = tf.placeholder_with_default(
            tf.constant(0., dtype=tf.float32), shape=(), name='dropout')
        if self.mode=='regression':
            self.labels = tf.placeholder(dtype=tf.float32, shape=(None, ), name='label')
        else:
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None, ), name='label')
        x = tf.reshape(self.input_ph, shape=(-1, np.prod(X.shape[1:])))
        for layer, hidden_unit in enumerate(self.hidden_units):
            with tf.variable_scope('dense_{}'.format(layer)):
                x = self._dense(x, hidden_unit, dropout=self.dropout_ph, 
                           initializer=self.initializer, activation=activation)
        with tf.variable_scope('final'):
            self.prelogits = x
            self._final_layer(self.prelogits, self.num_classes, self.mode)
        self._build_train_op()
        
    def _build_train_op(self):
        
        """Build taining specific ops for the graph."""
        learning_rate = tf.constant(self.learning_rate, tf.float32) ##fixit
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.loss, trainable_variables)
        self.grad_flat = tf.concat([tf.reshape(grad, (-1, 1)) for grad in grads], axis=0)
        if self.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif self.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        elif self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables),
            global_step=self.global_step, name='train_step')
        train_ops = [apply_op] + self._extra_train_ops + tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        previous_ops = [tf.group(*train_ops)]
        with tf.control_dependencies(previous_ops):
            self.train_op = tf.no_op(name='train')   
        self.is_built = True
    
    def _final_layer(self, x, num_classes, mode):
        
        if mode=='regression':
            self.logits = self._dense(x, 1, dropout=self.dropout_ph)
            self.predictions = tf.reduce_sum(self.logits, axis=-1)
            regression_loss = tf.nn.l2_loss(self.predictions - self.labels) ##FIXIT
            self.prediction_loss = tf.reduce_mean(regression_loss, name='l2')
            residuals = self.predictions - self.labels
            var_predicted = tf.reduce_mean(residuals**2) - tf.reduce_mean(residuals)**2
            var_labels = tf.reduce_mean(self.labels**2) - tf.reduce_mean(self.labels)**2
            self.prediction_score = 1 - var_predicted/(var_labels + 1e-12)
        else:
            self.logits = self._dense(x, num_classes, dropout=self.dropout_ph)
            self.probs = tf.nn.softmax(self.logits)
            xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=tf.cast(self.labels, tf.int32))
            self.prediction_loss = tf.reduce_mean(xent_loss, name='xent')
            self.predictions = tf.argmax(self.probs, axis=-1, output_type=tf.int32)
            correct_predictions = tf.equal(self.predictions, self.labels)
            self.prediction_score = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        self.loss = self.prediction_loss + self._reg_loss()
                
    def _dense(self, x, out_dim, dropout=tf.constant(0.), initializer=None, activation=None):
        
        if initializer is None:
            initializer = tf.initializers.variance_scaling(distribution='uniform')
        w = tf.get_variable('DW', [x.get_shape()[1], out_dim], initializer=initializer)
        b = tf.get_variable('Db', [out_dim], initializer=tf.constant_initializer())
        x = tf.nn.dropout(x, 1. - dropout)
        if activation:
            x = activation(x)
        return tf.nn.xw_plus_b(x, w, b)
    
    def _reg_loss(self, order=2):
        """Regularization loss for weight decay."""
        losss = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0 or var.op.name.find(r'CW') > 0: ##FIXIT
                if order==2:
                    losss.append(tf.nn.l2_loss(var))
                elif order==1:
                    losss.append(tf.abs(var))
                else:
                    raise ValueError("Invalid regularization order!")
        return tf.multiply(self.weight_decay, tf.add_n(losss))


class CShapNN(ShapNN):
    
    def __init__(self, mode, hidden_units=[100], kernel_sizes=[], 
                 strides=None, channels=[], learning_rate=0.001, 
                 dropout = 0., activation=None, initializer=None, global_averaging=False,
                weight_decay=0.0001, optimizer='adam', batch_size=128, 
                warm_start=False, max_epochs=100, validation_fraction=0.1,
                early_stopping=0, address=None, test_batch_size=1000, random_seed=666):
        
        self.mode = mode
        self.test_batch_size = test_batch_size
        self.kernels = []#FIXIT
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        self.global_averaging = global_averaging
        assert len(channels)==len(kernel_sizes), 'Invalid channels or kernel_sizes'
        if strides is None:
            self.strides = [1] * len(kernel_sizes)
        else:
            self.strides = strides
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.initializer = initializer
        self.activation = activation
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.warm_start = warm_start
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.address = address
        self._extra_train_ops = []
        self.random_seed = random_seed
        self.graph = tf.Graph()
        self.is_built = False
        with self.graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess = tf.Session(config=config)
            
    def _conv(self, x, filter_size, out_filters, strides, activation=None):
        
        in_filters = int(x.get_shape()[-1])
        n = filter_size * filter_size * out_filters
        kernel = tf.get_variable(
            'DW', [filter_size, filter_size, in_filters, out_filters],
            tf.float32, initializer=tf.random_normal_initializer(
                stddev=np.sqrt(2.0/n)))
        self.kernels.append(kernel)
        x = tf.nn.conv2d(x, kernel, strides, padding='SAME')
        if activation:
            x = activation(x)
        return x
    
    def _stride_arr(self, stride):
        
        if isinstance(stride, int):
            return [1, stride, stride, 1]
        if len(stride)==2:
            return [1, stride[0], stride[1], 1]
        if len(stride)==4:
            return stride
        raise ValueError('Invalid value!')  
        
    def _build_model(self, X, y):
        
        
        if self.initializer is None:
            initializer = tf.initializers.variance_scaling(distribution='uniform')
        if self.activation is None:
            activation = lambda x: tf.nn.relu(x)
        self.input_ph = tf.placeholder(dtype=tf.float32, shape=(None,) + X.shape[1:], name='input')
        self.dropout_ph = tf.placeholder_with_default(
            tf.constant(0., dtype=tf.float32), shape=(), name='dropout')
        if self.mode=='regression':
            self.labels = tf.placeholder(dtype=tf.float32, shape=(None, ), name='label')
        else:
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None, ), name='label')
        if len(X.shape[1:]) == 2:
            x = tf.reshape(self.input_ph, [-1, X.shape[0], X.shape[1], 1])
        else:
            x = self.input_ph
        for layer, (kernel_size, channels, stride) in enumerate(zip(
            self.kernel_sizes, self.channels, self.strides)):
            with tf.variable_scope('conv_{}'.format(layer)):
                x = self._conv(x, kernel_size, channels, self._stride_arr(stride), activation=activation)
        if self.global_averaging:
            x = tf.reduce_mean(x, axis=(1,2))
        else:
            x = tf.reshape(x, shape=(-1, np.prod(x.get_shape()[1:])))
        for layer, hidden_unit in enumerate(self.hidden_units):
            with tf.variable_scope('dense_{}'.format(layer)):
                x = self._dense(x, hidden_unit, dropout=self.dropout_ph, 
                           initializer=self.initializer, activation=activation)
                
        with tf.variable_scope('final'):
            self.prelogits = x
            self._final_layer(self.prelogits, len(set(y)), self.mode)
        self._build_train_op()
