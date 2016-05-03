import os
import pyma
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from regression_generator import RegressionGenerator, regression_keys
from vae import VariationalAutoencoder
from sklearn import preprocessing
from tensorflow.models.rnn import rnn_cell, rnn

# Model parameters
input_size = 128
batch_size = 128
save_interval = 10000

def unit_scale(t):
    m, v = tf.nn.moments(t, axes=[0])
    return  (t - m)/(v + 1e-9)

class LossNet:
    def __init__(self, sess, input_size, batch_size=128, learning_rate=1e-4):
        self.inputs = tf.placeholder(tf.float32, shape=(None, input_size))
        network_architecture = \
                               dict(n_hidden_recog_1=500, # 1st layer encoder neurons
                                    n_hidden_recog_2=500, # 2nd layer encoder neurons
                                    n_hidden_gener_1=500, # 1st layer decoder neurons
                                    n_hidden_gener_2=500, # 2nd layer decoder neurons
                                    n_input=input_size,   # input size
                                    n_z=input_size)  # dimensionality of latent space
        self.ema = pyma.EMA(0.1)
        self.iteration = 0
        self.loss_vae = VariationalAutoencoder(sess, network_architecture,
                                               learning_rate=learning_rate,
                                               batch_size=batch_size)

    def step(self, minibatch):
        minibatch = np.asarray(minibatch)
        batch_size = minibatch.shape[0]
        minibatch_var = np.var(minibatch) + 1e-9
        normalized_minibatch = (minibatch - np.mean(minibatch)) / minibatch_var
        # normalized_minibatch = preprocessing.scale(minibatch)

        _, z_m, z_log_sigma_sq = self.loss_vae.step(normalized_minibatch)

        # Update our moving average
        ema_update = round(self.ema.compute(np.sum(z_m)), 5)
        var_update = 3.0*np.sqrt(np.abs(np.sum(z_log_sigma_sq)))
        self.iteration += 1

        loss_scaler = np.abs(ema_update) > var_update
        loss_scaler = loss_scaler.astype(float)
        # if self.iteration > 3000: # settling time
        #     # If this is over 3x the std deviation then allow info flow
        #     loss_scaler = ema_update > var_update
        #     loss_scaler = loss_scaler.astype(float)
        # else:
        #     loss_scaler = 1.0

        print ' ema update: ', '{:.2f}'.format(ema_update), \
            ' | 3*sigma: ', '{:.2f}'.format(var_update), \
            ' | loss_scaler: ', loss_scaler,
        return loss_scaler

class Regression:
    def __init__(self, sess, input_size, hidden_sizes, batch_size=128,
                 learning_rate=1e-3, activation=tf.nn.elu, loss='l2'):
        self.input_size = self.output_size = input_size
        self.activation = activation
        self.loss_type = loss
        self.iteration = 0
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.activation_type = activation.__name__
        self.sizes = hidden_sizes + [input_size]
        self.inputs = tf.placeholder(tf.float32, shape=(None, input_size))
        self.targets = tf.placeholder(tf.float32, shape=(None, input_size))
        self.loss_scaler = tf.placeholder(tf.float32)#tf.placeholder(tf.float32, shape=(None, input_size))
        self.dropout_proba = tf.placeholder(tf.float32)
        self.current_model = tf.placeholder(tf.float32) # just for tensorboard

        self.inference = self.build_model() # self.build_lstm_model()
        self.cost = self.get_loss(loss)
        #self.cost_scaler = self.loss_net(self.cost, [input_size*2, input_size, input_size*2])
        self.updated_cost = self.loss_scaler * self.cost
        self.reduced_updated_cost = tf.reduce_sum(self.updated_cost)
        self.reduced_cost = tf.reduce_sum(self.cost)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.reduced_updated_cost)
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.reduced_updated_cost)

        # summaries and savers
        self.summary_cost = tf.scalar_summary("cost", self.reduced_cost)
        self.summary_cost = tf.scalar_summary("cost_scalar", self.loss_scaler)
        self.summary_cost = tf.scalar_summary("current_model", self.current_model)
        #self.summary_cost_scale = tf.histogram_summary("cost_scaler", self.loss_scaler)
        self.summary_cost_updated = tf.scalar_summary("cost_updated", self.reduced_updated_cost)
        self.summaries = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter("logs/" + self.get_name() + self.get_formatted_datetime(),
                                                     sess.graph)
        self.saver = tf.train.Saver()

    def get_formatted_datetime(self):
        return str(datetime.datetime.now()).replace(" ", "_") \
                                           .replace("-", "_") \
                                           .replace(":", "_")

    # http://web.as.uky.edu/statistics/users/pbreheny/764-F11/notes/12-1.pdf
    def get_loss(self, name, delta=1):
        filt = name.strip().lower()
        if filt == 'mae':
            return tf.abs(self.inference - self.targets)
        elif filt == 'l2':
            return tf.square(self.inference - self.targets)
        elif filt == 'Huber':
            # L_\delta (a) = \delta^2(\sqrt{1+(a/\delta)^2}-1).
            delta_sq = delta * delta
            return delta_sq * tf.sqrt(1 + tf.square((self.inference - self.targets) / delta_sq)) - 1
        else:
            raise Exception("unknown loss function")

    def build_lstm_model(self):
        # r = rnn_cell.LSTMCell(tf.split(0, self.batch_size, self.inputs), self.input_size,
        #                       initializer=tf.contrib.layers.xavier_initializer())
        r = rnn_cell.BasicLSTMCell(self.input_size)
        istate = r.zero_state(1, dtype=tf.float32)
        o, s = rnn.rnn(r, tf.split(0, self.batch_size, self.inputs), istate)
        return o[-1]

    def build_model(self):
        self.layers = [tf.contrib.layers.fully_connected(self.inputs,
                                                         self.input_size,
                                                         activation_fn=self.activation)]
                                                         #weight_regularizer=tf.contrib.layers.l2_regularizer(0.1))]
        for out_size in self.sizes[:-1]:
            self.layers.append(tf.nn.dropout(tf.contrib.layers.fully_connected(self.layers[-1],
                                                                               out_size,
                                                                               activation_fn=self.activation), keep_prob=self.dropout_proba))
            # self.layers.append(tf.contrib.layers.fully_connected(self.layers[-1],
            #                                                      out_size,
            #                                                      activation_fn=self.activation))


        # Last layer needs to be linear
        self.layers.append(tf.contrib.layers.fully_connected(self.layers[-1],
                                                             self.sizes[-1]))
        return self.layers[-1]

    def infer(self, sess, generator, loss_scalar, current_model,
              batch_size, summary_interval=100, model_save_interval=10000):
        inputs, targets = generator.get_batch_iter(batch_size)
        targets = np.roll(targets, 1, axis=0)
        feed_dict = {self.inputs: inputs,
                     self.loss_scaler: loss_scalar,
                     self.current_model: current_model,
                     self.dropout_proba: 0.5,
                     self.targets: targets}
        if self.iteration % summary_interval == 0:
            _, c, uc, cv, i, s = sess.run([self.optimizer, self.reduced_cost,
                                           self.reduced_updated_cost, self.cost, self.inference,
                                           self.summaries],
                                          feed_dict=feed_dict)
            self.summary_writer.add_summary(s, self.iteration)
        else:
            _, c, uc, cv, i = sess.run([self.optimizer, self.reduced_cost,
                                        self.reduced_updated_cost, self.cost, self.inference],
                                       feed_dict=feed_dict)

        if self.iteration !=0 and self.iteration % model_save_interval == 0:
            self.save(sess)

        self.iteration += 1
        print 'func = %s | loss = %.3f | updated_cost = %.1f' % (str(generator.function), c, uc)
        return c, uc, cv, i, targets

    def predict(self, sess, generator, loss_scalar, current_model,
                batch_size, summary_interval=100):
        inputs, targets = generator.get_batch_iter(batch_size)
        feed_dict = {self.inputs: inputs,
                     self.loss_scaler: loss_scalar,
                     self.current_model: current_model,
                     self.dropout_proba: 1.0,
                     self.targets: targets}
        c, uc, cv, i, s = sess.run([self.reduced_cost,
                                    self.reduced_updated_cost, self.cost, self.inference,
                                    self.summaries],
                                   feed_dict=feed_dict)
        self.summary_writer.add_summary(s, self.iteration)
        print 'func = %s | loss = %.3f | updated_cost = %.1f' % (str(generator.function), c, uc)
        return c, uc, cv, i, targets

    def restore(self, sess):
        model_filename = 'models/%s' % self.get_name()
        if os.path.isfile(model_filename):
            print 'restoring model from %s..' % model_filename
            self.saver.restore(sess, model_filename)

    def get_name(self):
        hidden_str = str(self.sizes).strip().lower().replace('[', '')\
                                                    .replace(']', '')\
                                                    .replace(' ', '')\
                                                    .replace(',', '_')
        return "model_%d_%s_%s_%s.ckpt" % (self.input_size,
                                           hidden_str,
                                           self.loss_type,
                                           self.activation_type)

    def save(self, sess):
        self.saver.save(sess, "models/%s" % self.get_name())

def main():
    # build some generators
    generators = [RegressionGenerator(w, input_size, sequential=False)
                  for w in regression_keys]

    with tf.device("/gpu:0"):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # with tf.device("/cpu:0"):
    #     with tf.Session(config=tf.ConfigProto(allow_soft_placement=False)) as sess:
            model = Regression(sess, input_size, [512*11, 4096*11])
            ls = LossNet(sess, input_size, batch_size=batch_size)
            sess.run(tf.initialize_all_variables())

            current_model = 0; iteration = 0
            loss_scaler = np.ones([batch_size, input_size])
            loss_scaler = 1.0
            current_rnd_iter = np.random.randint(57, high=79)

            if not os.path.isfile("models/%s" % model.get_name()):
                while True:
                    try:
                        if iteration != 0 and iteration % current_rnd_iter == 0:
                            current_rnd_iter = np.random.randint(57, high=79)
                            current_model = np.random.randint(len(regression_keys))

                        _, _, cv, _, _ = model.infer(sess, generators[current_model],
                                                     loss_scaler, current_model, batch_size)
                        loss_scaler = ls.step(cv)
                        iteration += 1

                    except KeyboardInterrupt:
                        print 'terminating as per user request'
                        model.save(sess)
                        break
            else:
                model.restore(sess)
                # for i in range(len(generators)):
                #     _, _, _, inference, targets = model.infer(sess, generators[i], 0.0, i, 1)
                #     plt.plot(targets.flatten())
                #     plt.plot(inference.flatten())
                #     plt.show()

                mse = []
                for i in range(save_interval * 2):
                    if iteration != 0 and iteration % current_rnd_iter == 0:
                        current_rnd_iter = np.random.randint(57, high=79)
                        current_model = np.random.randint(len(regression_keys))

                    c, _, _, inference, targets = model.infer(sess, generators[current_model],
                                                              loss_scaler, current_model, 1)
                    iteration += 1

                    plt.plot(targets.flatten())
                    plt.plot(inference.flatten())
                    plt.show()

                    # save the mean mse
                    mse.append(c)
                    if(len(mse) >= save_interval):
                        with open("mean_mse_regression.csv",'a') as f_handle:
                            np.savetxt(f_handle, mse, delimiter=",")
                            del mse[:]


if __name__ == '__main__':
    if not os.path.exists('models'):
            os.makedirs('models')
    main()
