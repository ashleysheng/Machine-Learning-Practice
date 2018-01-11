from __future__ import division

import numpy as np
import tensorflow as tf

from utils import reduce_logsumexp
from utils import logsoftmax

import matplotlib.pyplot as plt
LEARNINGRATE = 0.004

def normalDistributionGenerator(shape):
    temp = tf.random_normal(shape, 0.25)
    return tf.Variable(temp)

def lossFunction(X,mu):
    # X is B x D
    # mu is K x D
    X_2 = tf.mul(X,X)
    mu_transposed = tf.transpose(mu)
    mu_2 = tf.mul(mu_transposed,mu_transposed)
    X_2 = tf.reshape(tf.reduce_sum(X_2, 1), [-1, 1])
    mu_2 = tf.reshape(tf.reduce_sum(mu_2, 0), [1, -1])
    X_mu_2 = tf.mul(2.0, tf.matmul(X, mu_transposed))
    distanceMatrix = X_2+mu_2-X_mu_2 # B x K
    minDistance = tf.reduce_min(distanceMatrix, 1) # B x 1.  pick the closest cluster center
    closestClusterCentre = tf.argmin(distanceMatrix, 1)
    return tf.reduce_sum(minDistance,0), closestClusterCentre # sum all l2 distances together

def logGaussianDensity(X,mu,sigma_squared):
    # X is B x D
    # mu is K x D
    # sigma_squared is 1 x K
    X_2 = tf.mul(X,X)
    mu_transposed = tf.transpose(mu)
    mu_2 = tf.mul(mu_transposed,mu_transposed)
    X_2 = tf.reshape(tf.reduce_sum(X_2, 1), [-1, 1])
    mu_2 = tf.reshape(tf.reduce_sum(mu_2, 0), [1, -1])
    X_mu_2 = tf.mul(2.0, tf.matmul(X, mu_transposed))
    distanceMatrix = X_2+mu_2-X_mu_2 # B x K
    log_term = - tf.log(tf.sqrt(2*np.pi*sigma_squared)) # 1 x K
    return log_term - tf.mul(distanceMatrix, tf.reciprocal(sigma_squared)) * 0.5 #BxK

def logZGivenX(X,mu,sigma_squared,piK):
    # log P(z|x) = log P(x|z) + log P(z) - log P(x)
    logXGivenZ = logGaussianDensity(X,mu,sigma_squared) # B x K
    logZ = tf.log(piK) # 1 x K 
    logZ_X = logZ + logXGivenZ # B x K
    logX = reduce_logsumexp(logZ_X, 1,keep_dims = True)
    return logZ_X-logX, logX,logZ_X

def initialize_variance(shape):
    initial = tf.random_normal(shape, mean=0, stddev=0.5) 
    temp = tf.Variable(initial)
    return tf.exp(temp) 

def initialize_pi(shape):
    temp = tf.ones(shape)
    log = logsoftmax(temp)
    return tf.exp(log)

def negativeLogLikelihood(X, mu, var, pi):
    # X   is N X D
    # mu  is K x D
    # var is 1 x K
    # pi  is 1 x K
    _,logX,logZ_X = logZGivenX(X,mu,var,pi);
    loss = tf.reduce_sum(logX, 0) # 1 x 1
    return -loss # 1 x 1

def MoG(K):
    data = np.load('data2D.npy').astype(np.float32) # 10000x2
    training_data   = data[:7000]
    validation_data = data[7001:]

    N = training_data.shape[0]
    D = training_data.shape[1]
    X = tf.placeholder("float", shape=(None, D)) # N x D
    mu = normalDistributionGenerator((K,D))
    var = initialize_variance((1, K)) # 1 x K
    pi = initialize_pi((1, K)) # 1 x K
    L = negativeLogLikelihood(X, mu, var, pi)
    optimizer = tf.train.AdamOptimizer(LEARNINGRATE, beta1=0.9, beta2=0.99, epsilon=1e-5)
    train_op = optimizer.minimize(L)
    # Determine cluster assignments
    _,__,logZ_X = logZGivenX(X, mu, var, pi) # N x K
    assg =  tf.argmax(logZ_X, 1)
    init = tf.global_variables_initializer()
    loss_results = np.zeros(700)
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(700):
            loss, _ ,cluster_assignments= sess.run([L, train_op,assg], feed_dict={X: training_data})
            loss_results[epoch] = loss
    colors = iter(plt.cm.rainbow(np.linspace(0,1,K)))
    for i in range(K):
        cluster_data = training_data[cluster_assignments == i].T
        plt.scatter(cluster_data[0], cluster_data[1], color=next(colors))
    plt.show()
    return loss_results

for counter in range(5):
    loss_results = MoG(counter+1)  
    plt.plot(np.arange(700), loss_results)
    plt.show()


def k_mean(K):
    data = np.load('data2D.npy').astype(np.float32) # 10000x2
    training_data   = data[:7000]
    validation_data = data[7001:]
    D = data.shape[1]
    mu = normalDistributionGenerator((K,D))
    X = tf.placeholder("float", shape=(None, D))
    loss, closestClusterCentre= lossFunction(X,mu)
    optimizer = tf.train.AdamOptimizer(LEARNINGRATE, beta1=0.9, beta2=0.99, epsilon=1e-5)
    train_op = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    num_of_epochs = 500
    loss_vector = np.zeros(num_of_epochs)

    with tf.Session() as sess:
        sess.run(init)
        for epochIndex in range(num_of_epochs):
            l, temp,cluster_centres = sess.run([loss, train_op,closestClusterCentre], feed_dict={X: training_data})
            # loss_vector[epochIndex] = l
        validation_loss = sess.run(loss,feed_dict={X: validation_data})
        print validation_loss

    colors = iter(plt.cm.rainbow(np.linspace(0,1,K)))
    for i in range(K):
        # print cluster_centres
        cluster_data = training_data[cluster_centres == i].T
        # print float(cluster_data.shape[1]/10000)
        plt.scatter(cluster_data[0], cluster_data[1], color=next(colors))
    # plt.show()
    # plt.plot(np.arange(num_of_epochs), loss_vector)
    plt.show()

# for counter in range(5):
#     k_mean(counter+1)  
# plt.show()