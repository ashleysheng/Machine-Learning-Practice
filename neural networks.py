import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


learning_rate = 0.001
wdConst = 0.0003
numOfHiddenUnitsLayerOne = 1000
startX = 0
endX = 80

def inputToHiddenUnits(inputs,numOfUnits): #(3500,28,28)    
    reShapedInputs = tf.reshape(inputs,[-1,784])  #(3500,784)
    Weights = tf.Variable(tf.random_normal(shape=[784,numOfUnits], stddev=tf.sqrt(3.0/(784+numOfUnits))), name='weights')#(784,1000)
    Weights = tf.cast(Weights, tf.float64)
    withoutBias = tf.matmul(reShapedInputs, Weights) #(3500,1000)
    biasRow = tf.zeros(shape=[1,numOfUnits], dtype =tf.float64)
    util = tf.constant(1, shape=[15000, numOfUnits], dtype =tf.float64)
    bias = tf.multiply(util, biasRow)
    return {'output1':(withoutBias + bias), 'weights1':Weights, 'bias1':biasRow}

def hiddenUnitsToOutput(inputs): #(3500,1000)
    Weights = tf.Variable(tf.random_normal(shape=[numOfHiddenUnitsLayerOne,10], stddev=tf.sqrt(3.0/(numOfHiddenUnitsLayerOne+10))), name='weights')#(1000,10)
    Weights = tf.cast(Weights, tf.float64)
    withoutBias = tf.matmul(inputs, Weights) #(3500,10)
    biasRow = tf.zeros([1,10], tf.float64)
    util = tf.constant(1, shape=[15000, 10], dtype =tf.float64)
    bias = tf.multiply(util, biasRow)
    return {'output2':(withoutBias + bias), 'weights2':Weights,'bias2':biasRow}

with np.load("notMNIST.npz") as data:
    Data, Target = data ["images"], data["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx]/255.
    Target = Target[randIndx]
    trainData, trainTarget = Data[:15000], Target[:15000]
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    testData, testTarget = Data[16000:], Target[16000:]
    # print trainData.shape


retFromFirstLayer = inputToHiddenUnits(trainData, numOfHiddenUnitsLayerOne);
afterRELU = tf.nn.relu(retFromFirstLayer['output1']); #(3500,1000)
w1=retFromFirstLayer['weights1']
b1Row=retFromFirstLayer['bias1']
retFromOutput = hiddenUnitsToOutput(afterRELU)
finalOutput = retFromOutput['output2']
w2=retFromOutput['weights2']
b2Row=retFromOutput['bias2']
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
	(labels=tf.one_hot(trainTarget,10), logits=finalOutput))
loss = cross_entropy + wdConst*tf.nn.l2_loss(w1) + wdConst*tf.nn.l2_loss(w2)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
init = tf.global_variables_initializer()


trainingCrossEntropyVector = []
trainingClassificationErrorVector = []
validCrossEntropyVector = []
validClassificationErrorVector = []
testCrossEntropyVector = []
testClassificationErrorVector = []


saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    # ckpt = tf.train.get_checkpoint_state('./')
    # if ckpt and ckpt.model_checkpoint_path:
    #     saver.restore(sess, ckpt.model_checkpoint_path)
    for epochIndex in range(startX,endX):
        sess.run(optimizer)
        correct_prediction = tf.equal(tf.argmax(finalOutput,1), trainTarget)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        trainingCrossEntropyVector.append(cross_entropy.eval())
        trainingClassificationErrorVector.append((1-accuracy.eval()))
        print epochIndex
        # valid
        valid_input = tf.reshape(validData,[-1,784])
        util1 = tf.constant(1, shape=[1000, numOfHiddenUnitsLayerOne], dtype =tf.float64)
        b1 = tf.multiply(util1, b1Row)
        valid_predicted_1 = tf.matmul(valid_input,w1)+b1;
        util2 = tf.constant(1, shape=[1000, 10], dtype =tf.float64)
        b2 = tf.multiply(util2, b2Row)
        valid_predicted_2 = tf.matmul(valid_predicted_1,w2)+b2;
        valid_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(validTarget,10), logits=valid_predicted_2))
        valid_correct_prediction = tf.equal(tf.argmax(valid_predicted_2,1), validTarget)
        valid_accuracy = tf.reduce_mean(tf.cast(valid_correct_prediction, "float"))
        valid_class_error = 1-valid_accuracy.eval()
        validCrossEntropyVector.append(valid_cross_entropy.eval())
        validClassificationErrorVector.append(valid_class_error)
        # test
        test_input = tf.reshape(testData,[-1,784])
        util1 = tf.constant(1, shape=[2724, numOfHiddenUnitsLayerOne], dtype =tf.float64)
        b1 = tf.multiply(util1, b1Row)
        test_predicted_1 = tf.matmul(test_input,w1)+b1;
        util2 = tf.constant(1, shape=[2724, 10], dtype =tf.float64)
        b2 = tf.multiply(util2, b2Row)
        test_predicted_2 = tf.matmul(test_predicted_1,w2)+b2;
        test_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(testTarget,10), logits=test_predicted_2))
        test_correct_prediction = tf.equal(tf.argmax(test_predicted_2,1), testTarget)
        test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, "float"))
        test_class_error = 1-test_accuracy.eval()
        testCrossEntropyVector.append(test_cross_entropy.eval())
        testClassificationErrorVector.append(test_class_error)
        if epochIndex ==12:
            print b1Row.eval()
        # if epochIndex == 79:
        #     saver.save(sess,'my-model-1000units-at-80')
    
    # w1 = tf.transpose(w1) #1000x784
    # w1 = w1[:100] #100x784
    # gmin = w1.eval().min()
    # gmax = w1.eval().max()

    # fig, ax = plt.subplots(10,10)
    # fig.subplots_adjust(wspace=0.01,hspace=0.1)
    # for a in range(10):
    #     for b in range(10):
    #         new = tf.reshape(w1[(a+10*b)],[28,28])
    #         new = tf.transpose(new)

    #         ax[a,b].matshow(new.eval(), cmap=plt.cm.gray, vmin=0.5*gmin, vmax=0.5*gmax)
    #         ax[a,b].axis('off')   

epoch = np.linspace(startX, endX, num = (endX-startX))
plt.figure()
plt.subplot(321)
plt.title('training set - cross entropy')
plt.plot(epoch,trainingCrossEntropyVector)

plt.subplot(322)
plt.title('training set - classification error')
plt.plot(epoch,trainingClassificationErrorVector)

plt.subplot(323)
plt.title('valid set - cross entropy')
plt.plot(epoch,validCrossEntropyVector)

plt.subplot(324)
plt.title('valid set - classification error')
plt.plot(epoch,validClassificationErrorVector)

plt.subplot(325)
plt.title('test set - cross entropy')
plt.plot(epoch,testCrossEntropyVector)

plt.subplot(326)
plt.title('test set - classification error')
plt.plot(epoch,testClassificationErrorVector)
plt.show()


