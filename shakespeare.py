import tensorflow as tf
from tensorflow.contrib import layers
import os
import time
import math
import numpy as np
import my_txtutils as txt
tf.set_random_seed(0)


SEQLEN = 30
BATCHSIZE = 200
ALPHASIZE = txt.ALPHASIZE
INTERNALSIZE = 512
NLAYERS = 3
learning_rate = 0.001  # fixed learning rate
dropout_pkeep = 0.8    # some dropout

# load data, either shakespeare, or the Python source of Tensorflow itself
shakedir = "shakespeare/*.txt"
codetext, valitext, bookranges = txt.read_data_files(shakedir, validation=True)

# display some stats on the data
epoch_size = len(codetext) // (BATCHSIZE * SEQLEN)
txt.print_data_stats(len(codetext), len(valitext), epoch_size)

X = tf.placeholder(tf.int32, shape=[None, None], name='X')  # [BATCHSIZE, SEQLEN]
Xo = tf.one_hot(X, ALPHASIZE, 1.0, 0.0)                     # [BATCHSIZE, SEQLEN, ALPHASIZE]

Y = tf.placeholder(tf.int32, shape=[None, None], name='Y')  # [BATCHSIZE, SEQLEN]
Yo = tf.one_hot(Y, ALPHASIZE, 1.0, 0.0)                     # [BATCHSIZE, SEQLEN, ALPHASIZE]

p_keep = tf.placeholder(tf.float32, name='p_keep')

batch_size = tf.placeholder(tf.int32, name='batch_size')
# Hin = tf.zeros(shape=[batch_size, INTERNALSIZE*NLAYERS], name='Hin')

cells = [tf.nn.rnn_cell.LSTMCell(INTERNALSIZE) for _ in range(NLAYERS)]
drop_cells = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=p_keep) for cell in cells]
multi_cells = tf.nn.rnn_cell.MultiRNNCell(drop_cells)
multi_cells = tf.nn.rnn_cell.DropoutWrapper(multi_cells, output_keep_prob=p_keep)

Hin = tf.placeholder(dtype=tf.float32, shape=[NLAYERS, 2, None, INTERNALSIZE])
Hin_unstack = tuple([
    tf.nn.rnn_cell.LSTMStateTuple(Hin_per_layer[0], Hin_per_layer[1])
    for Hin_per_layer in tuple(tf.unstack(Hin, axis=0))
])
Yr, H = tf.nn.dynamic_rnn(multi_cells, Xo, dtype=tf.float32, initial_state=Hin_unstack)
# Yr [BATCHSIZE, SEQLEN, INTERNALSIZE]
# H  [NLAYERS, BATCHSIZE, INTERNALSIZE]

H = tf.identity(H, name='H')

Yr_flat = tf.reshape(Yr, shape=[-1, INTERNALSIZE])  # [BATCHSIZE x SEQLEN, INTERNALSIZE]
Y_logits = layers.linear(Yr_flat, ALPHASIZE)        # [BATCHSIZE x SEQLEN, ALPHASIZE]
Y_flat = tf.reshape(Yo, shape=[-1, ALPHASIZE])      # [BATCHSIZE x SEQLEN, ALPHASIZE]

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y_flat, logits=Y_logits)  # [BATCHSIZE x SEQLEN]
cross_entropy_flat = tf.reshape(cross_entropy, [batch_size, -1])  # [BATCHSIZE, SEQLEN]
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_flat)

Y_logits_sm = tf.nn.softmax(Y_logits)               # [BATCHSIZE x SEQLEN, ALPHASIZE]
Y_pred = tf.argmax(Y_logits_sm, 1)                  # [BATCHSIZE x SEQLEN]
Y_pred_flat = tf.reshape(Y_pred, [batch_size, -1])  # [BATCHSIZE, SEQLEN]


# stats for display
seqloss = tf.reduce_mean(cross_entropy_flat, 1)
batchloss = tf.reduce_mean(seqloss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_pred_flat, tf.cast(Y, tf.int64)), tf.float32))
loss_summary = tf.summary.scalar("batch_loss", batchloss)
acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
summaries = tf.summary.merge([loss_summary, acc_summary])

# Init Tensorboard stuff. This will save Tensorboard information into a different
# folder at each run named 'log/<timestamp>/'. Two sets of data are saved so that
# you can compare training and validation curves visually in Tensorboard.
timestamp = str(math.trunc(time.time()))
summary_writer = tf.summary.FileWriter("log/" + timestamp + "-training")
validation_writer = tf.summary.FileWriter("log/" + timestamp + "-validation")

# Init for saving models. They will be saved into a directory named 'checkpoints'.
# Only the last checkpoint is kept.
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1000)

# for display: init the progress bar
DISPLAY_FREQ = 50
_50_BATCHES = DISPLAY_FREQ * BATCHSIZE * SEQLEN
progress = txt.Progress(DISPLAY_FREQ, size=111+2, msg="Training on next "+str(DISPLAY_FREQ)+" batches")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

step = 0
Hin_current = np.zeros([NLAYERS, 2, BATCHSIZE, INTERNALSIZE])

# training loop
for x, y, epoch in txt.rnn_minibatch_sequencer(codetext, BATCHSIZE, SEQLEN, nb_epochs=40):
    # train on one minibatch
    feed_dict = {X: x, Y: y, p_keep: dropout_pkeep, batch_size: BATCHSIZE, Hin: Hin_current}
    _, h_current = sess.run([optimizer, H], feed_dict=feed_dict)

    if step % _50_BATCHES == 0:
        feed_dict = {X: x, Y: y, p_keep: 1, batch_size: BATCHSIZE, Hin: Hin_current}
        y, l, bl, acc, smm = sess.run([Y_pred_flat, seqloss, batchloss, accuracy, summaries], feed_dict=feed_dict)
        txt.print_learning_learned_comparison(x, y, l, bookranges, bl, acc, epoch_size, step, epoch)
        summary_writer.add_summary(smm, step)

    # display a short text generated with the current weights and biases (every 150 batches)
    if step // 3 % _50_BATCHES == 0:
        txt.print_text_generation_header()
        ry = np.array([[txt.convert_from_alphabet(ord("K"))]])
        Hin_gen = np.zeros([NLAYERS, 2, 1, INTERNALSIZE])
        for k in range(1000):
            ryo, Hin_gen = sess.run([Y_logits_sm, H], feed_dict={X: ry, p_keep: 1.0, batch_size: 1, Hin: Hin_gen})
            rc = txt.sample_from_probabilities(ryo, topn=10 if epoch <= 1 else 2)
            print(chr(txt.convert_to_alphabet(rc)), end="")
            ry = np.array([[rc]])
        txt.print_text_generation_footer()

    # save a checkpoint (every 500 batches)
    if step // 10 % _50_BATCHES == 0:
        saved_file = saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=step)
        print("Saved file: " + saved_file)

    # display progress bar
    progress.step(reset=step % _50_BATCHES == 0)

    Hin_current = h_current
    step += BATCHSIZE * SEQLEN
