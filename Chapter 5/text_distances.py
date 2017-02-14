# Text Distances
#----------------------------------
#
# This function illustrates how to use
# the Levenstein distance (edit distance)
# in Tensorflow.

import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start Graph Session
sess = tf.Session()

#----------------------------------
# First compute the edit distance between 'bear' and 'beers'
hypothesis = list('bear')
truth = list('beers')
h1 = tf.SparseTensor([[0,0,0], [0,0,1], [0,0,2], [0,0,3]],
                     hypothesis,
                     [1,1,1])

t1 = tf.SparseTensor([[0,0,0], [0,0,1], [0,0,2], [0,0,3],[0,0,4]],
                     truth,
                     [1,1,1])

print(sess.run(tf.edit_distance(h1, t1, normalize=False)))

#----------------------------------
# Compute the edit distance between ('bear','beer') and 'beers':
hypothesis2 = list('bearbeer')
truth2 = list('beersbeers')
h2 = tf.SparseTensor([[0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,1,0], [0,1,1], [0,1,2], [0,1,3]],
                     hypothesis2,
                     [1,2,4])

t2 = tf.SparseTensor([[0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,0,4], [0,1,0], [0,1,1], [0,1,2], [0,1,3], [0,1,4]],
                     truth2,
                     [1,2,5])

print(sess.run(tf.edit_distance(h2, t2, normalize=True)))

#----------------------------------
# Now compute distance between four words and 'beers' more efficiently:
hypothesis_words = ['bear','bar','tensor','flow']
truth_word = ['beers']

num_h_words = len(hypothesis_words)
h_indices = [[xi, 0, yi] for xi,x in enumerate(hypothesis_words) for yi,y in enumerate(x)]
h_chars = list(''.join(hypothesis_words))

h3 = tf.SparseTensor(h_indices, h_chars, [num_h_words,1,1])

truth_word_vec = truth_word*num_h_words
t_indices = [[xi, 0, yi] for xi,x in enumerate(truth_word_vec) for yi,y in enumerate(x)]
t_chars = list(''.join(truth_word_vec))

t3 = tf.SparseTensor(t_indices, t_chars, [num_h_words,1,1])

print(sess.run(tf.edit_distance(h3, t3, normalize=True)))

#----------------------------------
# Now we show how to use sparse tensors in a feed dictionary

# Create input data
hypothesis_words = ['bear','bar','tensor','flow']
truth_word = ['beers']

def create_sparse_vec(word_list):
    num_words = len(word_list)
    indices = [[xi, 0, yi] for xi,x in enumerate(word_list) for yi,y in enumerate(x)]
    chars = list(''.join(word_list))
    return(tf.SparseTensorValue(indices, chars, [num_words,1,1]))

hyp_string_sparse = create_sparse_vec(hypothesis_words)
truth_string_sparse = create_sparse_vec(truth_word*len(hypothesis_words))

hyp_input = tf.sparse_placeholder(dtype=tf.string)
truth_input = tf.sparse_placeholder(dtype=tf.string)

edit_distances = tf.edit_distance(hyp_input, truth_input, normalize=True)

feed_dict = {hyp_input: hyp_string_sparse,
             truth_input: truth_string_sparse}
             
print(sess.run(edit_distances, feed_dict=feed_dict))