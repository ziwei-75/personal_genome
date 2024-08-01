import tensorflow as tf
import tensorflow_probability as tfp


#from https://github.com/kundajelab/basepair/blob/cda0875571066343cdf90aed031f7c51714d991a/basepair/losses.py#L87
def multinomial_nll(true_counts, logits):
    """Compute the multinomial negative log-likelihood
    Args:
      true_counts: observed count values
      logits: predicted logit values
    """
    counts_per_example = tf.reduce_sum(true_counts, axis=-1)
    dist = tfp.distributions.Multinomial(total_count=counts_per_example,
                                         logits=logits)
    return (-tf.reduce_sum(dist.log_prob(true_counts)) / 
            tf.cast(tf.shape(true_counts)[0], dtype=tf.float32))




def multi_class_multinomial_nll(y_true, y_pred):
    out_pred_len=y_pred.shape[1]
    num_tasks=y_pred.shape[2]
    y_true = tf.ensure_shape(y_true,[None,out_pred_len, num_tasks])
    y_true = tf.unstack(y_true,axis=-1)
    y_pred = tf.unstack(y_pred,axis=-1)
    loss = [multinomial_nll(lab_pre_pair[0],lab_pre_pair[1]) for lab_pre_pair in zip(y_true, y_pred)]
    return tf.reduce_mean(loss)
    

def weighted_mse_wrapper(counts_loss_weight):
    def weighted_mse(y_true, y_pred):
        loss = tf.math.squared_difference(y_true, y_pred)
        loss = loss * counts_loss_weight
        return tf.reduce_mean(loss)
    return weighted_mse