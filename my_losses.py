import tensorflow as tf

def coral_loss(fs, ft):
    """
    Deep CORAL loss between two feature batches.
    fs, ft: [B, D] float tensors
    """
    fs = tf.cast(fs, tf.float32)
    ft = tf.cast(ft, tf.float32)

    d = tf.cast(tf.shape(fs)[1], tf.float32)

    fs = fs - tf.reduce_mean(fs, axis=0, keepdims=True)
    ft = ft - tf.reduce_mean(ft, axis=0, keepdims=True)

    ns = tf.cast(tf.shape(fs)[0] - 1, tf.float32)
    nt = tf.cast(tf.shape(ft)[0] - 1, tf.float32)

    Cs = tf.matmul(fs, fs, transpose_a=True) / tf.maximum(ns, 1.0)
    Ct = tf.matmul(ft, ft, transpose_a=True) / tf.maximum(nt, 1.0)

    return tf.reduce_sum(tf.square(Cs - Ct)) / (4.0 * d * d)


def _gather_domain(feats, dom_ids, dom_value):
    idx = tf.where(tf.equal(dom_ids, dom_value))[:, 0]
    return tf.gather(feats, idx)


def _coral_if_enough(fs, ft):
    ns = tf.shape(fs)[0]
    nt = tf.shape(ft)[0]
    return tf.cond(
        tf.logical_and(ns >= 2, nt >= 2),
        lambda: coral_loss(fs, ft),
        lambda: tf.constant(0.0, tf.float32),
    )


def coral_multi_source_to_target(feats, dom_ids, target_id, source_ids):
  
    dom_ids = tf.cast(dom_ids, tf.int32)
    ft = _gather_domain(feats, dom_ids, tf.cast(target_id, tf.int32))

    total = tf.constant(0.0, tf.float32)
    count = tf.constant(0.0, tf.float32)

    # python loop is fine if source_ids is a small fixed list
    for sid in source_ids:
        fs = _gather_domain(feats, dom_ids, tf.cast(sid, tf.int32))
        l = _coral_if_enough(fs, ft)
        total += l
        count += tf.cast(l > 0.0, tf.float32)

    return total / tf.maximum(count, 1.0)
