import tensorflow as tf


class JointsMSELoss(object):

    def __adapt_output(self, x):
        batch_size = tf.shape(x)[0]
        num_joints = tf.shape(x)[-1]
        out = tf.transpose(x, perm=[3, 0, 1, 2])
        out = tf.reshape(out, (num_joints, batch_size, -1))
        return out

    def __call__(self, out, gt): # TODO: apply class weights (keypoints)
        hout = self.__adapt_output(out)
        hgt = self.__adapt_output(gt)
        diff = tf.square(tf.subtract(hout, hgt)) * 0.5
        loss = tf.reduce_mean(tf.reduce_mean(diff), name='loss')
        metrics = {'mse_loss': loss}
        return loss, metrics
