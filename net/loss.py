import tensorflow as tf


class JointsMSELoss(object):

    def __adapt_output(self, x):
        batch_size = x.get_shape()[0]
        num_joints = x.get_shape()[3]
        out = tf.transpose(x, perm=[3, 0, 1, 2])
        out = tf.reshape(out, [num_joints, batch_size, -1])
        return out

    def __call__(self, out, gt):
        num_joints = out.get_shape()[3]
        heatmaps_pred = self.__adapt_output(out)
        heatmaps_gt = self.__adapt_output(gt)
        loss = 0.0

        for idx in range(num_joints):
            heatmap_pred = tf.squeeze(heatmaps_pred[idx])
            heatmap_gt = tf.squeeze(heatmaps_gt[idx])
            loss += 0.5 * tf.losses.mean_squared_error(heatmap_gt, heatmap_pred,
                                                       reduction=tf.losses.Reduction.MEAN)

        return tf.div(loss, tf.cast(num_joints, tf.float32))
