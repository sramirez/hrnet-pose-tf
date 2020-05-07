import tensorflow as tf
from net.model import HRNet

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_bool('enbl_multi_gpu', False, 'Enable training with multiple gpus')
tf.app.flags.DEFINE_string('data_path', './data/tfrecord', 'path to data tfrecords')
tf.app.flags.DEFINE_string('net_cfg', '../cfgs/w30_s4.cfg', 'config file of network')
tf.app.flags.DEFINE_bool('eval_only', False, 'Eval mode')
tf.app.flags.DEFINE_bool('resume_training', False, 'resume training')

if __name__ == '__main__':
    input = tf.ones((4, 512, 512, 3)) # dimensions should be equal
    model = HRNet(FLAGS.net_cfg)
    output = model.forward_eval(input)
    target = tf.ones((4, 512 // 4, 512 // 4, output.get_shape()[3]))
    print(output)
    loss = model.joints_mse_loss(output, target)
    print(loss)