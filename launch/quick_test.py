import tensorflow as tf
from net.model import HRNet
from datasets.coco_keypoints_dataset import coco_keypoints_dataset
from netutils import config
from net.loss import JointsMSELoss

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_bool('enbl_multi_gpu', False, 'Enable training with multiple gpus')
tf.app.flags.DEFINE_string('data_path', './data/tfrecord', 'path to data tfrecords')
tf.app.flags.DEFINE_string('net_cfg', '../cfgs/w30_s4.cfg', 'config file of network')
tf.app.flags.DEFINE_bool('eval_only', False, 'Eval mode')
tf.app.flags.DEFINE_bool('resume_training', False, 'resume training')

def loss_test():
    output = tf.zeros((4, 512 // 4, 512 // 4, 17)) # dimensions should be equal
    target = tf.ones((4, 512 // 4, 512 // 4, 17))
    loss, _ = JointsMSELoss()(output, target)
    print(loss)
    with tf.Session() as sess:
        with tf.device("/cpu:0"): # or `with sess:` to close on exit
            out = sess.run(loss)
            assert out == 0.5
            print(out)

def quick_test():
    input = tf.ones((4, 512, 512, 3)) # dimensions should be equal
    model = HRNet(FLAGS.net_cfg)
    output = model.forward_eval(input)
    print(output)
    target = tf.ones((4, 512 // 4, 512 // 4, output.get_shape()[3]))
    loss, _ = JointsMSELoss()(output, target)
    print(loss)
    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            sess.run(loss)

def full_test():
    cfg = config.load_net_cfg_from_file(FLAGS.net_cfg)
    coco = coco_keypoints_dataset(cfg, "../data/coco/", cfg['DATASET']['test_set'], False)
    inputs = coco.build(subset=10)
    images, labels = inputs.get_next()
    model = HRNet(FLAGS.net_cfg)
    output = model.forward_eval(images)
    print(output)
    print(labels)
    loss, _ = JointsMSELoss()(output, labels)
    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            real_loss = sess.run(loss)
            print(real_loss)

if __name__ == '__main__':
    quick_test()