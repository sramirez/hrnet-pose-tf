import tensorflow as tf
from launch.trainer import Trainer
from netutils.multi_gpu_wrapper import MultiGpuWrapper as mgw

from netutils.config import load_net_cfg_from_file

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_bool('enbl_multi_gpu', False, 'Enable training with multiple gpus')
tf.app.flags.DEFINE_string('data_path', '../data/coco', 'path to data tfrecords')
tf.app.flags.DEFINE_string('bbox_file', "person_detection_results/COCO_val2017_detections_AP_H_56_person.json", "JSON file for bounding boxes")
tf.app.flags.DEFINE_string('train_path', "train2017", "Subfolder inside datapath for training examples")
tf.app.flags.DEFINE_string('test_path', "val2017", "Subfolder inside datapath for validation examples")
tf.app.flags.DEFINE_string('net_cfg', './cfgs/w30_s4.cfg', 'config file of network')
tf.app.flags.DEFINE_bool('eval_only', False, 'Eval mode')
tf.app.flags.DEFINE_bool('resume_training', False, 'resume training')
tf.app.flags.DEFINE_integer('batch_size', 2, 'batch size')
tf.app.flags.DEFINE_bool('fine_tune', True, 'Fine-tune head layers (heatmap generation)')


def _load_cfg(self, cfgfile):
    self.cfg = load_net_cfg_from_file(cfgfile)


def main(args):
    """Main entry.

    Args:
    * unused_argv: unused arguments (after FLAGS is parsed)
    """
    tf.logging.set_verbosity(tf.logging.INFO)

    if FLAGS.enbl_multi_gpu:
        mgw.init()

    cfg = load_net_cfg_from_file(FLAGS.net_cfg)
    trainer = Trainer(netcfg=cfg)
    if FLAGS.eval_only:
        trainer.build_graph(is_train=False)
        trainer.eval()
    else:
        trainer.build_graph(is_train=True)
        trainer.train()


if __name__ == '__main__':
    tf.app.run()
