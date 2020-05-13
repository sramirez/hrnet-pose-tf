from netutils.misc_utils import auto_barrier as auto_barrier_impl
from netutils.misc_utils import is_primary_worker as is_primary_worker_impl
from netutils.multi_gpu_wrapper import MultiGpuWrapper as mgw
from net.model import HRNet
from net.loss import JointsMSELoss
from datasets.coco_keypoints_dataset import coco_keypoints_dataset

from launch.utils import *
from core.function import validate
from timeit import default_timer as timer
import numpy as np
import os

FLAGS = tf.app.flags.FLAGS # args coming from main method

class Trainer():

    def __init__(self, netcfg):
        self.data_scope = 'DATA'
        self.model_scope = 'HRNET'
        # initialize network
        self.hrnet = HRNet(netcfg)
        # initialize training & evaluation subsets
        #self.dataset_train = coco_keypoints_dataset(self.hrnet.cfg, FLAGS.data_path,
        #                                            FLAGS.test_path, True) # TODO: change test_path by test_path
        self.dataset_eval = coco_keypoints_dataset(self.hrnet.cfg, FLAGS.data_path,
                                                   FLAGS.test_path, False)

        # learning rate
        self.lr_init = self.hrnet.cfg['COMMON']['lr_rate_init']
        self.model_path = '../models'
        self.log_path = '../logs'
        self.summ_step = self.hrnet.cfg['COMMON']['summary_step']
        self.save_step = self.hrnet.cfg['COMMON']['save_step']
        self.nb_iters_start = 0

    def build_graph(self, is_train):
        with tf.Graph().as_default():
            # TensorFlow session
            config = tf.ConfigProto( # TODO: remove before launching on server
                device_count = {'GPU': 0}
            )
            config.gpu_options.visible_device_list = str(
                mgw.local_rank() if FLAGS.enbl_multi_gpu else 0)  # pylint: disable=no-member
            sess = tf.Session(config=config)

            # data input pipeline
            with tf.variable_scope(self.data_scope):
                iterator = self.dataset_train.build(10) if is_train else self.dataset_eval.build(10) # TODO: revise subsetting
                images, labels, metas = iterator.get_next()
                if not isinstance(images, dict):
                    tf.add_to_collection('images_final', images)
                else:
                    tf.add_to_collection('images_final', images['image'])

            # model definition - primary model
            with tf.variable_scope(self.model_scope):
                # forward pass
                logits = self.hrnet.forward_train(images) if is_train else self.hrnet.forward_eval(images)
                if not isinstance(logits, dict):
                    tf.add_to_collection('logits_final', logits)
                else:
                    for value in logits.values():
                        tf.add_to_collection('logits_final', value)

                # loss & extra evaluation metrics
                loss, metrics = JointsMSELoss()(logits, labels)
                tf.summary.scalar('loss', loss)
                for key, value in metrics.items():
                    tf.summary.scalar(key, value)

                # optimizer & gradients
                if is_train:
                    self.global_step = tf.train.get_or_create_global_step()
                    lrn_rate, self.nb_iters_train = self.setup_lrn_rate(self.global_step)
                    optimizer = tf.train.MomentumOptimizer(lrn_rate, self.hrnet.cfg['COMMON']['momentum'])
                    if FLAGS.enbl_multi_gpu:
                        optimizer = mgw.DistributedOptimizer(optimizer)
                    grads = optimizer.compute_gradients(loss, self.trainable_vars)

            # TF operations & model saver
            if is_train:
                self.sess_train = sess
                with tf.control_dependencies(self.update_ops):
                    steps = optimizer.apply_gradients(grads, global_step=self.global_step)
                    self.train_op = steps
                self.summary_op = tf.summary.merge_all()
                self.sm_writer = tf.summary.FileWriter(logdir=self.log_path)
                self.log_op = [lrn_rate, loss] + list(metrics.values())
                self.log_op_names = ['lr', 'loss'] + list(metrics.keys())
                self.init_op = tf.variables_initializer(self.vars)
                if FLAGS.enbl_multi_gpu:
                    self.bcast_op = mgw.broadcast_global_variables(0)
                self.saver_train = self._create_saver()
            else:
                self.sess_eval = sess
                self.eval_op = [loss, logits, labels, metas] + list(metrics.values())
                self.eval_op_names = ['loss', 'logits', 'metas'] + list(metrics.keys())
                self.saver_eval = self._create_saver()
            print("Graph build finished.")

    def train(self):
        """Train a model and periodically produce checkpoint files."""

        # initialization
        self.sess_train.run(self.init_op)
        if FLAGS.resume_training:
            save_path = tf.train.latest_checkpoint(os.path.dirname(self.model_path + '/model.ckpt'))
            if FLAGS.fine_tune:
                # check if saver will restore only some layers
                self.saver_train = self._create_saver(only_head=True)
                self.saver_train.restore(self.sess_train, save_path)
                self.saver_train = self._create_saver(only_head=False) # then we should allow saving all layers
            else:
                self.saver_train.restore(self.sess_train, save_path)
            self.nb_iters_start = get_global_step_from_ckpt(save_path)

        if FLAGS.enbl_multi_gpu:
            self.sess_train.run(self.bcast_op)

        # train the model through iterations and periodically save & evaluate the model
        # one iteration corresponds with a single batch run (# epochs * # batches)
        time_prev = timer()
        last_performance = 0.0
        for idx_iter in range(self.nb_iters_start, self.nb_iters_train):
            # train the model
            if (idx_iter + 1) % self.summ_step != 0:
                self.sess_train.run(self.train_op)
            else:
                __, summary, log_rslt = self.sess_train.run([self.train_op, self.summary_op, self.log_op])
                if self.is_primary_worker('global'):
                    time_step = timer() - time_prev
                    self.__monitor_progress(summary, self.summ_step, log_rslt, idx_iter, time_step)
                    time_prev = timer()

            # save and eval the model at certain steps (at the last iteration too)
            if self.is_primary_worker('global') and \
                    (((idx_iter + 1) % self.save_step == 0) or (idx_iter + 1 == self.nb_iters_train)):
                last_performance = self._save_and_eval(saver=self.saver_train,
                                                       sess=self.sess_train, last_performance=last_performance)

    def eval(self):
        ckpt_path = self.__restore_model(self.saver_eval, self.sess_eval)
        tf.logging.info('restore from %s' % (ckpt_path))
        print("Starting evaluation process")
        # eval
        nb_iters = int(np.ceil(float(self.dataset_eval.num_images) / FLAGS.batch_size))
        eval_rslts = np.zeros((nb_iters, len(self.eval_op)))

        for idx_iter in range(nb_iters):
            eval_rslts[idx_iter], logits, labels, metas = self.sess_eval.run(self.eval_op)

        perf_indicator = validate(self.cfg, self.dataset_eval, outputs=logits, targets=labels, metas=metas,
                                  output_dir=self.log_path, writer_dict=None)

        for idx, name in enumerate(self.eval_op_names):
            tf.logging.info('%s = %.4e' % (name, np.mean(eval_rslts[:, idx])))

        return perf_indicator

    def __restore_model(self, saver, session):
        ckpt_path = tf.train.latest_checkpoint(self.model_path)
        saver.restore(session, ckpt_path)
        return ckpt_path

    def setup_lrn_rate(self, global_step):
        """Setup the learning rate (and number of training iterations)."""

        nb_epochs = 100
        idxs_epoch = [30, 60, 90]
        decay_rates = [1.0, 0.1, 0.01, 0.001]
        nb_epochs_rat = 1.0
        batch_size = FLAGS.batch_size * (1 if not FLAGS.enbl_multi_gpu else mgw.size())
        lrn_rate = setup_lrn_rate_piecewise_constant(global_step, self.lr_init, batch_size,
                                                     idxs_epoch, decay_rates, self.dataset_train.num_images)
        nb_iters = int(self.dataset_train.num_images * nb_epochs * nb_epochs_rat / batch_size)

        return lrn_rate, nb_iters

    def __monitor_progress(self, summary, summ_step, log_rslt, idx_iter, time_step):
        """Monitor the training progress.

        Args:
        * summary: summary protocol buffer
        * summ_step: step to write summary
        * log_rslt: logging operations' results
        * idx_iter: index of the training iteration
        * time_step: time step between two summary operations
        """

        # write summaries for TensorBoard visualization
        self.sm_writer.add_summary(summary, idx_iter)

        # compute the training speed
        speed = FLAGS.batch_size * summ_step / time_step
        if FLAGS.enbl_multi_gpu:
            speed *= mgw.size()

        # display monitored statistics
        log_str = ' | '.join(['%s = %.4e' % (name, value)
                              for name, value in zip(self.log_op_names, log_rslt)])
        tf.logging.info('iter #%d: %s | speed = %.2f pics / sec' % (idx_iter + 1, log_str, speed))

    def auto_barrier(self):
        """Automatically insert a barrier for multi-GPU training, or pass for single-GPU training."""

        auto_barrier_impl(self.mpi_comm)

    @classmethod
    def is_primary_worker(cls, scope='global'):
        """Check whether is the primary worker of all nodes (global) or the current node (local).

        Args:
        * scope: check scope ('global' OR 'local')

        Returns:
        * flag: whether is the primary worker
        """

        return is_primary_worker_impl(scope)

    @property
    def vars(self):
        """List of all global variables."""
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope)

    @property
    def trainable_vars(self):
        """List of all trainable variables."""
        trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope)
        if FLAGS.fine_tune:
            trainable = [x for x in trainable if 'HEAD' in x.name]
        return trainable

    @property
    def update_ops(self):
        """List of all update operations."""
        return tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope)

    def _create_saver(self, only_head=False):
        vars_to_restore = [x for x in self.vars if 'HEAD' not in x.name] if only_head else self.vars
        return tf.train.Saver(vars_to_restore)

    def _save_and_eval(self, saver, sess, last_performance):
        perf_indicator = self.eval()
        if last_performance <= perf_indicator: # only save better models
            # save model
            saver.save(sess, os.path.join(self.model_path, 'model.ckpt'),
                                  global_step=self.global_step)
        return perf_indicator

