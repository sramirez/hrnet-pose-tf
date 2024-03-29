import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import layers
from net.head import PoseHead
from net.stage import HRStage
from net.front import HRFront
from collections import Counter
import functools

from tensorflow.python.ops.init_ops import VarianceScaling


def he_normal_fanout(seed=None):
  """He normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with `stddev = sqrt(2 / fan_out)`
  where `fan_in` is the number of input units in the weight tensor.
  To keep aligned with official implementation
  """
  return VarianceScaling(
      scale=2., mode="fan_out", distribution="truncated_normal", seed=seed)


class HRNet():

    def __init__(self, cfg):
        self.stages = []
        self.cfg = cfg
        self._build_components()

    def forward_train(self, train_input):

        batch_norm_params = {'epsilon': 1e-5,
                             'scale': True,
                             'is_training': True,
                             'updates_collections': ops.GraphKeys.UPDATE_OPS}

        with slim.arg_scope([layers.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=he_normal_fanout(),
                                weights_regularizer=slim.l2_regularizer(self.cfg['NET']['weight_l2_scale'])):
                final_output = self._forward(train_input)

        return final_output

    def forward_eval(self, eval_input):
        batch_norm_params = {'epsilon': 1e-5,
                             'scale': True,
                             'is_training': False,
                             'updates_collections': ops.GraphKeys.UPDATE_OPS}

        with slim.arg_scope([layers.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d],
                                weights_regularizer=slim.l2_regularizer(self.cfg['NET']['weight_l2_scale'])):
                final_output = self._forward(eval_input)

        return final_output

    def model_summary(self):

        cnt = Counter()
        ops = ['ResizeNearestNeighbor', 'Relu', 'Conv2D']

        for op in tf.get_default_graph().get_operations():
            if op.type in ops:
                cnt[op.type] += 1

        print(cnt)

    def _build_components(self):

        front = HRFront(num_channels=self.cfg['FRONT']['num_channels'],
                        bottlenect_channels=self.cfg['FRONT']['bottlenect_channels'],
                        output_channels=[i * self.cfg['FRONT']['output_channels'] for i in range(1, 3)],
                        num_blocks=self.cfg['FRONT']['num_blocks'])
        self.stages.append(front)

        num_stages = self.cfg['NET']['num_stages']
        for i in range(num_stages):
            _key = 'S{}'.format(i + 2)
            _stage = HRStage(stage_id=i + 2,
                             num_modules=self.cfg[_key]['num_modules'],
                             num_channels=self.cfg['NET']['num_channels'],
                             num_blocks=self.cfg[_key]['num_blocks'],
                             num_branches=self.cfg[_key]['num_branches'],
                             last_stage=True if i == num_stages - 1 else False)

            self.stages.append(_stage)

        clshead = PoseHead(num_keypoints=self.cfg['HEAD']['num_keypoints'],
                           final_conv_kernel=self.cfg['HEAD']['final_conv_kernel'])

        self.stages.append(clshead)

    def _forward(self, input):

        _out = input
        for stage in self.stages:
            _out = stage.forward(_out)

        return _out

    def _get_num_parameters(self):
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        var_size = 0
        for _var in vars:
            _size = functools.reduce(lambda a, b : a*b , _var.shape)
            var_size += _size
        return var_size