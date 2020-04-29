import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import batch_norm
from net.layers import *
from net.utils import *


class PoseHead():

    def __init__(self, num_keypoints, final_conv_kernel):
        '''
        :param num_keypoints: number of keypoints to be detected
        '''

        self.num_keypoints = num_keypoints
        self.final_kernel = final_conv_kernel
        self.scope = 'CLS_HEAD'

    def forward(self, inputs):
        assert len(inputs) > 0, \
            "input_channel {} must to be at least 1".format(len(inputs))
        # final layer
        # nn.Conv2d(
        #             in_channels=pre_stage_channels[0],
        #             out_channels=cfg.MODEL.NUM_JOINTS,
        #             kernel_size=extra.FINAL_CONV_KERNEL,
        #             stride=1,
        #             padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        #         )
        #
        #
        with tf.variable_scope(self.scope):
            _final_output = slim.conv2d(inputs[0], num_outputs=self.num_keypoints,
                                        kernel_size=[self.final_kernel, self.final_kernel],
                                        stride=1, activation_fn=tf.nn.relu,
                                        normalizer_fn=None, padding='SAME')
        return _final_output

