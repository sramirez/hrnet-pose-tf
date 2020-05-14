# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np

from core.evaluate import accuracy
from core.inference import get_final_preds
from datasets.joints_dataset import joints_dataset
#from utils.vis import save_debug_images

logger = logging.getLogger(__name__)

# TODO: do validation with flip
def validate(config, dataset, outputs, targets, ids, output_dir, writer_dict=None):
    losses = AverageMeter()
    acc = AverageMeter()
    num_samples = dataset.num_images
    all_preds = np.zeros(
        (num_samples, config['HEAD']['num_keypoints'], 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    idx = 0
    for i, (output, target, id) in enumerate(zip(outputs, targets, ids)):

        num_images = output.size(0)
        meta = dataset.augmented_db[str(id)]
        # measure accuracy and record loss
        _, avg_acc, cnt, pred = accuracy(output,
                                         target)
        acc.update(avg_acc, cnt)
        c = meta['center'].numpy()
        s = meta['scale'].numpy()
        score = meta['score'].numpy()
        preds, maxvals = get_final_preds(config, output.numpy(), c, s)

        # Prepare final data structure
        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
        # double check this all_boxes parts
        all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
        all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
        all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
        all_boxes[idx:idx + num_images, 5] = score

        idx += num_images
        image_path.extend(meta['image'])
        name_values, perf_indicator = dataset.evaluate(all_preds, output_dir, all_boxes, image_path)

        model_name = config['MODEL']['name']
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values + 1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
