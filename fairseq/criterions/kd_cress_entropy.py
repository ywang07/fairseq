# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('kd_cross_entropy')
class KnowledgeDistillationCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.T = args.kd_temperature
        self.alpha = args.kd_trade_off

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--kd-temperature', default=1., type=float, metavar='T',
                            help='the temperature acting in softmax computation during KD process')
        parser.add_argument('--kd-trade-off', default=1., type=float, metavar='A',
                            help='the trade-off parameter between kd loss and ce loss')

    def forward(self, model, sample, reduce=True, teacher_outputs=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        if teacher_outputs:
            assert torch.equal(target.size(), teacher_outputs.size())

        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]

        teacher_lprobs = F.softmax(teacher_outputs[0]/self.T, dim=-1)
        teacher_lprobs = teacher_lprobs.view(-1, teacher_lprobs.size(-1))
        kd_loss = nn.KLDivLoss()(lprobs, teacher_lprobs)[non_pad_mask]

        if reduce:
            nll_loss = nll_loss.sum()
            kd_loss= kd_loss.sum()
        loss = (1. - self.alpha) * nll_loss + self.alpha * self.T * self.T * kd_loss

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'sample_size': sample_size,
        }
