#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
BiEncoder component + loss function for 'all-in-batch' training
"""

import collections
import logging
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

from dpr.data.biencoder_data import BiEncoderSample
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import CheckpointState

logger = logging.getLogger(__name__)

BiEncoderBatch = collections.namedtuple(
    "BiENcoderInput",
    [
        "question_ids",
        "question_segments",
        "context_ids",
        "ctx_segments",
        "is_positive",
        "hard_negatives",
        "encoder_type",
    ],
)
# TODO: it is only used by _select_span_with_token. Move them to utils
rnd = random.Random(0)

class CrossAttentionModule(nn.Module):
    def __init__(self, device, attention_layers=6):
        super(CrossAttentionModule, self).__init__()
        self.cx_attention = [] # output remains query size, ie abatch x seqlen x 768
        self.attention_layers = attention_layers

        for _ in range(self.attention_layers):
            self.cx_attention.append(nn.MultiheadAttention(768, 8, batch_first=True, device=device))
            self.cx_attention.append(nn.Linear(in_features=768, out_features=768, device=device))
            self.cx_attention.append(nn.ReLU())

    def forward(self, query, context):
        query = query.cuda()
        key = context.cuda()
        value = context.cuda()
        for i in range(self.attention_layers):
            # cross attention layer
            query = self.cx_attention[i*3](query, key, value)[0] # 0th index is attn output, 1st index is attention weights
            # feedforward layer
            query = self.cx_attention[i*3+2](self.cx_attention[i*3+1](query))
        return query

def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2 -> dot product
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))

    # print the sizes of the q_vectors and ctx_vectors
    # print("q vector size: ", q_vectors.size())
    # print("ctx vector size: ", ctx_vectors.size())
    # # print the size of the result
    # print("result size: ", r.size())

    return r

def transformer_similarity_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    Applies cross attention mechanism between the query and context vectors to obtain a similarity score
    :param q_vectors: n1 x D
    :param ctx_vectors: n2 x D
    :return: n1 x n2 similarity score matrix
    """
    scores = similarity_score_fn(q_vectors, ctx_vectors)

    # print the sizes of the q_vectors and ctx_vectors
    print("q vector size: ", q_vectors.size())
    print("ctx vector size: ", ctx_vectors.size())
    # print the size of the result
    print("result size: ", scores.size())

    return scores

def cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return F.cosine_similarity(q_vector, ctx_vectors, dim=1)


class BiEncoder(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
    ):
        super(BiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder

        # TODO - add cross attention scoring module
        self.cross_attention = CrossAttentionModule("cuda:0")
        self.linear = nn.Linear(in_features=768, out_features=1)

    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
        fix_encoder: bool = False,
        representation_token_pos=0,
    ) -> (T, T, T):
        sequence_output = None
        pooled_output = None
        hidden_states = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    sequence_output, pooled_output, hidden_states = sub_model(
                        ids,
                        segments,
                        attn_mask,
                        representation_token_pos=representation_token_pos,
                    )

                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                sequence_output, pooled_output, hidden_states = sub_model(
                    ids,
                    segments,
                    attn_mask,
                    representation_token_pos=representation_token_pos,
                )

        # TODO - test the seq output

        return sequence_output, pooled_output, hidden_states

    # TODO -
    def forward(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        encoder_type: str = None,
        representation_token_pos=0,
    ) -> Tuple[T, T]:
        q_encoder = self.question_model if encoder_type is None or encoder_type == "question" else self.ctx_model
        _q_seq, q_pooled_out, _q_hidden = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
            self.fix_q_encoder,
            representation_token_pos=representation_token_pos,
        )
        print("q_pooled_out type: ", type(q_pooled_out))
        print("q_hidden type: ", type(_q_hidden))
        print("q_seq type: ", type(_q_seq))

        print("q_pooled_out: ", q_pooled_out.shape)
        # print("q_hidden: ", _q_hidden.shape)
        print("q_seq: ", _q_seq.shape)

        ctx_encoder = self.ctx_model if encoder_type is None or encoder_type == "ctx" else self.question_model
        _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(
            ctx_encoder, context_ids, ctx_segments, ctx_attn_mask, self.fix_ctx_encoder
        )

        print("ctx_pooled_out type: ", type(ctx_pooled_out))
        print("ctx_hidden type: ", type(_ctx_hidden))
        print("ctx_seq type: ", type(_ctx_seq))

        print("ctx_pooled_out: ", ctx_pooled_out.shape)
        # print("ctx_hidden: ", _ctx_hidden.shape)
        print("ctx_seq: ", _ctx_seq.shape)

        return q_pooled_out, _ctx_seq

    def create_biencoder_input(
        self,
        samples: List[BiEncoderSample],
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
        query_token: str = None,
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of BiEncoderSample-s to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        for sample in samples:
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only

            if shuffle and shuffle_positives:
                positive_ctxs = sample.positive_passages
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample.positive_passages[0]

            neg_ctxs = sample.negative_passages
            hard_neg_ctxs = sample.hard_negative_passages
            question = sample.query
            # question = normalize_question(sample.query)

            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]

            neg_ctxs = neg_ctxs[0:num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

            all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)

            current_ctxs_len = len(ctx_tensors)

            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(ctx.text, title=ctx.title if (insert_title and ctx.title) else None)
                for ctx in all_ctxs
            ]

            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx,
                    )
                ]
            )

            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    query_span = _select_span_with_token(question, tensorizer, token_str=query_token)
                    question_tensors.append(query_span)
                else:
                    question_tensors.append(tensorizer.text_to_tensor(" ".join([query_token, question])))
            else:
                question_tensors.append(tensorizer.text_to_tensor(question))

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)

        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            "question",
        )

    def load_state(self, saved_state: CheckpointState, strict: bool = True):
        # TODO: make a long term HF compatibility fix
        # if "question_model.embeddings.position_ids" in saved_state.model_dict:
        #    del saved_state.model_dict["question_model.embeddings.position_ids"]
        #    del saved_state.model_dict["ctx_model.embeddings.position_ids"]
        self.load_state_dict(saved_state.model_dict, strict=strict)

    def get_state_dict(self):
        return self.state_dict()


class BiEncoderNllLoss(object):
    def __init__(self, model):
        self.biencoder = model
    # TODO - add cross attention scoring module as the input
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        # TESTING - new score function
        # scores = self.get_scores(q_vectors, ctx_vectors)
        scores = self.biencoder.cross_attention(q_vectors, ctx_vectors)
        scores = self.biencoder.linear(scores).squeeze(-1)

        # if len(q_vectors.size()) > 1:
        #     q_num = q_vectors.size(0)
        #     scores = scores.view(q_num, -1)
        # softmax_scores = F.log_softmax(scores, dim=1)
        # softmax_scores = F.sigmoid(scores)

        softmax_scores = scores

        print("softmax scores: ", softmax_scores)
        print("positive idx per question: ", torch.tensor(positive_idx_per_question).to(softmax_scores.device))

        # TODO - convert the positive_idx_per_question to a label tensor


        # TODO - try with BCELogitsLoss and sigmoid
        loss = F.binary_cross_entropy_with_logits(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_cross_attention_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_cross_attention_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores

    @staticmethod
    def get_cross_attention_similarity_function():
        return transformer_similarity_scores


def _select_span_with_token(text: str, tensorizer: Tensorizer, token_str: str = "[START_ENT]") -> T:
    id = tensorizer.get_token_id(token_str)
    query_tensor = tensorizer.text_to_tensor(text)

    if id not in query_tensor:
        query_tensor_full = tensorizer.text_to_tensor(text, apply_max_len=False)
        token_indexes = (query_tensor_full == id).nonzero()
        if token_indexes.size(0) > 0:
            start_pos = token_indexes[0, 0].item()
            # add some randomization to avoid overfitting to a specific token position

            left_shit = int(tensorizer.max_length / 2)
            rnd_shift = int((rnd.random() - 0.5) * left_shit / 2)
            left_shit += rnd_shift

            query_tensor = query_tensor_full[start_pos - left_shit :]
            cls_id = tensorizer.tokenizer.cls_token_id
            if query_tensor[0] != cls_id:
                query_tensor = torch.cat([torch.tensor([cls_id]), query_tensor], dim=0)

            from dpr.models.reader import _pad_to_len

            query_tensor = _pad_to_len(query_tensor, tensorizer.get_pad_id(), tensorizer.max_length)
            query_tensor[-1] = tensorizer.tokenizer.sep_token_id
            # logger.info('aligned query_tensor %s', query_tensor)

            assert id in query_tensor, "query_tensor={}".format(query_tensor)
            return query_tensor
        else:
            raise RuntimeError("[START_ENT] toke not found for Entity Linking sample query={}".format(text))
    else:
        return query_tensor
