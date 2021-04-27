
import json
import sys
import numpy as np
import random
from collections import namedtuple

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as D

from utils import vocab_pad_idx, vocab_eos_idx, flatten, structured_map, try_cuda
from utils import vocab_bos_idx

# from speaker.py
def _batch_observations_and_actions(path_obs, path_actions, encoded_instructions, include_last_ob=False):
    #print("_batch_observations_and_actions")
    seq_lengths = np.array([len(a) for a in path_actions])
    max_path_length = seq_lengths.max()

    # DO NOT permute the sequence, since here we are doing manual LSTM unrolling in encoder
    # perm_indices = np.argsort(-seq_lengths)
    perm_indices = np.arange(len(path_obs))
    #path_obs, path_actions, encoded_instructions, seq_lengths = zip(*sorted(zip(path_obs, path_actions, encoded_instructions, seq_lengths), key=lambda p: p[-1], reverse=True))
    # path_obs = [path_obs[i] for i in perm_indices]
    # path_actions = [path_actions[i] for i in perm_indices]
    # if encoded_instructions:
    #     encoded_instructions = [encoded_instructions[i] for i in perm_indices]
    # seq_lengths = [seq_lengths[i] for i in perm_indices]

    batch_size = len(path_obs)
    assert batch_size == len(path_actions)

    mask = np.ones((batch_size, max_path_length), np.bool)
    action_embedding_dim = path_obs[0][0]['action_embedding'].shape[-1]
    batched_action_embeddings = [
        np.zeros((batch_size, action_embedding_dim), np.float32)
        for _ in range(max_path_length)]
    feature_list = path_obs[0][0]['feature']
    assert len(feature_list) == 1
    image_feature_shape = feature_list[0].shape
    batched_image_features = [
        np.zeros((batch_size,) + image_feature_shape, np.float32)
        for _ in range(max_path_length)]
    #for i, (obs, actions) in enumerate(zip(path_obs, path_actions)):
    #    for t, (ob, a) in enumerate(zip(obs, actions)):
    #        possible_actions = ob['action_embedding'].shape[0]
    #        assert a<possible_actions,(a,possible_actions)
    for i, (obs, actions) in enumerate(zip(path_obs, path_actions)):
        if include_last_ob:
            # include the last observation, e.g. the current observation (before taking the next action)
            # combined with the next action candidates, and used in the generative language grounded policy
            assert len(obs) == len(actions) , (i, len(obs) , len(actions) )
        else:
            # don't include the last state, which should result after the stop action, when the whole trajectory is given
            assert len(obs) == len(actions) + 1 , (i, len(obs) , len(actions) + 1)
            obs = obs[:-1]
        mask[i, :len(actions)] = False
        for t, (ob, a) in enumerate(zip(obs, actions)):
            #assert a >= 0
            assert type(a)==int,(a,type(a))
            #print("t,a",t,a,ob['action_embedding'].shape)
            batched_image_features[t][i] = ob['feature'][0]
            batched_action_embeddings[t][i] = ob['action_embedding'][a]
    #import ipdb; ipdb.set_trace()
    batched_action_embeddings = [
        try_cuda(Variable(torch.from_numpy(act), requires_grad=False))
        for act in batched_action_embeddings]
    batched_image_features = [
        try_cuda(Variable(torch.from_numpy(feat), requires_grad=False))
        for feat in batched_image_features]
    mask = try_cuda(torch.from_numpy(mask))

    start_obs = [obs[0] for obs in path_obs]

    return start_obs, \
           batched_image_features, \
           batched_action_embeddings, \
           mask, \
           list(seq_lengths), \
           encoded_instructions, \
           list(perm_indices)

# from follower.py
def batch_instructions_from_encoded(encoded_instructions, max_length, reverse=False, sort=False):
    # encoded_instructions: list of lists of token indices (should not be padded, or contain BOS or EOS tokens)
    #seq_tensor = np.array(encoded_instructions)
    # make sure pad does not start any sentence
    num_instructions = len(encoded_instructions)
    seq_tensor = np.full((num_instructions, max_length), vocab_pad_idx)
    seq_lengths = []
    for i, inst in enumerate(encoded_instructions):
        if len(inst) > 0:
            assert inst[-1] != vocab_eos_idx
        if reverse:
            inst = inst[::-1]
        inst = np.concatenate((inst, [vocab_eos_idx]))
        inst = inst[:max_length]
        seq_tensor[i,:len(inst)] = inst
        seq_lengths.append(len(inst))

    seq_tensor = torch.from_numpy(seq_tensor)
    if sort:
        seq_lengths, perm_idx = torch.from_numpy(np.array(seq_lengths)).sort(0, True)
        seq_lengths = list(seq_lengths)
        seq_tensor = seq_tensor[perm_idx]

    mask = (seq_tensor == vocab_pad_idx)[:, :max(seq_lengths)]

    ret_tp = try_cuda(Variable(seq_tensor, requires_grad=False).long()), \
             try_cuda(mask.bool()), \
             seq_lengths
    if sort:
        ret_tp = ret_tp + (list(perm_idx),)
    return ret_tp
