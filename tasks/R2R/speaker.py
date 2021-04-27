import json
import sys
import numpy as np
import random
import tqdm
from collections import namedtuple

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as D

from utils import vocab_pad_idx, vocab_bos_idx, vocab_eos_idx, flatten, try_cuda
from batcher import batch_instructions_from_encoded ,_batch_observations_and_actions
#from model_transformer import _generate_square_subsequent_mask

import logging
logger = logging.getLogger(__name__)

InferenceState = namedtuple("InferenceState", "prev_inference_state, flat_index, last_word, word_count, score, last_alpha")

def backchain_inference_states(last_inference_state):
    word_indices = []
    inf_state = last_inference_state
    scores = []
    last_score = None
    attentions = []
    while inf_state is not None:
        word_indices.append(inf_state.last_word)
        attentions.append(inf_state.last_alpha)
        if last_score is not None:
            scores.append(last_score - inf_state.score)
        last_score = inf_state.score
        inf_state = inf_state.prev_inference_state
    scores.append(last_score)
    return list(reversed(word_indices))[1:], list(reversed(scores))[1:], list(reversed(attentions))[1:] # exclude BOS

class Seq2SeqSpeaker(object):
    feedback_options = ['teacher', 'argmax', 'sample']

    def __init__(self, args, env, results_path, encoder, decoder, instruction_len, max_episode_len=10):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

        self.encoder = encoder
        self.decoder = decoder
        self.instruction_len = instruction_len

        self.losses = []
        self.max_episode_len = max_episode_len

        self.speaker_transformer = ("transformer" in args.speaker_type)

        self.r4r_follow_detailed_path = args.r4r_follow_detailed_path

    def write_results(self):
        with open(self.results_path, 'w') as f:
            json.dump(self.results, f)

    # def n_inputs(self):
    #     return self.decoder.vocab_size
    #
    # def n_outputs(self):
    #     return self.decoder.vocab_size-1 # Model doesn't output start

    def _feature_variable(self, obs, beamed=False):
        ''' Extract precomputed features into variable. '''
        features = [ob['feature'] for ob in (flatten(obs) if beamed else obs)]
        assert all(len(f) == 1 for f in features)  #currently only support one image featurizer (without attention)
        features = np.stack(features)
        return try_cuda(Variable(torch.from_numpy(features), requires_grad=False))

    # Score of p_S(X|a_t,s_t)
    def _score_obs_actions_and_instructions(self, path_obs, path_actions, encoded_instructions, feedback):
        assert len(path_obs) == len(path_actions)
        assert len(path_obs) == len(encoded_instructions)
        for path_ob, path_action in zip(path_obs,path_actions):
            assert len(path_ob)==len(path_action)+1
        #print(path_actions)
        start_obs, batched_image_features, batched_action_embeddings, path_mask, \
            path_lengths, encoded_instructions, perm_indices = \
            _batch_observations_and_actions(
                path_obs, path_actions, encoded_instructions)

        instr_seq, _, _ = batch_instructions_from_encoded(encoded_instructions, self.instruction_len)

        batch_size = len(start_obs)

        # self.encoder.eval()
        # self.decoder.eval()
        # trunc = 13
        # batch_size=trunc
        # batched_action_embeddings=batched_action_embeddings
        # batched_image_features=batched_image_features
        # instr_seq=instr_seq[:trunc]

        if self.speaker_transformer:
            #seq_size = len(batched_action_embeddings)
            #src_mask = try_cuda(_generate_square_subsequent_mask(seq_size))
            mask = path_mask
            e_outputs = self.encoder(batched_action_embeddings, batched_image_features, pad_mask=mask)
            #print("e_outputs",e_outputs.shape)
            #e_outputs=e_outputs[:,:13,:].contiguous()
        else:
            ctx,h_t,c_t = self.encoder(batched_action_embeddings, batched_image_features)

        w_t = try_cuda(Variable(torch.from_numpy(np.full((batch_size,), vocab_bos_idx, dtype='int64')).long(),
                                requires_grad=False))
        if self.speaker_transformer:
            all_w_t = [w_t]
            w_t_view = w_t.view(-1, 1)
        ended = np.array([False] * batch_size)

        assert len(perm_indices) == batch_size
        outputs = [None] * batch_size
        for perm_index, src_index in enumerate(perm_indices):
            outputs[src_index] = {
                'instr_id': start_obs[perm_index]['instr_id'],
                'word_indices': [],
                'scores': [],
                #'actions': ' '.join(FOLLOWER_MODEL_ACTIONS[ac] for ac in path_actions[src_index]),
            }
        assert all(outputs)

        # for i in range(batch_size):
        #     assert outputs[i]['instr_id'] != '1008_0', "found example at index {}".format(i)

        if self.speaker_transformer and self.is_train:
            # BOS are not part of the encoded sequences
            #target = instr_seq[:,:].contiguous()

            tgt = torch.cat([torch.unsqueeze(w_t,1),instr_seq[:,:-1]], dim=1)
            #print("stacked_w_t",stacked_w_t.shape)
            #print(stacked_w_t)
            tgt_input = torch.transpose(tgt,1,0).contiguous()
            #print("tgt_input",tgt_input.shape)
            #output = self.decoder(tgt_input[:,:13].contiguous(), e_outputs[:,:13].contiguous())
            output = self.decoder(tgt_input.contiguous(), e_outputs.contiguous(),
                                  rectangle_mask=True, mem_pad_mask=mask, fix_emb=self.fix_emb)
            #print("output",output.shape)
            #sequence_last = output[-1,:,:]
            #logit = torch.squeeze(sequence_last,1)
            #print("output",output.shape)
            #import ipdb; ipdb.set_trace()
            #target_t2 = target_t[:-1,:].contiguous().view(-1)# .reshape(-1)
            #instr_seq_t = instr_seq.transpose(0,1)[:,:13]
            instr_seq_t = instr_seq.transpose(0,1)
            target_idx = instr_seq_t.contiguous().view(-1)
            loss_all = F.cross_entropy(output.view(-1, output.size(-1)), target_idx, ignore_index=vocab_pad_idx, reduction="none")
            loss = F.cross_entropy(output.view(-1, output.size(-1)), target_idx, ignore_index=vocab_pad_idx, reduction="mean")
            #import ipdb; ipdb.set_trace()
            logger.debug(str(loss))

            for t in range(self.instruction_len):
                for perm_index, src_index in enumerate(perm_indices):
                    word_idx = instr_seq_t[t,perm_index].item()#.data[0]
                    if not ended[perm_index]:
                        outputs[src_index]['word_indices'].append(int(word_idx))
                        #outputs[src_index]['score'] = float(sequence_scores[perm_index])
                        #outputs[src_index]['score'] = float(sequence_scores[perm_index].cpu().numpy())
                        #outputs[src_index]['scores'].append(word_scores[perm_index].data.tolist())
                        outputs[src_index]['score'] = float(0)
                        outputs[src_index]['scores'].append([0])
                    if word_idx == vocab_eos_idx:
                        ended[perm_index] = True

        else:
            # Do a sequence rollout and calculate the loss
            loss = 0
            sequence_scores = try_cuda(torch.zeros(batch_size))
            for t in range(self.instruction_len):
            #for t in range(20):
                #print(t)
                if self.speaker_transformer:
                    #import ipdb; ipdb.set_trace()
                    stacked_w_t = torch.stack(all_w_t, 0)
                    #print("stacked_w_t",stacked_w_t.shape)
                    #print(stacked_w_t)
                    output = self.decoder(stacked_w_t, e_outputs)
                    #print("output",output.shape)
                    sequence_last = output[-1,:,:]
                    logit = torch.squeeze(sequence_last,0)
                    #assert logit.shape==
                    logger.debug("t: "+str(t)+", logit"+str(logit.shape)+str(logit[:3,:20]))
                    #import ipdb; ipdb.set_trace()
                else:
                    h_t,c_t,alpha,logit = self.decoder(w_t.view(-1, 1), h_t, c_t, ctx, path_mask)
                # Supervised training

                # BOS are not part of the encoded sequences
                target = instr_seq[:,t].contiguous()

                # Determine next model inputs
                if feedback == 'teacher':
                    w_t = target
                elif feedback == 'argmax':
                    _,w_t = logit.max(1)        # student forcing - argmax
                    w_t = w_t.detach()
                elif feedback == 'sample':
                    probs = F.softmax(logit)    # sampling an action from model
                    m = D.Categorical(probs)
                    w_t = m.sample()
                    #w_t = probs.multinomial(1).detach().squeeze(-1)
                else:
                    sys.exit('Invalid feedback option')

                #print(logit.shape)
                log_probs = F.log_softmax(logit, dim=1)
                #import ipdb; ipdb.set_trace()
                word_scores = -F.nll_loss(log_probs, w_t, ignore_index=vocab_pad_idx, reduce=False)
                sequence_scores += word_scores.data
                if not self.speaker_transformer:
                    loss += F.nll_loss(log_probs, target, ignore_index=vocab_pad_idx, reduce=True, size_average=True)

                for perm_index, src_index in enumerate(perm_indices):
                    word_idx = w_t[perm_index].item()#.data[0]
                    if not ended[perm_index]:
                        outputs[src_index]['word_indices'].append(int(word_idx))
                        #outputs[src_index]['score'] = float(sequence_scores[perm_index])
                        outputs[src_index]['score'] = float(sequence_scores[perm_index].cpu().numpy())
                        outputs[src_index]['scores'].append(word_scores[perm_index].data.tolist())
                    if word_idx == vocab_eos_idx:
                        ended[perm_index] = True

                if self.speaker_transformer:
                    all_w_t.append(w_t)
                # print("t: %s\tstate: %s\taction: %s\tscore: %s" % (t, world_states[0], a_t.data[0], sequence_scores[0]))

                # Early exit if all ended
                if ended.all():
                    break

            for item in outputs:
                item['words'] = self.env.tokenizer.decode_sentence(item['word_indices'], break_on_eos=True, join=False)
                logger.debug(str(item['words']))
                logger.debug(str(item['word_indices']))

        #print("End.")

        return outputs, loss

    def rollout(self, load_next_minibatch=True):
        path_obs, path_actions, encoded_instructions = self.env.gold_obs_actions_and_instructions(
            self.max_episode_len, load_next_minibatch=load_next_minibatch, r4r_follow_detailed_path=self.r4r_follow_detailed_path)
        outputs, loss = self._score_obs_actions_and_instructions(path_obs, path_actions, encoded_instructions, self.feedback)
        self.loss = loss
        self.losses.append(loss.item() if isinstance(loss,torch.Tensor) else loss)#data[0])
        return outputs

    def beam_search(self, beam_size, path_obs, path_actions):

        # TODO: here
        assert len(path_obs) == len(path_actions)

        start_obs, batched_image_features, batched_action_embeddings, path_mask, \
            path_lengths, _, perm_indices = \
            _batch_observations_and_actions(path_obs, path_actions, None)
        batch_size = len(start_obs)
        assert len(perm_indices) == batch_size

        ctx,h_t,c_t = self.encoder(batched_action_embeddings, batched_image_features)

        completed = []
        for _ in range(batch_size):
            completed.append([])

        beams = [
            [InferenceState(prev_inference_state=None,
                            flat_index=i,
                            last_word=vocab_bos_idx,
                            word_count=0,
                            score=0.0,
                            last_alpha=None)]
            for i in range(batch_size)
        ]

        for t in range(self.instruction_len):
            flat_indices = []
            beam_indices = []
            w_t_list = []
            for beam_index, beam in enumerate(beams):
                for inf_state in beam:
                    beam_indices.append(beam_index)
                    flat_indices.append(inf_state.flat_index)
                    w_t_list.append(inf_state.last_word)
            w_t = try_cuda(Variable(torch.LongTensor(w_t_list), requires_grad=False))
            if len(w_t.shape) == 1:
                w_t = w_t.unsqueeze(0)

            h_t,c_t,alpha,logit = self.decoder(w_t.view(-1, 1), h_t[flat_indices], c_t[flat_indices], ctx[beam_indices], path_mask[beam_indices])

            log_probs = F.log_softmax(logit, dim=1).data
            _, word_indices = logit.data.topk(min(beam_size, logit.size()[1]), dim=1)
            word_scores = log_probs.gather(1, word_indices)
            assert word_scores.size() == word_indices.size()

            start_index = 0
            new_beams = []
            all_successors = []
            for beam_index, beam in enumerate(beams):
                successors = []
                end_index = start_index + len(beam)
                if beam:
                    for inf_index, (inf_state, word_score_row, word_index_row) in \
                        enumerate(zip(beam, word_scores[start_index:end_index], word_indices[start_index:end_index])):
                        for word_score, word_index in zip(word_score_row, word_index_row):
                            flat_index = start_index + inf_index
                            successors.append(
                                InferenceState(
                                    prev_inference_state=inf_state,
                                    flat_index=flat_index,
                                    last_word=word_index,
                                    word_count=inf_state.word_count + 1,
                                    score=inf_state.score + word_score,
                                    last_alpha=alpha[flat_index].data)
                            )
                start_index = end_index
                successors = sorted(successors, key=lambda t: t.score, reverse=True)[:beam_size]
                all_successors.append(successors)

            for beam_index, successors in enumerate(all_successors):
                new_beam = []
                for successor in successors:
                    if successor.last_word == vocab_eos_idx or t == self.instruction_len - 1:
                        completed[beam_index].append(successor)
                    else:
                        new_beam.append(successor)
                if len(completed[beam_index]) >= beam_size:
                    new_beam = []
                new_beams.append(new_beam)

            beams = new_beams

            if not any(beam for beam in beams):
                break

        outputs = []
        for _ in range(batch_size):
            outputs.append([])

        for perm_index, src_index in enumerate(perm_indices):
            this_outputs = outputs[src_index]
            assert len(this_outputs) == 0

            this_completed = completed[perm_index]
            instr_id = start_obs[perm_index]['instr_id']
            for inf_state in sorted(this_completed, key=lambda t: t.score, reverse=True)[:beam_size]:
                word_indices, scores, attentions = backchain_inference_states(inf_state)
                this_outputs.append({
                    'instr_id': instr_id,
                    'word_indices': word_indices,
                    'score': inf_state.score,
                    'scores': scores,
                    'words': self.env.tokenizer.decode_sentence(word_indices, break_on_eos=True, join=False),
                    'attentions': attentions,
                })
        return outputs

    def test(self, args, use_dropout=False, feedback='argmax', allow_cheat=False, beam_size=1):
        ''' Evaluate once on each instruction in the current environment '''
        if not allow_cheat: # permitted for purpose of calculating validation loss only
            assert feedback in ['argmax', 'sample'] # no cheating by using teacher at test time!
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
        self.beam_size = beam_size
        self.env.reset_epoch()
        self.losses = []
        self.results = {}
        self.is_train = False
        self.fix_emb = False

        # We rely on env showing the entire batch before repeating anything
        looped = False
        # rollout_scores = []
        # beam_10_scores = []
        n_iters = len(self.env.data)//self.env.batch_size + (0 if len(self.env.data)%self.env.batch_size==0 else 1)
        logger.debug("test2: n_iters: %d"%n_iters)
        it = range(1, n_iters + 1)
        it = tqdm.tqdm(it,dynamic_ncols=True)
        while True:
            it.update(1)
            with torch.no_grad():
                rollout_results = self.rollout()
            # if self.feedback == 'argmax':
            #     path_obs, path_actions, _ = self.env.gold_obs_actions_and_instructions(self.max_episode_len, load_next_minibatch=False)
            #     beam_results = self.beam_search(1, path_obs, path_actions)
            #     assert len(rollout_results) == len(beam_results)
            #     for rollout_traj, beam_trajs in zip(rollout_results, beam_results):
            #         assert rollout_traj['instr_id'] == beam_trajs[0]['instr_id']
            #         assert rollout_traj['word_indices'] == beam_trajs[0]['word_indices']
            #         assert np.allclose(rollout_traj['score'], beam_trajs[0]['score'])
            #     print("passed check: beam_search with beam_size=1")
            #
            #     self.env.set_beam_size(10)
            #     beam_results = self.beam_search(10, path_obs, path_actions)
            #     assert len(rollout_results) == len(beam_results)
            #     for rollout_traj, beam_trajs in zip(rollout_results, beam_results):
            #         rollout_score = rollout_traj['score']
            #         rollout_scores.append(rollout_score)
            #         beam_score = beam_trajs[0]['score']
            #         beam_10_scores.append(beam_score)
            #         # assert rollout_score <= beam_score
            # # print("passed check: beam_search with beam_size=10")

            for result in rollout_results:
                if result['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[result['instr_id']] = result
            if looped:
                break
        it.close()
        # if self.feedback == 'argmax':
        #     print("avg rollout score: ", np.mean(rollout_scores))
        #     print("avg beam 10 score: ", np.mean(beam_10_scores))
        return self.results

    def train(self, args, encoder_optimizer, decoder_optimizer, n_iters, feedback='teacher', fix_emb=False):
        ''' Train for a given number of iterations '''
        assert feedback in self.feedback_options
        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()
        self.losses = []
        self.is_train = True
        self.fix_emb = fix_emb
        it = range(1, n_iters + 1)
        it = tqdm.tqdm(it)
        count=0
        for _ in it:
            #print(count)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            self.rollout()
            self.loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            self.loss=None
            count+=1
            if args.debug and count==10:
                it.close()
                break

    def _encoder_and_decoder_paths(self, base_path):
        return base_path + "_enc", base_path + "_dec"
    def _encoder_and_decoder_paths3(self, base_path):
        return base_path + "_spkEnc", base_path + "_spkDec"

    def save(self, path):
        ''' Snapshot models '''
        encoder_path, decoder_path = self._encoder_and_decoder_paths(path)
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load(self, path, **kwargs):
        ''' Loads parameters (but not training state) '''
        encoder_path, decoder_path = self._encoder_and_decoder_paths(path)
        self.encoder.load_state_dict(torch.load(encoder_path, **kwargs))
        self.decoder.load_state_dict(torch.load(decoder_path, **kwargs))

    def load_speaker(self, path, sep=False, **kwargs):
        ''' Loads parameters (but not training state) '''
        if sep:
            encoder_path, decoder_path = self._encoder_and_decoder_paths3(path)
        else:
            encoder_path, decoder_path = self._encoder_and_decoder_paths(path)
        self.encoder.load_state_dict(torch.load(encoder_path, **kwargs))
        self.decoder.load_state_dict(torch.load(decoder_path, **kwargs))
