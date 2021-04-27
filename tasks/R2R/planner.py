''' Agents: stop/random/shortest/seq2seq  '''

import json
import numpy as np
import random
from collections import namedtuple, defaultdict
import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as D

from utils import vocab_pad_idx, vocab_eos_idx, flatten, structured_map, try_cuda, assert2, update_state_dict
from utils import vocab_bos_idx
from batcher import _batch_observations_and_actions, batch_instructions_from_encoded

#from env import FOLLOWER_MODEL_ACTIONS, FOLLOWER_ENV_ACTIONS, IGNORE_ACTION_INDEX, LEFT_ACTION_INDEX, RIGHT_ACTION_INDEX, START_ACTION_INDEX, END_ACTION_INDEX, FORWARD_ACTION_INDEX, index_action_tuple
from env import WorldState

import logging
logger = logging.getLogger(__name__)

InferenceState = namedtuple("InferenceState", "prev_inference_state, world_state, observation, flat_index, last_action, last_action_embedding, action_count, score, h_t, c_t, last_alpha")
#FollowerState = namedtuple("FollowerState","ctx, seq_mask,"
#                            "h_t, c_t, alpha, follow_logit, alpha_v,"
#                            "u_t_prev, all_u_t, f_t_list")
def FollowerState(ctx, seq_mask, h_t, c_t, alpha, follow_logit, alpha_v, u_t_prev, all_u_t, f_t_list):
    return {"ctx":ctx, "seq_mask":seq_mask,
        "h_t":h_t, "c_t":c_t, "alpha":alpha, "follow_logit":follow_logit, "alpha_v":alpha_v,
        "u_t_prev":u_t_prev, "all_u_t":all_u_t, "f_t_list":f_t_list}
def SpeakerState(encoded_instructions,record_obs,record_actions,t):
    return {"record_obs":record_obs,"record_actions":record_actions,
            "encoded_instructions":encoded_instructions, "t":t}

def detach_follow_state(fs):
    fs["h_t"] = fs["h_t"].detach()
    fs["c_t"] = fs["c_t"].detach()
    fs["all_u_t"] = fs["all_u_t"].detach()
    return fs
def follow_state_detach_split_for_next(fs,all_u_t,batch_size):
    fs = detach_follow_state(fs)
    assert fs["ctx"].shape[0]==batch_size
    u_t_prev = all_u_t.detach()
    return [ FollowerState(fs["ctx"],fs["seq_mask"],
              fs["h_t"][idx],fs["c_t"][idx],None,None,None,
              u_t_prev[idx],None,None) for idx in range(batch_size)]
def follow_state_action(fs,action):
    return FollowerState(fs["ctx"],fs["seq_mask"],
                         fs["h_t"],fs["c_t"],None,None,None,
                         fs["u_t_prev"][action],None,None)
def speak_state_action(fs,action):
    #print("record_actions",fs["record_actions"],action)
    return SpeakerState(fs["encoded_instructions"],
                        fs["record_obs"],
                        fs["record_actions"]\
                        +[action.cpu().numpy().item() if isinstance(action,torch.Tensor) else action],
                        fs["t"]+1)
    #h_t_data, c_t_data = h_t.detach(),c_t.detach()
    #u_t_data = all_u_t.detach()

# for analyses
def np_softmax(x,dim):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=dim)

def np_softmax_last(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x,axis=-1)[...,np.newaxis])
    return e_x / e_x.sum(axis=-1)[...,np.newaxis]

def np_norm(x):
    return x / x.sum()

# For FAST
import networkx as nx
from utils_fast import PriorityQueue
from running_mean_std import RunningMean
SearchState = namedtuple("SearchState", "flogit,flogp, world_state, observation, action, follower_states, speaker_states, action_count, father") # flat_index,
CandidateState = namedtuple("CandidateState", "flogit,flogp,world_states,actions,pm,speaker,scorer") # flat_index,

Cons = namedtuple("Cons", "first, rest")
def cons_to_list(cons):
    l = []
    while True:
        l.append(cons.first)
        cons = cons.rest
        if cons is None:
            break
    return l

def backchain_inference_states(last_inference_state):
    states = []
    observations = []
    actions = []
    inf_state = last_inference_state
    scores = []
    last_score = None
    attentions = []
    while inf_state is not None:
        states.append(inf_state.world_state)
        observations.append(inf_state.observation)
        actions.append(inf_state.last_action)
        attentions.append(inf_state.last_alpha)
        if last_score is not None:
            scores.append(last_score - inf_state.score)
        last_score = inf_state.score
        inf_state = inf_state.prev_inference_state
    scores.append(last_score)
    return list(reversed(states)), list(reversed(observations)), list(reversed(actions))[1:], list(reversed(scores))[1:], list(reversed(attentions))[1:] # exclude start action

def least_common_viewpoint_path(inf_state_a, inf_state_b):
    # return inference states traversing from A to X, then from Y to B,
    # where X and Y are the least common ancestors of A and B respectively that share a viewpointId
    path_to_b_by_viewpoint =  {
    }
    b = inf_state_b
    b_stack = Cons(b, None)
    while b is not None:
        path_to_b_by_viewpoint[b.world_state.viewpointId] = b_stack
        b = b.prev_inference_state
        b_stack = Cons(b, b_stack)
    a = inf_state_a
    path_from_a = [a]
    while a is not None:
        vp = a.world_state.viewpointId
        if vp in path_to_b_by_viewpoint:
            path_to_b = cons_to_list(path_to_b_by_viewpoint[vp])
            assert path_from_a[-1].world_state.viewpointId == path_to_b[0].world_state.viewpointId
            return path_from_a + path_to_b[1:]
        a = a.prev_inference_state
        path_from_a.append(a)
    raise AssertionError("no common ancestor found")

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

    def write_results(self):
        results = {}
        for key, item in self.results.items():
            results[key] = {
                'instr_id': item['instr_id'],
                'trajectory': item['trajectory'],
            }
        with open(self.results_path, 'w') as f:
            json.dump(results, f)

    def rollout(self):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self):
        self.env.reset_epoch()
        self.losses = []
        self.results = {}

        # We rely on env showing the entire batch before repeating anything
        #print 'Testing %s' % self.__class__.__name__
        looped = False
        # rollout_scores = []
        # beam_10_scores = []
        n_iters = len(self.env.data)//self.env.batch_size + (0 if len(self.env.data)%self.env.batch_size==0 else 1)
        logger.debug("test2: n_iters: %d"%n_iters)
        it = range(1, n_iters + 1)
        it = tqdm.tqdm(it,dynamic_ncols=True)
        count=0
        while True:
            it.update(1)
            with torch.no_grad():
                rollout_results = self.rollout()

            for result in rollout_results:
                if result['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[result['instr_id']] = result
            if looped:
                break
            count+=1
            if self.debug_interval>0 and count>=self.debug_interval:
                break
        it.close()
        # if self.feedback == 'argmax':
        #     print("avg rollout score: ", np.mean(rollout_scores))
        #     print("avg beam 10 score: ", np.mean(beam_10_scores))
        return self.results

def path_element_from_observation(ob):
    return (ob['viewpoint'], ob['heading'], ob['elevation'])

# FAST
def realistic_jumping(graph, start_step, dest_obs):
    if start_step == path_element_from_observation(dest_obs):
        return []
    s = start_step[0]
    t = dest_obs['viewpoint']
    path = nx.shortest_path(graph,s,t)
    traj = [(vp,0,0) for vp in path[:-1]]
    traj.append(path_element_from_observation(dest_obs))
    return traj

class StopAgent(BaseAgent):
    ''' An agent that doesn't move! '''

    def rollout(self):
        world_states = self.env.reset()
        obs = self.env.observe(world_states)
        traj = [{
            'instr_id': ob['instr_id'],
            'trajectory': [path_element_from_observation(ob) ]
        } for ob in obs]
        return traj


class RandomAgent(BaseAgent):
    ''' An agent that picks a random direction then tries to go straight for
        five viewpoint steps and then stops. '''

    def rollout(self):
        world_states = self.env.reset()
        obs = self.env.observe(world_states)
        traj = [{
            'instr_id': ob['instr_id'],
            'trajectory': [path_element_from_observation(ob)]
        } for ob in obs]
        ended = [False] * len(obs)

        self.steps = [0] * len(obs)
        for t in range(6):
            actions = []
            for i, ob in enumerate(obs):
                if self.steps[i] >= 5:
                    actions.append(0)  # do nothing, i.e. end
                    ended[i] = True
                elif self.steps[i] == 0:
                    a = np.random.randint(len(ob['adj_loc_list']) - 1) + 1
                    actions.append(a)  # choose a random adjacent loc
                    self.steps[i] += 1
                else:
                    assert len(ob['adj_loc_list']) > 1
                    actions.append(1)  # go forward
                    self.steps[i] += 1
            world_states = self.env.step(world_states, actions, obs)
            obs = self.env.observe(world_states)
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['trajectory'].append(path_element_from_observation(ob))
        return traj

class ShortestAgent(BaseAgent):
    ''' An agent that always takes the shortest path to goal. '''

    def rollout(self):
        world_states = self.env.reset()
        #obs = self.env.observe(world_states)
        all_obs, all_actions = self.env.shortest_paths_to_goals(world_states, 20)
        return [
            {
                'instr_id': obs[0]['instr_id'],
                # end state will appear twice because stop action is a no-op, so exclude it
                'trajectory': [path_element_from_observation(ob) for ob in obs[:-1]]
            }
            for obs in all_obs
        ]

class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    # env_actions = FOLLOWER_ENV_ACTIONS
    # start_index = START_ACTION_INDEX
    # ignore_index = IGNORE_ACTION_INDEX
    # forward_index = FORWARD_ACTION_INDEX
    # end_index = END_ACTION_INDEX
    feedback_options = ['teacher', 'argmax', 'sample']

    def __init__(self, env, results_path, encoder, decoder, speaker, speak_encoder, speak_decoder, episode_len=10, beam_size=1, reverse_instruction=True, max_instruction_length=80):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.encoder = encoder
        self.decoder = decoder
        self.speaker = speaker
        self.speak_encoder = speak_encoder
        self.speak_decoder = speak_decoder
        self.episode_len = episode_len
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.nllloss = nn.NLLLoss(ignore_index=-1)
        self.logsm = nn.LogSoftmax(dim=1)
        self.beam_size = beam_size
        self.reverse_instruction = reverse_instruction
        self.max_instruction_length = max_instruction_length
        self.beam_env = None
        self.beta = 0.5
        self.is_train=True
        self.torch_version = float(torch.__version__[:-2])
        self.overall_speaker = None
        self.bert = False
        self.bert_orig = False
        self.analysis_entropy =False
        self.analysis_text_gen =False
        self.analysis_text_score =False
        self.analysis_text_bleu =False
        self.require_logit = False

    # @staticmethod
    # def n_inputs():
    #     return len(FOLLOWER_MODEL_ACTIONS)
    #
    # @staticmethod
    # def n_outputs():
    #     return len(FOLLOWER_MODEL_ACTIONS)-2 # Model doesn't output start or ignore

    def _feature_variables(self, obs, beamed=False):
        ''' Extract precomputed features into variable. '''
        feature_lists = list(zip(*[ob['feature'] for ob in (flatten(obs) if beamed else obs)]))
        assert len(feature_lists) == len(self.env.image_features_list)
        batched = []
        for featurizer, feature_list in zip(self.env.image_features_list, feature_lists):
            batched.append(featurizer.batch_features(feature_list))
        return batched

    def _action_variable(self, obs):
        # get the maximum number of actions of all sample in this batch
        max_num_a = -1
        for i, ob in enumerate(obs):
            max_num_a = max(max_num_a, len(ob['adj_loc_list']))

        is_valid = np.zeros((len(obs), max_num_a), np.float32)
        action_embedding_dim = obs[0]['action_embedding'].shape[-1]
        action_embeddings = np.zeros(
            (len(obs), max_num_a, action_embedding_dim),
            dtype=np.float32)
        for i, ob in enumerate(obs):
            adj_loc_list = ob['adj_loc_list']
            num_a = len(adj_loc_list)
            is_valid[i, 0:num_a] = 1.
            for n_a, adj_dict in enumerate(adj_loc_list):
                action_embeddings[i, :num_a, :] = ob['action_embedding']
        return (
            try_cuda(Variable(torch.from_numpy(action_embeddings), requires_grad=False)),
            try_cuda(Variable(torch.from_numpy(is_valid), requires_grad=False)),
            is_valid)

    def _teacher_action(self, obs, ended):
        ''' Extract teacher actions into variable. '''
        a = torch.LongTensor(len(obs))
        for i,ob in enumerate(obs):
            # Supervised teacher only moves one axis at a time
            a[i] = ob['teacher'] if not ended[i] else -1
        return try_cuda(Variable(a, requires_grad=False))

    def _teacher_action_np(self, obs, ended):
        ''' Extract teacher actions into variable. '''
        a = np.zeros((len(obs),),dtype=np.int32)
        b = []
        for i,ob in enumerate(obs):
            # Supervised teacher only moves one axis at a time
            a[i] = ob['teacher'] if not ended[i] else -1
            if not ended[i]:
                b.append(ob['teacher'])
        return a,np.array(b,dtype=np.int32)

    def _extract_encoded_instructions(self, obs, beamed=False):
        bert_encoded_instructions = [ob['instr_encoding_bert'] for ob in (flatten(obs) if beamed else obs)] if self.bert else None
        encoded_instructions = [ob['instr_encoding'] for ob in (flatten(obs) if beamed else obs)]
        return encoded_instructions, bert_encoded_instructions

    def _proc_batch(self, obs, beamed=False):
        encoded_instructions = [ob['instr_encoding'] for ob in (flatten(obs) if beamed else obs)]
        return batch_instructions_from_encoded(encoded_instructions, self.max_instruction_length, reverse=self.reverse_instruction)

    def rollout(self):
        if self.fast_greedy_search:
            self.records = defaultdict(list)
            #print("_rollout_with_search")
            return self._rollout_with_search()
        if self.beam_size == 1:
            return self._rollout_with_loss()
        else:
            assert self.beam_size >= 1
            beams, _, _ = self.beam_search(self.beam_size)
            return [beam[0] for beam in beams]

    # Score of p_F(a_t,s_t|X)
    # Score of \Pi_t p_F(a_(t+1)|X,a_t,s_t)
    def _score_obs_actions_and_instructions(self, path_obs, path_actions, encoded_instructions):
        batch_size = len(path_obs)
        assert len(path_actions) == batch_size
        assert len(encoded_instructions) == batch_size
        for path_o, path_a in zip(path_obs, path_actions):
            assert len(path_o) == len(path_a) + 1

        seq, seq_mask, seq_lengths, perm_indices = \
            batch_instructions_from_encoded(
                encoded_instructions, self.max_instruction_length,
                reverse=self.reverse_instruction, sort=True)
        loss = 0

        ctx, h_t, c_t = self.encoder[0](seq, seq_lengths)
        u_t_prev = self.decoder[0].u_begin.expand(batch_size, -1)  # init action
        ended = np.array([False] * batch_size)
        sequence_scores = try_cuda(torch.zeros(batch_size))

        traj = [{
            'instr_id': path_o[0]['instr_id'],
            'trajectory': [path_element_from_observation(path_o[0])],
            'actions': [],
            'scores': [],
            'observations': [path_o[0]],
            'instr_encoding': path_o[0]['instr_encoding']
        } for path_o in path_obs]

        obs = None
        for t in range(self.episode_len):
            next_obs = []
            next_target_list = []
            for perm_index, src_index in enumerate(perm_indices):
                path_o = path_obs[src_index]
                path_a = path_actions[src_index]
                if t < len(path_a):
                    next_target_list.append(path_a[t])
                    next_obs.append(path_o[t])
                else:
                    next_target_list.append(-1)
                    next_obs.append(obs[perm_index])

            obs = next_obs

            target = try_cuda(Variable(torch.LongTensor(next_target_list), requires_grad=False))

            f_t_list = self._feature_variables(obs) # Image features from obs
            all_u_t, is_valid, _ = self._action_variable(obs)

            assert len(f_t_list) == 1, 'for now, only work with MeanPooled feature'
            h_t, c_t, alpha, logit, alpha_v = self.decoder[0](
                u_t_prev, all_u_t, f_t_list[0], h_t, c_t, ctx, seq_mask)

            # Mask outputs of invalid actions
            logit[is_valid == 0] = -float('inf')

            # Supervised training
            loss += self.criterion(logit, target)

            # Determine next model inputs
            a_t = torch.clamp(target, min=0)  # teacher forcing
            # update the previous action
            u_t_prev = all_u_t[np.arange(batch_size), a_t, :].detach()

            action_scores = -F.cross_entropy(logit, target, ignore_index=-1, reduce=False).data
            sequence_scores += action_scores

            # Save trajectory output
            for perm_index, src_index in enumerate(perm_indices):
                ob = obs[perm_index]
                if not ended[perm_index]:
                    traj[src_index]['trajectory'].append(path_element_from_observation(ob))
                    traj[src_index]['score'] = float(sequence_scores[perm_index])
                    traj[src_index]['scores'].append(action_scores[perm_index])
                    traj[src_index]['actions'].append(a_t.data[perm_index])
                    # traj[src_index]['observations'].append(ob)

            # Update ended list
            for i in range(batch_size):
                action_idx = a_t[i].item()#.data[0]
                if action_idx == 0:
                    ended[i] = True

            # Early exit if all ended
            if ended.all():
                break

        return traj, loss

    # Score of p_S(X|a_t,s_t)
    #def action2speak_decoder(self, path_obs, path_actions, encoded_instructions, feedback,batch_size):
    #def _score_obs_actions_and_instructions(self, path_obs, path_actions, encoded_instructions):
    def _score_obs_actions_and_instructions_speaker(self, speak_encoder, path_obs, path_actions, encoded_instructions,
                                                    loss_signs, loss_beam,
                                                    feedback):
        assert len(path_obs) == len(path_actions)
        assert len(path_obs) == len(encoded_instructions)
        batch_size = len(path_obs)

        start_obs, batched_image_features, batched_action_embeddings, path_mask, \
            path_lengths, encoded_instructions, perm_indices = \
            _batch_observations_and_actions(
                path_obs, path_actions, encoded_instructions, include_last_ob=True)

        # path_mask is a mask for ctx to seal the future information in ctx.
        # However ctx have only current and past information in this implementation.
        # Therefore path_mask is always False with the shape of [batch, seq_len] == ctx.shape
        # ctx : [batch, seq_len, dim]
        #print("path_mask: ",path_mask)

        # encoded_instructions : list(data) of np.ndarray, shape=(instruction_words,)
        # instruction_words can vary.
        instr_seq, _, _ = batch_instructions_from_encoded(encoded_instructions, self.max_instruction_length)
        #seq, seq_mask, seq_lengths

        # instr_seq: torch.Tensor, shape=(data,)
        # batched_action_embeddings : list[time_sequence], Tensor(batch_size, feature_dim)
        # batched_image_features    : list[time_sequence], Tensor(batch_size, 32, feature_dim)

        ctx,h_t,c_t = speak_encoder(batched_action_embeddings, batched_image_features)

        w_t = try_cuda(Variable(torch.from_numpy(np.full((batch_size,), vocab_bos_idx, dtype='int64')).long(),
                                requires_grad=False))
        ended = np.array([False] * batch_size)

        perm_indices = np.arange(len(path_obs))
        #print(len(perm_indices), batch_size)
        assert len(perm_indices) == batch_size
        #import ipdb; ipdb.set_trace()
        #start_obs = [obs[0] for obs in path_obs]
        #start_obs = path_obs

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

        # Do a sequence rollout and calculate the loss
        loss,logit = None,None
        sequence_scores = try_cuda(torch.zeros(batch_size))
        #print("self.max_instruction_length",self.max_instruction_length)
        # h_t : torch.Size([batch, 512])
        # c_t : torch.Size([batch, 512])
        # ctx : torch.Size([batch, seq_len, 512])
        # path_mask : torch.Size([batch, 1])
        for t in range(self.max_instruction_length):
            #logger.debug("speak rollout t:%d"%t)
            h_t,c_t,alpha,sublogit = self.speak_decoder[0](w_t.view(-1, 1), h_t, c_t, ctx, path_mask)
            # Supervised training

            # BOS are not part of the encoded sequences
            target = instr_seq[:,t].contiguous()
            #print("target.shape",target.shape)
            #print(target.data)

            # Determine next model inputs
            if feedback == 'teacher':
                w_t = target
            elif feedback == 'argmax':
                _,w_t = sublogit.max(1)        # student forcing - argmax
                w_t = w_t.detach()
            elif feedback == 'sample':
                probs = F.softmax(sublogit)    # sampling an action from model
                m = D.Categorical(probs)
                w_t = m.sample()
                #w_t = probs.multinomial(1).detach().squeeze(-1)
            elif feedback == 'teacher+sample':
                if self.delta > torch.rand([1]):
                    probs = F.softmax(sublogit)    # sampling an action from model
                    m = D.Categorical(probs)
                    w_t = m.sample()
                else:
                    w_t = target
            else:
                raise NameError('Invalid feedback option: %s'%feedback)

            # log_probs.shape   == (minibatch, words)
            # log_prob_word_scores.shape == (minibatch,)
            log_probs = F.log_softmax(sublogit, dim=1)
            #log_prob_word_scores = -F.nll_loss(log_probs, w_t, ignore_index=vocab_pad_idx, reduction='none')
            subloss = F.nll_loss(log_probs, target, ignore_index=vocab_pad_idx, reduce=False, size_average=False)
            loss = subloss if loss is None else loss+subloss

            # compute logit for debug and FAST
            assert sublogit.shape[0]==w_t.shape[0],(sublogit.shape,w_t.shape)
            sublogit = sublogit.detach()
            w_t2 = w_t.detach()
            logit = sublogit[np.arange(w_t2.shape[0]),w_t2] if logit is None else logit+sublogit[np.arange(w_t2.shape[0]),w_t2]

            if not self.is_train:
                log_prob_word_scores = -F.nll_loss(log_probs, w_t, ignore_index=vocab_pad_idx, reduce=False)
                sequence_scores += log_prob_word_scores.data
                for perm_index, src_index in enumerate(perm_indices):
                    word_idx = w_t[perm_index].item()#.data[0]
                    if not ended[perm_index]:
                        outputs[src_index]['word_indices'].append(int(word_idx))
                        outputs[src_index]['score'] = float(sequence_scores[perm_index])
                        outputs[src_index]['scores'].append(log_prob_word_scores[perm_index].data.tolist())
                    if word_idx == vocab_eos_idx:
                        ended[perm_index] = True

            # print("t: %s\tstate: %s\taction: %s\tscore: %s" % (t, world_states[0], a_t.data[0], sequence_scores[0]))

            # Early exit if all ended
            if ended.all():
                break

        for item in outputs:
            item['words'] = self.env.tokenizer.decode_sentence(item['word_indices'], break_on_eos=True, join=False)

        neg_log_prob = loss
        if self.is_train:
            loss_signs = try_cuda(torch.from_numpy(loss_signs))
            loss_beam = try_cuda(torch.from_numpy(loss_beam))
            # Note that loss is negative log likelifood of seqences
            neg_log_p = torch.sum(loss.repeat(loss_beam.shape[0],1)*loss_signs,dim=1)
            expanded_logit = torch.masked_fill(-loss,loss_beam==0,float("-inf"))
            log_Z = torch.logsumexp(expanded_logit, dim=1)
            loss = torch.mean(neg_log_p + log_Z)
        else:
            loss_beam = try_cuda(torch.from_numpy(loss_beam))
            expanded_logit = torch.masked_fill(-loss,loss_beam==0,float("-inf"))
            log_Z = torch.logsumexp(expanded_logit, dim=1)

        return outputs, loss, neg_log_prob, log_Z, logit

    def speaker_loss(self,batch_size,is_valid_numpy,
                     record_obs,record_actions,world_states,encoded_instructions,ended,
                     np_target=None,ts=None):

        # target = self._teacher_action(obs, ended)
        # np_target = target.cpu().data.numpy()
        current_batch_size = batch_size-np.sum(ended)

        # sanity check
        if self.debug:
            #import ipdb; ipdb.set_trace()
            assert2(current_batch_size,is_valid_numpy.shape[0])
            assert2(batch_size,len(record_obs))
            assert2(batch_size,len(record_actions))
            assert2(batch_size,len(world_states))
            assert2(batch_size,len(encoded_instructions))
            #import ipdb; ipdb.set_trace()
            for i in range(batch_size):
                if not ended[i]:
                    print(record_actions[i],i)
                    assert all([a>=0 for a in record_actions[i]]),i
                    assert all([a is not None for a in record_obs[i]])
                    assert isinstance(world_states[i],WorldState)

        beam_path_actions=[]
        beam_path_obs=[]
        beam_instructions=[]
        beam_world_states=[]
        loss_sign_gen = []
        loss_beam_gen = []  # store where the one agent starts and ends in the beam
        #condition=lambda a_t_1,valid,a_target: valid==1 and a_target>=0 # valid==1
        local_beam_size=0
        count=0
        for i in range(batch_size):
            beam_path_actions.append([])
            beam_path_obs.append([])
            beam_instructions.append([])
            beam_world_states.append([])
            loss_beam_start=local_beam_size
            if not ended[i]:
                for a_t_1, valid in enumerate(is_valid_numpy[count]):
                    #print(np_target)
                    #if condition(a_t_1,valid,np_target[i]):
                    if valid:
                        assert a_t_1<record_obs[i][-1]['action_embedding'].shape[0],(i,a_t_1,record_obs[i][-1]['action_embedding'].shape)
                        copy_record_actions = record_actions[i][:]
                        copy_record_actions.append(a_t_1)
                        assert ts is None or ts[i]+1==len(record_obs[i]),       (i,ts[i],len(record_obs[i]))
                        assert ts is None or ts[i]+1==len(copy_record_actions), (i,ts[i],len(copy_record_actions))
                        beam_path_actions[i].append(copy_record_actions)
                        beam_instructions[i].append(encoded_instructions[i])
                        beam_path_obs[i].append(record_obs[i][:])
                        beam_world_states[i].append(world_states[i])
                        local_beam_size+=1
                count+=1
                loss_beam_gen.append([loss_beam_start,local_beam_size])
                if self.is_train or self.analysis_text_gen or self.analysis_text_bleu:
                    loss_sign_gen.append(loss_beam_start+np_target[i])
            #loss_signs_negative.append(loss_sign_neg)
        logger.debug("create beam")
        def beam_splitter(beams):
            a=[[o[0] for o in beam] for beam in beams ]
            b=[[o[1] for o in beam] for beam in beams ]
            return a,b
        # Current batch_size can be smaller than batch_size because of omitted (finished) seqs
        # beam_size = [len(beam_ob) for beam_ob in beam_path_obs]

        # sanity check
        for i, (path_obs, actions) in enumerate(zip(flatten(beam_path_obs), flatten(beam_path_actions))):
            # don't include the last state, which should result after the stop action
            assert len(path_obs) == len(actions) , (i, len(path_obs) , len(actions) )
            for t_, (pbs, a) in enumerate(zip(path_obs, actions)):
                possible_actions = pbs['action_embedding'].shape[0]
                assert a<possible_actions,(t,t_,a,possible_actions)
        logger.debug("sanity")

        flat_beam_path_obs = flatten(beam_path_obs)
        flat_beam_path_actions = flatten(beam_path_actions)
        flat_beam_instructions = flatten(beam_instructions)
        assert local_beam_size == len(flat_beam_path_obs)

        logger.debug("batches as batch_size, current_batch_size, local_beam_size: %d %d %d" \
                     %(batch_size, current_batch_size, local_beam_size))

        if self.is_train:
            try:
            #if True:
                # if self.torch_version>1.2:
                #     loss_signs = np.array(flatten(loss_signs),dtype=np.bool)
                #import ipdb; ipdb.set_trace()
                assert batch_size        >=len(loss_sign_gen), (local_beam_size,loss_sign_gen)
                assert current_batch_size==len(loss_sign_gen), (local_beam_size,loss_sign_gen)
                loss_signs = np.zeros([current_batch_size,local_beam_size])
                loss_signs[np.arange(current_batch_size),loss_sign_gen]=1
                loss_beam  = [[0]*s+[1]*(e-s)+[0]*(local_beam_size-e) for s,e in loss_beam_gen]

                if self.torch_version>1.2:
                    loss_signs = np.array(loss_signs,dtype=np.int32)
                    loss_beam  = np.array(loss_beam,dtype=np.int32)
                else:
                    loss_signs = np.array(loss_signs,dtype=np.float32)
                    loss_beam  = np.array(loss_beam,dtype=np.float32)
                # sanity check
                #if logger.getEffectiveLevel()<=20:
                if True:
                    assert loss_signs.shape==(current_batch_size,local_beam_size)
                    assert loss_beam.shape ==(current_batch_size,local_beam_size)
                    for i,p in enumerate(np.argmax(loss_signs,axis=1)):
                        assert loss_beam[i,p]==1,(i,p,loss_beam[i],loss_signs[i])
                    assert (loss_beam.sum(axis=0)==np.ones(local_beam_size)).all(), loss_beam.sum(axis=0)
            except Exception as e:
                import ipdb; ipdb.set_trace()
                pass

            outputs, speak_loss, neg_log_prob, log_Z, logit = self._score_obs_actions_and_instructions_speaker(
                self.speak_encoder[0], flat_beam_path_obs, flat_beam_path_actions, flat_beam_instructions,
                loss_signs, loss_beam,
                feedback="teacher")
        else:
            loss_beam  = [[0]*s+[1]*(e-s)+[0]*(local_beam_size-e) for s,e in loss_beam_gen]
            if self.torch_version>1.2:
                loss_beam  = np.array(loss_beam,dtype=np.int32)
            else:
                loss_beam  = np.array(loss_beam,dtype=np.float32)
            logger.debug("batch_size to speaker: %d"%len(flat_beam_path_obs))
            outputs, speak_loss, neg_log_prob, log_Z, logit = self._score_obs_actions_and_instructions_speaker(
                self.speak_encoder[0], flat_beam_path_obs, flat_beam_path_actions, flat_beam_instructions,
                None, loss_beam,
                feedback="teacher")

        logger.debug("loss speaker: "+str(speak_loss))

        # re-construct beam (un-flatten)
        np_log_Z = log_Z.cpu().data.numpy()
        np_neg_log_prob = neg_log_prob.cpu().data.numpy()
        assert np_log_Z.shape==(current_batch_size,), np_log_Z.shape
        assert np_neg_log_prob.shape==(local_beam_size,), np_neg_log_prob.shape
        assert logit.shape==(local_beam_size,), logit.shape
        beam_outputs = [[] for _ in beam_path_obs]
        beam_log_Z = [[] for _ in beam_path_obs]
        beam_neg_log_prob = [[] for _ in beam_path_obs]
        beam_logp,beam_logits,beam_unlogits = [],[],[]
        c = 0
        j = 0
        for i,path_obs in enumerate(beam_path_obs):
            if ended[i]: continue
            logits = []
            for _ in path_obs:
                beam_outputs[i].append(outputs[c])
                beam_log_Z[i].append(np_log_Z[j])
                beam_neg_log_prob[i].append(np_neg_log_prob[c])
                logits.append(logit[c])
                c+=1
            # print(c,i,beam_neg_log_prob[i],beam_log_Z[i])
            beam_log_Z[i] = np.array(beam_log_Z[i])
            beam_neg_log_prob[i] = np.array(beam_neg_log_prob[i])
            # print(c,i,beam_neg_log_prob[i],beam_log_Z[i])
            speak_logp = -(beam_neg_log_prob[i]+beam_log_Z[i])
            unlogits = - beam_neg_log_prob[i]    # logit constructed from logp + logZ
            #if logger.getEffectiveLevel()<=20:
            beam_logp.append(speak_logp)
            beam_logits.append(logits)
            beam_unlogits.append(unlogits)
            j+=1
        assert local_beam_size==len(outputs), (local_beam_size,len(outputs))
        assert local_beam_size==c, (local_beam_size,c)
        assert current_batch_size==j, (current_batch_size,j)

        #logger.debug("re-beamed")
        if self.analysis_entropy or self.analysis_text_score:
            for i,outputs_ in enumerate(beam_outputs):
                log_prob_action_word = np.array([out_["scores"] for out_ in outputs_])  # [actions, words]
                if len(log_prob_action_word)==0: continue
                prob_action_word = np_softmax(log_prob_action_word,dim=0)   # We use softmax for nomalization over actions
                word_ent = -np.sum(prob_action_word*np.log(prob_action_word),axis=0)
                #self.word_entropy
                beam_outputs[i][0]["word_entropy"] = word_ent
                beam_outputs[i][0]["word_log_prob"] = log_prob_action_word

        m = -1e20
        valid_actions=is_valid_numpy.shape[1]
        speak_logps, speak_logits, speak_unlogits = [],[],[]
        assert2(len(beam_logp),len(beam_logits))
        assert2(len(beam_logp),len(beam_unlogits))
        for scores,logits,unlogits in zip(beam_logp,beam_logits,beam_unlogits):
            scores = scores.tolist() + [m for _ in range(valid_actions-len(scores))]
            logits = logits + [m for _ in range(valid_actions-len(logits))]
            unlogits = unlogits.tolist() + [m for _ in range(valid_actions-len(unlogits))]
            speak_logps.append(scores)
            speak_logits.append(logits)
            speak_unlogits.append(unlogits)
        speak_logps = try_cuda(torch.from_numpy(np.array(speak_logps,dtype=np.float32)))
        speak_logits = try_cuda(torch.from_numpy(np.array(speak_logits,dtype=np.float32)))
        speak_unlogits = try_cuda(torch.from_numpy(np.array(speak_unlogits,dtype=np.float32)))

        if not self.is_train:
            if self.analysis_text_gen or self.analysis_text_bleu:
                outputs_, speak_loss_, neg_log_prob_, log_Z_, logit = self._score_obs_actions_and_instructions_speaker(
                    self.speak_encoder[0], flat_beam_path_obs, flat_beam_path_actions, flat_beam_instructions,
                    loss_signs=None, loss_beam=loss_beam,
                    feedback="argmax")
                #print(t)
                assert loss_sign_gen[-1] < loss_beam.shape[1],(loss_sign_gen[-1] , loss_beam.shape[1])
                #import ipdb; ipdb.set_trace()
                movements = np.concatenate([score==np.max(score) for score in beam_unlogits ])
                assert movements.shape[0]==len(outputs)
                for i,(item,item_) in enumerate(zip(outputs,outputs_)):
                    item["gen_text"] = item_["words"]
                    item["correct"] = 1 if i in loss_sign_gen else 0
                    item["movement"] = 1 if movements[i] else 0
                    #if item["instr_id"]=="1990_2":
                    #    print(item['instr_id']," ".join(item["words"]))

        return speak_loss, speak_logps, speak_logits, speak_unlogits, beam_outputs, beam_neg_log_prob

    def follow_loss(self,decoder,obs,fs,f_t_list,all_u_t,is_valid):
        fs["h_t"], fs["c_t"], fs["alpha"], fs["follow_logit"], fs["alpha_v"] = decoder(
            fs["u_t_prev"], all_u_t, f_t_list[0], fs["h_t"], fs["c_t"], fs["ctx"], fs["seq_mask"],
            )
        # Mask outputs of invalid actions
        fs["follow_logit"][is_valid == 0] = -float('inf')
        return fs

    def _rollout_with_loss(self):
        initial_world_states = self.env.reset(sort=True)
        initial_obs = self.env.observe(initial_world_states, is_train=self.is_train)
        initial_obs = np.array(initial_obs)
        batch_size = len(initial_obs)
        history_path = [[ob['viewpoint']] for ob in initial_obs] # required when follow_detailed_path

        encoded_instructions, bert_encoded_instructions = self._extract_encoded_instructions(initial_obs)
        #seq, seq_mask, seq_lengths = self._proc_batch(initial_obs)

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        # TODO consider not feeding this into the decoder, and just using attention

        self.loss = 0

        feedback = self.feedback
        loss_type = self.loss_type

        logger.debug("rollout : loss_type="+self.loss_type)
        logger.debug("rollout : compurte follower encoder")
        if not self.loss_type == "speaker":
            follower_states=[]
            if self.bert:
                # get mask and lengths
                seq, seq_mask, seq_lengths = batch_instructions_from_encoded(
                    bert_encoded_instructions,
                    self.max_instruction_length, reverse=self.reverse_instruction)
                encoded_layers = self.encoder[0](seq, seq_lengths, fix_emb=self.fix_emb)
                if self.bert_orig:
                    last_hidden = encoded_layers[0]   # for pytorch-pretrained-bert
                else:
                    last_hidden = encoded_layers
                if self.fix_bert:
                    last_hidden = last_hidden.detach()
                hidden_dim=last_hidden.size(2)
                seq_max=np.max(seq_lengths)
                np_seq_lengths = np.array(seq_lengths)
                ctx = last_hidden[:,:seq_max,:512]
                h_t = last_hidden[np.arange(last_hidden.size(0)),np_seq_lengths-1,hidden_dim-512:]
                c_t = last_hidden[:,0,hidden_dim-512:]
            else:
                # get mask and lengths
                seq, seq_mask, seq_lengths = batch_instructions_from_encoded(
                    encoded_instructions,
                    self.max_instruction_length, reverse=self.reverse_instruction)
                ctx,h_t,c_t = self.encoder[0](seq, seq_lengths)
            follower_states.append(FollowerState(ctx,seq_mask,
                           h_t,c_t,None,None,None,
                           None,None,None))
            #ctx,seq_mask,h_t,c_t=None,None,None,None
        #import ipdb; ipdb.set_trace()
        #else:
        #    print(loss_type,len(loss_type),type(loss_type))
        #    raise

        #init_ctx = ctx.clone()
        #init_h_t = h_t.clone()
        #init_c_t = c_t.clone()

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'trajectory': [path_element_from_observation(ob)],
            'actions': [],
            'scores': [],
            'observations': [ob],
            'instr_encoding': ob['instr_encoding'],
            'gen_texts': [],
            'gen_texts_gold': [],
        } for ob in initial_obs]
        #'word_entropy': [],

        if self.analysis_text_gen or self.analysis_text_score:
            #EXTRACT_IDX = ["1990_0"]
            EXTRACT_IDX = self.analysis_examples
            EXTRACT_I = []
            for i,ob in enumerate(initial_obs):
                if ob["instr_id"] in EXTRACT_IDX:
                    EXTRACT_I.append(i)
                    print("GOLD:",ob["instructions"])
                    break
            else:
                return traj

        obs = initial_obs
        world_states = initial_world_states
        record_obs = [[ob] for ob in initial_obs]
        record_actions = [[] for ob in initial_obs]

        # Initial action
        if not self.loss_type == "speaker":
            follower_states[0]["u_t_prev"] = self.decoder[0].u_begin.expand(batch_size, -1)  # init action
        ended = np.array([False] * batch_size)

        # Do a sequence rollout and calculate the loss
        sequence_scores = np.zeros((batch_size,))
        # if self.is_train:
        #     sequence_scores = np.zeros(batch_size)
        # else:
        #     sequence_scores = try_cuda(torch.zeros(batch_size))
        action_scores = np.zeros((batch_size,))
        #print("self.episode_len",self.episode_len)
        #print("feedback",feedback)
        #max_episode_len = 8 if feedback == "sample" else self.episode_len # TODO: self.episode_len was 10. CUDA_MEMORY_ERROR
        max_episode_len = self.episode_len # TODO: self.episode_len was 10. CUDA_MEMORY_ERROR
        for t in range(max_episode_len):
            #print(t)
            f_t_list = self._feature_variables(obs) # Image features from obs
            all_u_t, is_valid, is_valid_numpy = self._action_variable(obs)
            logger.debug("rollout t:%d"%t+", valid_shape:%s"%str(is_valid_numpy.shape))
            np_target, np_target_shrink = self._teacher_action_np(obs, ended)

            assert len(f_t_list) == 1, 'for now, only work with MeanPooled feature'
            is_valid_numpy_notended = is_valid_numpy[np.logical_not(ended)]
            follow_logit = None
            if "follower" in loss_type:
                follower_states[0] = self.follow_loss(self.decoder[0],obs,follower_states[0],
                                                      f_t_list,all_u_t,is_valid)
                # Note that we assume that this logit is used as argmax!
                # We assume that this logit is NOT used for training nor sampling.
                follow_logit=torch.sum(torch.cat([fs["follow_logit"].unsqueeze(dim=-1) for fs in follower_states]),dim=-1)
                follow_logit_not_ended = follow_logit[np.logical_not(ended),:]

                target = self._teacher_action(obs, ended)

            if "speaker" in self.loss_type:
                speak_loss, speak_log_p, _, speak_logit, beam_outputs, beam_neg_log_prob = self.speaker_loss(
                    batch_size, is_valid_numpy_notended,
                    record_obs,record_actions,world_states,encoded_instructions,
                    ended, np_target, ts=[t for _ in range(batch_size)])

            if not self.is_train:
                if self.analysis_text_gen or self.analysis_text_score:
                    print("# t",t)
                    for i in EXTRACT_I:
                        print(path_element_from_observation(obs[i])) # the last two position would be same because of STOP.
                    for beam in beam_outputs:
                        for item in beam:
                            if item["instr_id"] in EXTRACT_IDX:
                                if self.analysis_text_gen:
                                    print(item['instr_id'],item['correct'],item['movement']," ".join(item["gen_text"]))
                                if self.analysis_text_score:
                                    #import ipdb; ipdb.set_trace()
                                    print(" ".join(item["words"]))
                                    print(" ".join(item["scores"]))
                if self.analysis_text_bleu:
                    gen_texts = [
                        [item["gen_text"] for item in beam] \
                        for beam in beam_outputs]
                    gen_texts_gold = [
                        [item["gen_text"] for item in beam if item["correct"]==1] \
                        for beam in beam_outputs]
                    for beam in gen_texts_gold:
                        assert len(beam) in [0,1],len(beam)
                    gen_texts_gold = [beam[0] if len(beam)>0 else None for beam in gen_texts_gold ]
                if self.analysis_entropy:
                    words = [
                        beam[0]["words"] for beam in beam_outputs  if len(beam)>0]
                    word_log_prob = [
                        beam[0]["word_log_prob"] for beam in beam_outputs  if len(beam)>0]
                    word_entropy = [
                        beam[0]["word_entropy"]  for beam in beam_outputs  if len(beam)>0]

            if self.loss_type == "follower":
                follow_loss = self.criterion(follow_logit, target)
                if self.is_train:
                    self.loss += follow_loss
                logit = follow_logit_not_ended
            elif self.loss_type == "speaker":
                if self.is_train:
                    self.loss += speak_loss
                #for log_probs, mini_target in gold_loss_outputs:
                #    mini_loss = F.nll_loss(log_probs, mini_target, ignore_index=vocab_pad_idx, reduce=True, size_average=True)
                #    self.loss += mini_loss
                logit = speak_logit
            else:
                #follow_logit = torch.sum([fs["follow_logit"] for fs in follower_states],dim=-1)
                follow_loss = self.criterion(follow_logit, target)
                if self.is_train:
                    self.loss += speak_loss + follow_loss

                speak_score = speak_logit
                log_follow_logit = self.logsm(follow_logit)
                log_speak_logit  = self.logsm(speak_score)
                # log_follow_logit has -inf for masks
                # avoid -inf * 0 = nan
                if self.beta==0.0:
                    log_logit = log_follow_logit[np.logical_not(ended),:]
                elif self.beta==1.0:
                    log_logit = log_speak_logit
                else:
                    log_logit = self.beta*log_speak_logit + (1-self.beta)*log_follow_logit[np.logical_not(ended),:]
                logit = torch.exp(log_logit)
                valid_actions=is_valid_numpy.shape[1]
                current_batch_size = batch_size-np.sum(ended)
                assert logit.shape == torch.Size([current_batch_size, valid_actions]), (logit.shape, [current_batch_size, valid_actions])

            # Export for the statistical analyses
            if self.explore_log is not None:
                is_valid_numpy_notended = is_valid_numpy[np.logical_not(ended)]
                if "follower" in loss_type:
                    np_follow_logit = follow_logit[np.logical_not(ended),:].cpu().data.numpy()
                    _,follow_a_t = follow_logit[np.logical_not(ended),:].max(1)        # student forcing - argmax
                    npFollow_a_t = follow_a_t.cpu().data.numpy()
                else:
                    npFollow_a_t = 0
                    np_follow_logit = 0
                if "speaker" in loss_type:
                    _,speak_a_t = speak_log_p.max(1)        # student forcing - argmax
                    npSpeak_a_t = speak_a_t.cpu().data.numpy()
                    np_speak_log_p = speak_log_p.cpu().data.numpy()
                else:
                    npSpeak_a_t = 0
                    beam_neg_log_prob = 0
                    speak_log_p = 0
                if "speaker" in loss_type and "follower" in loss_type:
                    _,beta_a_t = log_logit.max(1)        # student forcing - argmax
                    npbeta_a_t = beta_a_t.cpu().data.numpy()
                else:
                    npbeta_a_t = 0
                list_instr_id = [ob["instr_id"] for ob in obs]
                list_location = [path_element_from_observation(ob) for ob in obs]
                list_goal     = [item['path'][-1] for item in self.env.batch]
                gts = [self.env.gt[int(instr_id.split('_')[0])] for instr_id in list_instr_id]
                nav_errors    = [self.env.distances[gt['scan']][location[0]][goal] for gt,location,goal in zip(gts,list_location,list_goal)]
                words2 = [ob["instructions"] for ob in obs]
                self.explore_log.append([t,np_target,np_target_shrink,npSpeak_a_t,npFollow_a_t,npbeta_a_t,
                                         beam_neg_log_prob,np_speak_log_p,np_follow_logit,is_valid_numpy_notended,
                                         ended.copy(), words, words2, word_log_prob, word_entropy,
                                         list_instr_id, list_location, list_goal, nav_errors,
                                         ])

            # Determine next model inputs
            if feedback == 'teacher':
                # turn -1 (ignore) to 0 (stop) so that the action is executable
                #a_t = torch.clamp(target, min=0)
                a_t = np_target_shrink
            elif feedback == 'argmax':
                _,a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
                a_t = a_t.cpu().data.numpy()
            elif feedback == 'sample':
                probs = F.softmax(logit, dim=1)    # sampling an action from model
                # Further mask probs where agent can't move forward
                # Note input to `D.Categorical` does not have to sum up to 1
                # http://pytorch.org/docs/stable/torch.html#torch.multinomial
                is_valid_shrinked = is_valid[np.logical_not(ended),:]
                probs[is_valid_shrinked == 0] = 0.
                m = D.Categorical(probs)
                a_t = m.sample()
                a_t = a_t.cpu().data.numpy()
            elif feedback == 'teacher+sample':
                if self.delta > torch.rand([1]):
                    probs = F.softmax(logit, dim=1)    # sampling an action from model
                    # Further mask probs where agent can't move forward
                    # Note input to `D.Categorical` does not have to sum up to 1
                    # http://pytorch.org/docs/stable/torch.html#torch.multinomial
                    is_valid_shrinked = is_valid[np.logical_not(ended),:]
                    probs[is_valid_shrinked == 0] = 0.
                    m = D.Categorical(probs)
                    a_t = m.sample()
                    a_t = a_t.cpu().data.numpy()
                else:
                    a_t = np_target_shrink
            else:
                raise NameError('Invalid feedback option: %s'%feedback)

            if not self.is_train:
                #torch_a_t = try_cuda(Variable(a_t, requires_grad=False))
                #action_scores = -F.cross_entropy(logit, a_t, ignore_index=-1, reduce=False).data
                #action_scores = -F.cross_entropy(logit, torch_a_t, ignore_index=-1, reduce=False)
                #action_scores = action_scores.cpu().data.numpy()
                np_logit = logit.cpu().data.numpy()
                action_scores = np_logit[np.arange(np_logit.shape[0]),a_t]
                j=0
                conv=[]
                for i in range(batch_size):
                    conv.append(0 if ended[i] else action_scores[j])
                    if not ended[i]:
                        j+=1
                sequence_scores += np.array(conv)

            # dfried: I changed this so that the ended list is updated afterward; this causes <end> to be added as the last action, along with its score, and the final world state will be duplicated (to more closely match beam search)
            # Make environment action
            env_action = []
            j=0
            for i in range(batch_size):
                if ended[i]:
                    action_idx = 0 # obvious?
                else:
                    action_idx = a_t[j].item()#.data[0]
                    j+=1
                env_action.append(action_idx)
            assert batch_size==len(env_action)

            # update the previous action
            if not loss_type == "speaker":
                # TODO: re-beam with ended (of speaker) for speaker+follower
                # TODO: need expanson of a_t to batch_size
                #torch_a_t = try_cuda(Variable(action_idx, requires_grad=False))
                np_a_t = np.array(env_action,dtype=np.int)
                follower_states[0]["u_t_prev"] = all_u_t[np.arange(batch_size), np_a_t, :].detach()

            # sanity check
            # print("t: %s\tstate: %s\taction: %s\tscore: %s" % (t, world_states[0], a_t.data[0], sequence_scores[0]))
            assert len(obs)==len(record_obs), "%d %d"%(len(obs),len(record_obs))
            assert batch_size==len(obs)
            assert batch_size==len(env_action)
            j=0
            for i,ob in enumerate(obs):
                if ended[i]: continue
                try:
                    assert len(ob['action_embedding'])>env_action[i] and env_action[i]>=0 , (env_action[i], ob['action_embedding'])
                except:
                    import ipdb; ipdb.set_trace()
                j+=1

            #logger.debug("step")

            world_states = self.env.step(world_states, env_action, obs)
            assert len(world_states)==len(history_path), (len(world_states)==len(history_path))
            obs = self.env.observe(world_states, is_train=self.is_train, history_path=history_path)
            for i in range(batch_size):
                if not ended[i]:
                    history_path[i].append(obs[i]['viewpoint'])

            # record for the next transition
            for i,ob in enumerate(obs):
                record_obs[i].append(ob)
                record_actions[i].append(env_action[i])
                #print("a_t",a_t)
            #print()

            # Save trajectory output
            if self.analysis_text_bleu and not self.is_train:
                assert2(len(gen_texts),batch_size)
                assert2(len(gen_texts_gold),batch_size)
            if self.analysis_text_bleu and not self.is_train:
                assert2(len(beam_outputs),batch_size-sum(ended))
            if not self.is_train:
                j=0
                for i,ob in enumerate(obs):
                    if not ended[i]:
                        traj[i]['trajectory'].append(path_element_from_observation(ob)) # the last two position would be same because of STOP.
                        if self.is_train:
                            traj[i]['score'] = sequence_scores[i]
                        else:
                            traj[i]['score'] = sequence_scores[i]
                        traj[i]['scores'].append(action_scores[j])
                        #traj[i]['actions'].append(a_t.data[i])
                        traj[i]['actions'].append(env_action[i])
                        traj[i]['observations'].append(ob)
                        if self.analysis_text_bleu and not self.is_train:
                            traj[i]['gen_texts'].append(gen_texts[i])
                            traj[i]['gen_texts_gold'].append(gen_texts_gold[i])
                        #if self.analysis_entropy:
                        #    traj[i]['word_entropy'].append(beam_outputs[j][0]["word_entropy"])
                        j+=1

            # Update ended list
            for i in range(batch_size):
                if not ended[i]:
                    #action_idx = a_t[i].item()#.data[0]
                    action_idx = env_action[i]#.data[0]
                    if action_idx == 0:
                        ended[i] = True

            # Early exit if all ended
            if ended.all():
                break

        #self.losses.append(self.loss.data[0] / self.episode_len)
        # shouldn't divide by the episode length because of masking
        if self.is_train:
            self.losses.append(self.loss.item())#.data[0])
        return traj

    # we don't support beam search with generative language grounded policy.
    def beam_search(self, beam_size, load_next_minibatch=True, mask_undo=False):
        assert self.env.beam_size >= beam_size
        world_states = self.env.reset(sort=True, beamed=True, load_next_minibatch=load_next_minibatch)
        obs = self.env.observe(world_states, beamed=True)
        batch_size = len(world_states)

        # get mask and lengths
        seq, seq_mask, seq_lengths = self._proc_batch(obs, beamed=True)

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        ctx,h_t,c_t = self.encoder(seq, seq_lengths)

        completed = []
        for _ in range(batch_size):
            completed.append([])

        beams = [
            [InferenceState(prev_inference_state=None,
                            world_state=ws[0],
                            observation=o[0],
                            flat_index=i,
                            last_action=-1,
                            last_action_embedding=self.decoder.u_begin.view(-1),
                            action_count=0,
                            score=0.0, h_t=None, c_t=None, last_alpha=None)]
            for i, (ws, o) in enumerate(zip(world_states, obs))
        ]

        # Do a sequence rollout and calculate the loss
        for t in range(self.episode_len):
            flat_indices = []
            beam_indices = []
            u_t_list = []
            for beam_index, beam in enumerate(beams):
                for inf_state in beam:
                    beam_indices.append(beam_index)
                    flat_indices.append(inf_state.flat_index)
                    u_t_list.append(inf_state.last_action_embedding)

            u_t_prev = torch.stack(u_t_list, dim=0)
            assert len(u_t_prev.shape) == 2
            flat_obs = flatten(obs)
            f_t_list = self._feature_variables(flat_obs) # Image features from obs
            all_u_t, is_valid, is_valid_numpy = self._action_variable(flat_obs)

            assert len(f_t_list) == 1, 'for now, only work with MeanPooled feature'

            h_t, c_t, alpha, logit, alpha_v = self.decoder(
                u_t_prev, all_u_t, f_t_list[0], h_t, c_t, ctx, seq_mask)

            # Mask outputs of invalid actions
            logit[is_valid == 0] = -float('inf')
            # # Mask outputs where agent can't move forward
            # no_forward_mask = [len(ob['navigableLocations']) <= 1 for ob in flat_obs]

            if mask_undo:
                masked_logit = logit.clone()
            else:
                masked_logit = logit

            log_probs = F.log_softmax(logit, dim=1).data

            # force ending if we've reached the max time steps
            # if t == self.episode_len - 1:
            #     action_scores = log_probs[:,self.end_index].unsqueeze(-1)
            #     action_indices = torch.from_numpy(np.full((log_probs.size()[0], 1), self.end_index))
            # else:
            #action_scores, action_indices = log_probs.topk(min(beam_size, logit.size()[1]), dim=1)
            _, action_indices = masked_logit.data.topk(min(beam_size, logit.size()[1]), dim=1)
            action_scores = log_probs.gather(1, action_indices)
            assert action_scores.size() == action_indices.size()

            start_index = 0
            new_beams = []
            assert len(beams) == len(world_states)
            all_successors = []
            for beam_index, (beam, beam_world_states, beam_obs) in enumerate(zip(beams, world_states, obs)):
                successors = []
                end_index = start_index + len(beam)
                assert len(beam_world_states) == len(beam)
                assert len(beam_obs) == len(beam)
                if beam:
                    for inf_index, (inf_state, world_state, ob, action_score_row, action_index_row) in \
                            enumerate(zip(beam, beam_world_states, beam_obs, action_scores[start_index:end_index], action_indices[start_index:end_index])):
                        flat_index = start_index + inf_index
                        for action_score, action_index in zip(action_score_row, action_index_row):
                            if is_valid_numpy[flat_index, action_index] == 0:
                                continue
                            successors.append(
                                InferenceState(prev_inference_state=inf_state,
                                               world_state=world_state, # will be updated later after successors are pruned
                                               observation=ob, # will be updated later after successors are pruned
                                               flat_index=flat_index,
                                               last_action=action_index,
                                               last_action_embedding=all_u_t[flat_index, action_index].detach(),
                                               action_count=inf_state.action_count + 1,
                                               score=float(inf_state.score + action_score), h_t=None, c_t=None,
                                               last_alpha=alpha[flat_index].data)
                            )
                start_index = end_index
                successors = sorted(successors, key=lambda t: t.score, reverse=True)[:beam_size]
                all_successors.append(successors)

            successor_world_states = [
                [inf_state.world_state for inf_state in successors]
                for successors in all_successors
            ]

            successor_env_actions = [
                [inf_state.last_action for inf_state in successors]
                for successors in all_successors
            ]

            successor_last_obs = [
                [inf_state.observation for inf_state in successors]
                for successors in all_successors
            ]

            successor_world_states = self.env.step(successor_world_states, successor_env_actions, successor_last_obs, beamed=True)
            successor_obs = self.env.observe(successor_world_states, beamed=True)

            all_successors = structured_map(lambda inf_state, world_state, obs: inf_state._replace(world_state=world_state, observation=obs),
                                   all_successors, successor_world_states, successor_obs, nested=True)

            # if all_successors[0]:
            #     print("t: %s\tstate: %s\taction: %s\tscore: %s" % (t, all_successors[0][0].world_state, all_successors[0][0].last_action, all_successors[0][0].score))

            for beam_index, successors in enumerate(all_successors):
                new_beam = []
                for successor in successors:
                    if successor.last_action == 0 or t == self.episode_len - 1:
                        completed[beam_index].append(successor)
                    else:
                        new_beam.append(successor)
                if len(completed[beam_index]) >= beam_size:
                    new_beam = []
                new_beams.append(new_beam)

            beams = new_beams

            world_states = [
                [inf_state.world_state for inf_state in beam]
                for beam in beams
            ]

            obs = [
                [inf_state.observation for inf_state in beam]
                for beam in beams
            ]

            # Early exit if all ended
            if not any(beam for beam in beams):
                break

        trajs = []

        for this_completed in completed:
            assert this_completed
            this_trajs = []
            for inf_state in sorted(this_completed, key=lambda t: t.score, reverse=True)[:beam_size]:
                path_states, path_observations, path_actions, path_scores, path_attentions = backchain_inference_states(inf_state)
                # this will have messed-up headings for (at least some) starting locations because of
                # discretization, so read from the observations instead
                ## path = [(obs.viewpointId, state.heading, state.elevation)
                ##         for state in path_states]
                trajectory = [path_element_from_observation(ob) for ob in path_observations]
                this_trajs.append({
                    'instr_id': path_observations[0]['instr_id'],
                    'instr_encoding': path_observations[0]['instr_encoding'],
                    'trajectory': trajectory,
                    'observations': path_observations,
                    'actions': path_actions,
                    'score': inf_state.score,
                    'scores': path_scores,
                    'attentions': path_attentions
                })
            trajs.append(this_trajs)
        traversed_lists = None # todo
        return trajs, completed, traversed_lists


    # from FAST
    def _search_collect(self, batch_queue, wss, current_idx, ended):
        cand_wss = []
        cand_acs = []
        for idx,_q in enumerate(batch_queue):
            _wss = [wss[idx]]
            _acs = [0]
            _step = current_idx[idx]
            while not ended[idx] and _step > 0:
                _wss.append(_q.queue[_step].world_state)
                _acs.append(_q.queue[_step].action)
                _step = _q.queue[_step].father
            cand_wss.append(list(reversed(_wss)))
            cand_acs.append(list(reversed(_acs)))
        return cand_wss, cand_acs

    def _wss_to_obs(self, cand_wss, instr_ids):
        cand_obs = []
        for _wss,_instr_id in zip(cand_wss, instr_ids):
            ac_len = len(_wss)
            #cand_obs.append(self.env.observe(_wss, instr_id=_instr_id))
            cand_obs.append(self.env.observe(_wss))
        return cand_obs

    def _rollout_with_search(self):
        self.soft_align = False
        self.prog_monitor = False
        self.want_loss = False
        self.clean_results = {}
        #if self.env.notTest:
        #    self._init_loss()
        #    ce_criterion = self.criterion
        #    pm_criterion = self.pm_criterion
        #    bt_criterion = nn.CrossEntropyLoss(ignore_index=-1)

        initial_world_states = self.env.reset(sort=True)
        initial_obs = self.env.observe(initial_world_states)
        batch_size = len(initial_obs)

        seq, seq_mask, seq_lengths = self._proc_batch(initial_obs)
        #ctx,h_t,c_t = self.encoder[0](seq, seq_lengths)
        encoded_instructions, bert_encoded_instructions = self._extract_encoded_instructions(initial_obs)

        follower_states=[]
        if not self.loss_type == "speaker":
            if self.bert:
                # get mask and lengths
                seq, seq_mask, seq_lengths = batch_instructions_from_encoded(
                    bert_encoded_instructions,
                    self.max_instruction_length, reverse=self.reverse_instruction)
                encoded_layers = self.encoder[0](seq, seq_lengths, fix_emb=self.fix_emb)
                if self.bert_orig:
                    last_hidden = encoded_layers[0]   # for pytorch-pretrained-bert
                else:
                    last_hidden = encoded_layers
                if self.fix_bert:
                    last_hidden = last_hidden.detach()
                hidden_dim=last_hidden.size(2)
                seq_max=np.max(seq_lengths)
                np_seq_lengths = np.array(seq_lengths)
                ctx = last_hidden[:,:seq_max,:512]
                h_t_ = last_hidden[np.arange(last_hidden.size(0)),np_seq_lengths-1,hidden_dim-512:]
                c_t_ = last_hidden[:,0,hidden_dim-512:]
            else:
                # get mask and lengths
                seq, seq_mask, seq_lengths = batch_instructions_from_encoded(
                    encoded_instructions,
                    self.max_instruction_length, reverse=self.reverse_instruction)
                ctx,h_t_,c_t_ = self.encoder[0](seq, seq_lengths)
            u_t_prev = self.decoder[0].u_begin.view(-1).detach()  # init action
            # inner-batch follower_state
            # batch x followers x state_dict
            follower_states = [[FollowerState(ctx,seq_mask,
                           h_t_[idx].detach(),c_t_[idx].detach(),None,None,None,
                           u_t_prev,None,None)] \
                                    for idx in range(batch_size)]
            assert2(len(initial_world_states),len(follower_states))
        else:
            follower_states = [[None] for idx in range(batch_size)]

        traj = [{
            'instr_id': ob['instr_id'],
            'instr_encoding': ob['instr_encoding'],
            'trajectory': [path_element_from_observation(ob)],
        } for ob in initial_obs]

        clean_traj = [{
            'instr_id': ob['instr_id'],
            'trajectory': [path_element_from_observation(ob)],
        } for ob in initial_obs]

        obs = initial_obs
        world_states = initial_world_states
        record_obs = [[ob] for ob in initial_obs]
        record_actions = [[] for ob in initial_obs]
        if not self.loss_type == "follower":
            speaker_states =[[SpeakerState(\
                encoded_instructions[idx],record_obs[idx],record_actions[idx], t=0)] \
                for idx in range(batch_size)]
        else:
            speaker_states = [[None] for idx in range(batch_size)]

        batch_queue = [PriorityQueue() for _ in range(batch_size)]
        ending_queue = [PriorityQueue() for _ in range(batch_size)]

        visit_graphs = [nx.Graph() for _ in range(batch_size)]
        for ob, g in zip(initial_obs, visit_graphs): g.add_node(ob['viewpoint'])

        ended = np.array([False] * batch_size)

        for i, (ws, o) in enumerate(zip(initial_world_states, initial_obs)):
            batch_queue[i].push(
                SearchState(
                    flogit=RunningMean(),
                    flogp=RunningMean(),
                    world_state=ws,
                    observation=o,
                    action=0,
                    follower_states=follower_states[i],
                    speaker_states=speaker_states[i],
                    action_count=0,
                    father=-1),
                0)
                    #action_embedding=self.decoder[0].u_begin.view(-1).detach(),
                    #h_t=h_t[i].detach(),c_t=c_t[i].detach(),,(a,type(a))
#        import ipdb;ipdb.set_trace()

        #state_factored = [{} for _ in range(batch_size)]

        for t in range(self.episode_len):
            current_idx, priority, current_batch = \
                    zip(*[_q.pop() for _q in batch_queue])
#SearchState = namedtuple("SearchState", "flogit,flogp, world_state, observation, action, follower_states, speaker_states,
            # action_count, father") # flat_index,
            #(last_logit,last_logp,last_world_states,last_obs,acs,acs_embedding,
            #        ac_counts,prev_h_t,prev_c_t,prev_father) = zip(*current_batch)
            (last_logit,last_logp,last_world_states,last_obs,acs,follower_states,speaker_states,
             ac_counts,prev_father) = zip(*current_batch)
            #import ipdb;ipdb.set_trace()
            if "follower" in self.loss_type:
                logger.debug("%d %d "%(len(follower_states),len(follower_states[0])))
                logger.debug("%d %d %d"%(len(follower_states),len(follower_states[0]),len(follower_states[0][0])))

            #print(type(acs),[a.cpu().numpy().item() if isinstance(a,torch.Tensor) else a for a in acs])

            if t > 0:
                for i,ob in enumerate(last_obs):
                    if not ended[i]:
                        last_vp = traj[i]['trajectory'][-1]
                        traj[i]['trajectory'] += realistic_jumping(
                            visit_graphs[i], last_vp, ob)

                world_states = self.env.step(last_world_states,acs,last_obs)
                obs = self.env.observe(world_states)
                for i in range(batch_size):
                    if (not ended[i] and
                        not visit_graphs[i].has_edge(last_obs[i]['viewpoint'], obs[i]['viewpoint'])):
                        traj[i]['trajectory'].append(path_element_from_observation(obs[i]))
                        visit_graphs[i].add_edge(last_obs[i]['viewpoint'], obs[i]['viewpoint'])
                for idx, ac in enumerate(acs):
                    if ac == 0:
                        ended[idx] = True
                        batch_queue[idx].lock()
                if "speaker" in self.loss_type:
                    record_obs = [speaker_states[idx][0]["record_obs"]+[obs[idx]] for idx in range(batch_size)]

            if ended.all(): break
            logger.debug("obs: %d, ended: %d, batch: %d "%(len(obs),sum(ended),batch_size))
            #print(ended)
            #print([len(ro) for ro in record_obs])

            #u_t_prev = torch.stack(acs_embedding, dim=0)
            #prev_h_t = torch.stack(prev_h_t,dim=0)
            #prev_c_t = torch.stack(prev_c_t,dim=0)
            f_t_list = self._feature_variables(obs)
            all_u_t, is_valid, is_valid_numpy = self._action_variable(obs)
            logger.debug("rollout t:%d"%t+", valid_shape:%s"%str(is_valid_numpy.shape))
            assert len(f_t_list) == 1, 'for now, only work with MeanPooled feature'
            is_valid_numpy_notended = is_valid_numpy[np.logical_not(ended)]

            # 1. local scorer
            #h_t, c_t, t_ground, v_ground, alpha_t, logit, alpha_v = self.decoder[0](
            #    u_t_prev, all_u_t, f_t_list[0], prev_h_t, prev_c_t, ctx, seq_mask)

            if "follower" in self.loss_type:
                u_t_prev = torch.stack([fs[0]["u_t_prev"] for fs in follower_states], dim=0)
                logger.debug("u_t_prev:%s"%str(u_t_prev.shape))
                prev_h_t = torch.stack([fs[0]["h_t"] for fs in follower_states],dim=0)
                prev_c_t = torch.stack([fs[0]["c_t"] for fs in follower_states],dim=0)
                follower_state = FollowerState(ctx,seq_mask,
                              prev_h_t,prev_c_t,None,None,None,
                              u_t_prev,all_u_t,f_t_list)
                follower_state = self.follow_loss(self.decoder[0],obs,follower_state,
                                                      f_t_list,all_u_t,is_valid)
                next_follower_state = follow_state_detach_split_for_next(follower_state,all_u_t,batch_size)
                target = self._teacher_action(obs, ended)
            if "speaker" in self.loss_type:
                #import ipdb; ipdb.set_trace()
                encoded_instructions = [ss[0]["encoded_instructions"] for ss in speaker_states]
                #record_obs           = [ss[0]["record_obs"]           for ss in speaker_states]
                record_actions       = [ss[0]["record_actions"]       for ss in speaker_states]
                ts                   = [ss[0]["t"]                    for ss in speaker_states]
                np_target, np_target_shrink = self._teacher_action_np(obs, ended)
                speak_loss, speak_log_p, _, speak_logit, beam_outputs, beam_neg_log_prob = self.speaker_loss(
                    batch_size,is_valid_numpy_notended,
                    record_obs,record_actions,world_states,encoded_instructions,
                    ended,np_target,ts=ts)
                next_speaker_state =[ SpeakerState(\
                    encoded_instructions[idx],
                    record_obs[idx],
                    record_actions[idx],ts[idx]) \
                    for idx in range(batch_size)]

            if self.loss_type == "follower":
                follow_logit=follower_state["follow_logit"]
                follow_loss = self.criterion(follow_logit, target)
                if self.is_train:
                    self.loss += follow_loss
                logit = follow_logit[np.logical_not(ended),:]
            elif self.loss_type == "speaker":
                if self.is_train:
                    self.loss += speak_loss
                logit = speak_logit
            else:
                follow_logit=follower_state["follow_logit"]
                follow_loss = self.criterion(follow_logit, target)
                if self.is_train:
                    self.loss += speak_loss + follow_loss

                log_follow_logit = self.logsm(follow_logit)
                log_speak_logit  = self.logsm(speak_logit)
                #print("speak",log_speak_logit,"follow",log_follow_logit)
                #print()
                # log_follow_logit has -inf for masks
                # avoid -inf * 0 = nan
                if self.beta==0.0:
                    log_logit = log_follow_logit[np.logical_not(ended),:]
                elif self.beta==1.0:
                    log_logit = log_speak_logit
                else:
                    log_logit = self.beta*log_speak_logit + (1-self.beta)*log_follow_logit[np.logical_not(ended),:]
                logit = torch.exp(log_logit)
                valid_actions=is_valid_numpy.shape[1]
                current_batch_size = batch_size-np.sum(ended)
                assert logit.shape == torch.Size([current_batch_size, valid_actions]), (logit.shape, [current_batch_size, valid_actions])


            # 2. prog monitor
            progress_score = [0] * batch_size # We don' use this.

            # Mask outputs of invalid actions
            _logit = logit.detach()
            #_logit[is_valid == 0] = -float('inf')

            # Expand nodes
            #import ipdb; ipdb.set_trace()
            ac_lens = np.argmax(is_valid.cpu() == 0, axis=1).detach()
            ac_lens[ac_lens == 0] = is_valid.shape[1]
            ac_lens2 = np.sum(is_valid_numpy,axis=1).astype(int)
            np.testing.assert_array_equal(ac_lens,ac_lens2)#,str(ac_lens,ac_lens2,ac_lens-ac_lens2))
            assert2(ac_lens.shape[0],batch_size)

            #h_t_data, c_t_data = h_t.detach(),c_t.detach()
            #u_t_data = all_u_t.detach()
            log_prob = F.log_softmax(_logit, dim=1).detach()

            # 4. prepare ending evaluation
            cand_instr= [_traj['instr_encoding'] for _traj in traj]
            cand_wss, cand_acs = self._search_collect(batch_queue, world_states, current_idx, ended)
            instr_ids = [_traj['instr_id'] for _traj in traj]

            speaker_scores = [0] * batch_size
            #cand_obs = self._wss_to_obs(cand_wss, instr_ids)
            #if self.speaker is not None:
            #    speaker_scored_cand, _ = \
            #        self.speaker._score_obs_actions_and_instructions(
            #            cand_obs,cand_acs,cand_instr,feedback='teacher')
            #    speaker_scores = [_s['score'] for _s in speaker_scored_cand]

            goal_scores = [0] * batch_size
            '''
            if self.goal_button is not None:
                # encode text
                instr_enc_h, instr_enc_c = final_text_enc(cand_instr, self.max_instruction_length, self.goal_button.text_encoder)
                # encode traj / the last image
                traj_enc = self.goal_button.encode_traj(cand_obs,cand_acs)
                goal_scores = self.goal_button.score(instr_enc, traj_enc)
            '''

            _logit = -1/_logit
            log_prob = -1/log_prob
            if self.show_loss:
                print("t:",t,"mean of max  logits:",  torch.mean(torch.max(_logit,dim=1).values).cpu().numpy().item(),  "mean of max  posit logits:", torch.mean(torch.max(_logit,dim=1).values).cpu().numpy().item())
                print()

            jdx = 0
            for idx in range(batch_size):
                if ended[idx]: continue

                #import ipdb; ipdb.set_trace()
                _len = ac_lens[idx]
                new_logit = last_logit[idx].fork(_logit[jdx][:_len].cpu().tolist())
                new_logp = last_logp[idx].fork(log_prob[jdx][:_len].cpu().tolist())

                # entropy
                entropy = torch.sum(-log_prob[jdx][:_len] * torch.exp(log_prob[jdx][:_len]))

                # record
                #if self.env.notTest:
                #    _dev = obs[idx]['deviation']
                #    self.records[_dev].append(
                #        (ac_counts[idx],
                #         obs[idx]['progress'],
                #         _logit[idx][:_len].cpu().tolist(),
                #         log_prob[idx][:_len].cpu().tolist(),
                #         entropy.item(),
                #         speaker_scores[idx]))

                # selectively expand nodes
                K = 20
                select_k = _len if _len < K else K
                top_ac = list(torch.topk(_logit[jdx][:_len],select_k)[1])
                if self.inject_stop and 0 not in top_ac:
                    top_ac.append(0)

                # compute heuristic scores
                new_heur = new_logit

                if self.search_mean:
                    _new_heur = [_h.mean for _h in new_heur]
                else:
                    _new_heur = [_h.sum for _h in new_heur]

                visitedVps = [ws[1] for ws in cand_wss[idx]]
                for ac_idx, ac in enumerate(top_ac):
                    nextViewpointId = obs[idx]['adj_loc_list'][ac]['nextViewpointId']

                    #if (ac > 0 and nextViewpointId in visitedVps): # revisit
                    #    print("Revisited: ",t,ac)

                    # when the revisiting is the best predicted action,
                    # search the next best actions including the actions in previous states
                    if ac > 0 and nextViewpointId in visitedVps:
                        continue

                    # we always choose the action with the best new_heur score if it is not revisiting
                    if ac_idx == 0:
                        _new_heur[ac] = float('inf')

                    # STOP is added to ending_queue
                    if ac == 0:
                        ending_heur = _new_heur[ac]
                        new_ending = CandidateState(
                                flogit=new_logit[ac],
                                flogp=new_logp[ac],
                                world_states=cand_wss[idx],
                                actions=cand_acs[idx],
                                pm=progress_score[idx],
                                speaker=speaker_scores[idx],
                                scorer=_logit[jdx][ac],
                                )
                        ending_queue[idx].push(new_ending, ending_heur)

                    if "follower" in self.loss_type:
                        next_fs = [follow_state_action(next_follower_state[idx],ac)]
                    else:
                        next_fs = None
                    if "speaker" in self.loss_type:
                        next_ss = [speak_state_action(next_speaker_state[idx],ac)]
                    else:
                        next_ss = None

                    new_node = SearchState(
                        flogit=new_logit[ac],
                        flogp=new_logp[ac],
                        world_state=world_states[idx],
                        observation=obs[idx],
                        action=ac,
                        follower_states=next_fs,
                        speaker_states=next_ss,
                        action_count=ac_counts[idx]+1,
                        father=current_idx[idx])
                        # h_t=h_t_data[idx],c_t=c_t_data[idx],
                        # action_embedding=u_t_data[idx,ac],
                    batch_queue[idx].push(new_node, _new_heur[ac])

                if batch_queue[idx].size() == 0:
                    logger.info("batch_queue[%d].size() == 0, This is a corner case."%idx)
                    #print(top_ac)
                    batch_queue[idx].lock()
                    ended[idx] = True
                jdx+=1
            assert2(jdx,_logit.shape[0])

                #if ending_queue[idx].size() > 0:
                #    batch_queue[idx].lock()
                #    ended[idx] = True
            #print([bq.size() for bq in batch_queue])
            #print([bq.size() for bq in ending_queue])
#            print(np.array([len(t["trajectory"]) for t in traj]))

        # def count_inf(eq):
        #     return len([p for p,idx in eq.pri if p == float("-inf")])
        # print("ending_queue with inf:",[count_inf(ending_queue[idx]) for idx in range(batch_size)])

        # actually move the cursor
        for idx in range(batch_size):
            #import ipdb;ipdb.set_trace()
            instr_id = traj[idx]['instr_id']
            if ending_queue[idx].size() == 0:
                #print("Warning: some instr does not have ending, ",
                #        "this can be a desired behavior though")
                self.clean_results[instr_id] = {
                        'instr_id': traj[idx]['instr_id'],
                        'trajectory': traj[idx]['trajectory'],
                        }
                continue

            last_vp = traj[idx]['trajectory'][-1]
            if hasattr(self, 'reranker') and ending_queue[idx].size() > 1:
                inputs = []
                inputs_idx = []
                num_candidates = 100
                while num_candidates > 0 and ending_queue[idx].size() > 0:
                    _idx, _pri, item = ending_queue[idx].pop()
                    inputs_idx.append(_idx)
                    inputs.append([len(item.world_states), item.flogit.sum, item.flogit.mean, item.flogp.sum, item.flogp.mean, item.pm, item.speaker] * 4)
                    num_candidates -= 1
                inputs = try_cuda(torch.Tensor(inputs))
                reranker_scores = self.reranker(inputs)
                sel_cand = inputs_idx[torch.argmax(reranker_scores)]
                cur = ending_queue[idx].queue[sel_cand]
            else:
                cur = ending_queue[idx].peak()[-1]
            # keep switching if cur is not the shortest path?

            ob = self.env.observe([cur.world_states[-1]], instr_id=instr_id)
            traj[idx]['trajectory'] += realistic_jumping(
                visit_graphs[idx], last_vp, ob[0])
            ended[idx] = 1

            for _ws in cur.world_states: # we don't collect ws0, this is fine.
                clean_traj[idx]['trajectory'].append((_ws.viewpointId, _ws.heading, _ws.elevation))
                self.clean_results[instr_id] = clean_traj[idx]

        return traj

    def set_beam_size(self, beam_size):
        if self.env.beam_size < beam_size:
            self.env.set_beam_size(beam_size)
        self.beam_size = beam_size

    def set_for_test(self, args=None, use_dropout=False, feedback='argmax', allow_cheat=False,
                     beam_size=1, loss_type=None, debug_interval=-1, explore_log=None):
        if not allow_cheat: # permitted for purpose of calculating validation loss only
            assert feedback in ['argmax', 'sample'] # no cheating by using teacher at test time!
        self.feedback = feedback
        self.loss_type = loss_type if loss_type is not None else self.loss_type
        self.explore_log = explore_log
        self.analysis_entropy = args.analysis_entropy
        self.analysis_examples = args.analysis_examples
        self.analysis_text_gen = args.analysis_text_gen
        self.analysis_text_score = args.analysis_text_score
        self.analysis_text_bleu = args.analysis_text_bleu
        self.bert = (args.bert_follower!="" or args.bert_follower_orig!="")
        self.bert_orig = args.bert_follower_orig!=""
        self.debug_interval = debug_interval
        self.is_train=False
        self.debug = args.debug
        self.fix_bert = False
        self.fix_emb = False
        self.beta = args.beta

        # FAST
        self.fast_greedy_search = args.fast_greedy_search
        self.search_mean = args.search_mean
        self.show_loss = args.show_loss
        self.inject_stop = args.inject_stop

        if use_dropout:
            if self.encoder:
                [x.train() for x in self.encoder if x]
                [x.train() for x in self.decoder if x]
            if self.speak_encoder:
                [x.train() for x in self.speak_encoder if x]
                [x.train() for x in self.speak_decoder if x]
        else:
            #if self.encoder:
            #    self.encoder.eval()
            #    self.decoder.eval()
            #if self.speak_encoder:
            #    self.speak_encoder.eval()
            #    self.speak_decoder.eval()
            if self.encoder:
                [x.eval() for x in self.encoder if x]
                [x.eval() for x in self.decoder if x]
            if self.speak_encoder:
                [x.eval() for x in self.speak_encoder if x]
                [x.eval() for x in self.speak_decoder if x]
        #beam_size = 2 # for less cuda memory # not work
        self.set_beam_size(beam_size)
    def test(self, **kwrgs):
        ''' Evaluate once on each instruction in the current environment '''
        self.set_for_test(**kwrgs)
        logger.debug("test beam size: %d"%self.beam_size)
        if self.beta!=-1:
            logger.info("beta: %f"%self.beta)
        return super(Seq2SeqAgent, self).test()

    def train(self, args, optimizers, n_iters, feedback='teacher', loss_type='follower',
              fix_bert=False, fix_emb=False, delta=0.2):
        ''' Train for a given number of iterations '''
        assert all(f in self.feedback_options for f in feedback.split("+"))
        self.feedback = feedback
        self.loss_type = loss_type
        self.explore_log = None
        self.bert = (args.bert_follower!="" or args.bert_follower_orig!="")
        self.bert_orig = args.bert_follower_orig!=""
        if len(self.encoder)>0 and self.encoder[0]:
            [x.train() for x in self.encoder]
            [x.train() for x in self.decoder]
        if self.speak_encoder:
            [x.train() for x in self.speak_encoder]
            [x.train() for x in self.speak_decoder]
        self.losses = []
        self.is_train=True
        self.debug = args.debug
        self.fix_bert = fix_bert
        self.fix_emb = fix_emb
        self.delta = delta
        it = range(1, n_iters + 1)
        try:
            #import tqdm
            it = tqdm.tqdm(it,dynamic_ncols=True)
        except:
            pass
        for _ in it:
            for optimizer in optimizers:
                if optimizer: optimizer.zero_grad()
            #encoder_optimizer.zero_grad()
            #decoder_optimizer.zero_grad()
            self._rollout_with_loss()
            self.loss.backward()
            for optimizer in optimizers:
                if optimizer: optimizer.step()
            #encoder_optimizer.step()
            #decoder_optimizer.step()
        for optimizer in optimizers:
            if optimizer: optimizer.zero_grad()

    def _encoder_and_decoder_paths(self, base_path):
        return base_path + "_flwEnc", base_path + "_flwDec", base_path + "_spkEnc", base_path + "_spkDec"
    def _encoder_and_decoder_paths_typo(self, base_path):
        return base_path + "_fwlEnc", base_path + "_fwlDec", base_path + "_spkEnc", base_path + "_spkDec"
    def _encoder_and_decoder_paths2(self, base_path):
        return base_path + "_enc", base_path + "_dec"

    def save(self, path):
        ''' Snapshot models  Note that we only save the first model'''
        if "follower" in self.loss_type:
            torch.save(self.encoder[0].state_dict(), path+"_flwEnc")
            torch.save(self.decoder[0].state_dict(), path+"_flwDec")
        if "speaker" in self.loss_type:
            torch.save(self.speak_encoder[0].state_dict(), path+"_spkEnc")
            torch.save(self.speak_decoder[0].state_dict(), path+"_spkDec")

    def load_speaker(self, idx, path, sep=False, **kwargs):
        ''' Loads parameters (but not training state) '''
        if sep:
            encoder_path, decoder_path = self._encoder_and_decoder_paths(path)[2:4]
        else:
            encoder_path, decoder_path = self._encoder_and_decoder_paths2(path)
        self.speak_encoder[idx].load_state_dict(torch.load(encoder_path, **kwargs))
        self.speak_decoder[idx].load_state_dict(torch.load(decoder_path, **kwargs))

    def load_follower(self, idx, path, sep=0, **kwargs):
        ''' Loads parameters (but not training state) '''
        if sep==0:
            encoder_path, decoder_path = self._encoder_and_decoder_paths2(path)
        elif sep==1:
            encoder_path, decoder_path = self._encoder_and_decoder_paths(path)[0:2]
        else:
            encoder_path, decoder_path = self._encoder_and_decoder_paths_typo(path)[0:2]
        self.encoder[idx].load_state_dict(torch.load(encoder_path, **kwargs))
        self.decoder[idx].load_state_dict(torch.load(decoder_path, **kwargs))
