''' Evaluation of agent trajectories '''

import json
from collections import defaultdict
import networkx as nx
import numpy as np
import pprint; pp = pprint.PrettyPrinter(indent=4)  # NoQA

from env import R2RBatch, ImageFeatures
import utils
from utils import load_datasets, load_nav_graphs, Tokenizer
from follower import BaseAgent

import torch
import train
import copy

from collections import namedtuple
from bleu import multi_bleu

EvalResult = namedtuple(
    "EvalResult", "nav_error, oracle_error, trajectory_steps, "
                  "trajectory_length, success, oracle_success, reference_length, reference_steps, success_path_length")

class Evaluation(object):
    ''' Results submission format:
        [{'instr_id': string,
          'trajectory':[(viewpoint_id, heading_rads, elevation_rads),]}] '''

    def __init__(self, splits, instructions_per_path=None, r2r_dataset_path=None,
                 use_reference_path=False, spl_based_on_annotated_length=False):
        self.error_margin = 3.0
        self.splits = splits
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        if instructions_per_path is None:
            instructions_per_path = 3
        self.instructions_per_path = instructions_per_path
        self.use_reference_path = use_reference_path # r4r
        self.spl_based_on_annotated_length = spl_based_on_annotated_length # r4r
        self.reference_paths = {}

        for item in load_datasets(splits, base_path=r2r_dataset_path):
            self.gt[item['path_id']] = item
            self.scans.append(item['scan'])
            self.instr_ids += [
                '%d_%d' % (item['path_id'], i) for i in range(self.instructions_per_path)]
                #'%d_%d' % (item['path_id'], i) for i in range(3)]
            if self.use_reference_path:
                self.reference_paths[item['path_id']] = item['path'][:]
        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        self.graphs = load_nav_graphs(self.scans)
        self.distances = {}
        self.paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _score_item(self, instr_id, path):
        ''' Calculate error based on the final position in trajectory, and also
            the closest position (oracle stopping rule). '''
        gt = self.gt[int(instr_id.split('_')[0])]
        start = gt['path'][0]
        #import ipdb; ipdb.set_trace()
        assert start == path[0][0], \
            'Result trajectories should include the start position'
        goal = gt['path'][-1]
        final_position = path[-1][0]
        nearest_position = self._get_nearest(gt['scan'], goal, path)
        nav_error = self.distances[gt['scan']][final_position][goal]
        oracle_error = self.distances[gt['scan']][nearest_position][goal]
        if self.use_reference_path:
            reference_path = self.reference_paths[int(instr_id.split('_')[0])]
            if self.spl_based_on_annotated_length:
                R=reference_path
                reference_length = sum([self.distances[gt['scan']][R[i]][R[i+1]] for i in range(len(R)-1)]) # For R4R
            else:
                reference_length = self.distances[gt['scan']][start][goal]
        else:
            reference_path = self.paths[gt['scan']][start][goal]
            reference_length = self.distances[gt['scan']][start][goal]
        trajectory_steps = len(path)-1
        reference_steps = len(gt['path'])-1
        trajectory_length = 0  # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            trajectory_length += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr

        success = nav_error < self.error_margin
        shortest_ = reference_length if reference_length>0 else 1e-20 # prevent devision by 0
        success_path_length = int(success)*(float(shortest_)/max(shortest_,trajectory_length))
        # check for type errors
        oracle_success = oracle_error < self.error_margin
        # assert oracle_success == True or oracle_success == False
        return EvalResult(nav_error=nav_error, oracle_error=oracle_error,
                          trajectory_steps=trajectory_steps,
                          trajectory_length=trajectory_length, success=success,
                          oracle_success=oracle_success,
                          reference_length=reference_length, reference_steps=reference_steps,
                          success_path_length=success_path_length,
                          )

    def score_results(self, results, debug_minimal=False, close_look=[], spekaer_bleu=False):
        # results should be a dictionary mapping instr_ids to dictionaries,
        # with each dictionary containing (at least) a 'trajectory' field
        self.scores = defaultdict(list)
        model_scores = []
        instr_ids = set(self.instr_ids)

        instr_count = 0
        all_refs = []
        all_hyps = []
        for instr_id, result in results.items():
            if instr_id in instr_ids:
                instr_count += 1
                instr_ids.remove(instr_id)
                eval_result = self._score_item(instr_id, result['trajectory'])
                if instr_id in close_look:
                    print(instr_id,eval_result)
                # if spekaer_bleu:
                #     tokenized_refs = [
                #         Tokenizer.split_sentence(ref) for ref in gt['instructions']]
                #     tokenized_hyp = result['words']

                    # replaced_gt = gt.copy()
                    # replaced_gt['instructions'] = [' '.join(tokenized_hyp)]
                    # instruction_replaced_gt.append(replaced_gt)

                    # if 'score' in result:
                    #     model_scores.append(result['score'])

                    # if len(tokenized_refs) != self.instructions_per_path:
                    #     skip_count += 1
                    #     skipped_refs.add(base_id)
                    #     continue
                    # all_refs.append(tokenized_refs)
                    # all_hyps.append(tokenized_hyp)

                self.scores['nav_errors'].append(eval_result.nav_error)
                self.scores['oracle_errors'].append(eval_result.oracle_error)
                self.scores['trajectory_steps'].append(
                    eval_result.trajectory_steps)
                self.scores['trajectory_lengths'].append(
                    eval_result.trajectory_length)
                self.scores['reference_length'].append(eval_result.reference_length)
                self.scores['reference_steps'].append(eval_result.reference_steps)
                self.scores['success'].append(eval_result.success)
                self.scores['success_path_length'].append(eval_result.success_path_length)
                self.scores['oracle_success'].append(
                    eval_result.oracle_success)
                if 'score' in result:
                    score=result['score']
                    #print(score)
                    #print(type(score))
                    if isinstance(score,torch.Tensor):
                        score=score.cpu().numpy()
                    model_scores.append(score)

        if not debug_minimal:
            assert len(instr_ids) == 0, \
                'Missing %d of %d instruction ids from %s' % (
                    len(instr_ids), len(self.instr_ids), ",".join(self.splits))

            assert len(self.scores['nav_errors']) == len(self.instr_ids), (len(self.scores['nav_errors']),len(self.instr_ids))
        score_summary = {
            'nav_error': np.average(self.scores['nav_errors']),
            'oracle_error': np.average(self.scores['oracle_errors']),
            'steps': np.average(self.scores['trajectory_steps']),
            'lengths': np.average(self.scores['trajectory_lengths']),
            'gold_length': np.average(self.scores['reference_length']),
            'success_rate': float(
                sum(self.scores['success']) / len(self.scores['success'])),
            'spl': float(
                sum(self.scores['success_path_length']) / len(self.scores['success_path_length'])),
            'oracle_rate': float(sum(self.scores['oracle_success'])
                                 / len(self.scores['oracle_success'])),
        }
        if len(model_scores) > 0:
            if not debug_minimal:
                assert len(model_scores) == instr_count
            score_summary['model_score'] = np.average(model_scores)

        num_successes = len(
            [i for i in self.scores['nav_errors'] if i < self.error_margin])
        # score_summary['success_rate'] = float(num_successes)/float(len(self.scores['nav_errors']))  # NoQA
        assert float(num_successes) / float(len(self.scores['nav_errors'])) == score_summary['success_rate']  # NoQA
        oracle_successes = len(
            [i for i in self.scores['oracle_errors'] if i < self.error_margin])
        assert float(oracle_successes) / float(len(self.scores['oracle_errors'])) == score_summary['oracle_rate']  # NoQA
        # score_summary['oracle_rate'] = float(oracle_successes) / float(len(self.scores['oracle_errors']))  # NoQA
        return score_summary, self.scores

    def score_file(self, output_file):
        ''' Evaluate each agent trajectory based on how close it got to the
        goal location '''
        with open(output_file) as f:
            return self.score_results(json.load(f))

    def score_gen_instrcutions(self, results, debug_minimal=False, verbose=False):
        # results should be a dictionary mapping instr_ids to dictionaries,
        # with each dictionary containing (at least) a 'words' field
        instr_ids = set(self.instr_ids)
        instr_count = 0
        results_by_base_id = {}
        mismatches = []
        for instr_id, result in results.items():
            if instr_id in instr_ids:
                instr_ids.remove(instr_id)

                base_id = int(instr_id.split('_')[0])

                if base_id in results_by_base_id:
                    #old_predicted = results_by_base_id[base_id]['words']
                    #new_predicted = result['words']
                    #if old_predicted != new_predicted:
                    #    mismatches.append((old_predicted, new_predicted))
                    continue
                else:
                    results_by_base_id[base_id] = result

        if mismatches:
            #print("mismatching outputs for sentences:")
            #for old_pred, new_pred in mismatches:
            #    print(old_pred)
            #    print(new_pred)
            #    print()
            print("mismatching outputs for %d sentences."%len(mismatches))

        if not debug_minimal:
            assert len(instr_ids) == 0, \
                'Missing %d of %d instruction ids from %s' % (
                len(instr_ids), len(self.instr_ids), ",".join(self.splits))

        all_refs = [[] for t in range(20)]
        all_hyps = [[] for t in range(20)]

        model_scores = []

        instruction_replaced_gt = []

        skip_count = 0
        skipped_refs = set()
        for base_id, result in sorted(results_by_base_id.items()):
            instr_count += 1
            gt = self.gt[base_id]
            tokenized_refs = [
                Tokenizer.split_sentence(ref) for ref in gt['instructions']]
            for t,tokenized_hyp in enumerate(result["gen_texts_gold"]):
                #for tokenized_hyp in gen_text:
                #tokenized_hyp = result['words']
                #print("tokenized_hyp",tokenized_hyp)

                replaced_gt = gt.copy()
                replaced_gt['instructions'] = [' '.join(tokenized_hyp)]
                instruction_replaced_gt.append(replaced_gt)

                if 'score' in result:
                    model_scores.append(result['score'])

                if len(tokenized_refs) != self.instructions_per_path:
                    skip_count += 1
                    skipped_refs.add(base_id)
                    continue
                all_refs[t].append(tokenized_refs)
                all_hyps[t].append(tokenized_hyp)

                if verbose and instr_count % 100 == 0:
                    for i, ref in enumerate(tokenized_refs):
                        print("ref {}:\t{}".format(i, ' '.join(ref)))
                    print("pred  :\t{}".format(' '.join(tokenized_hyp)))
                    print()

        if skip_count != 0:
            print("skipped {} instructions without {} refs: {}".format(
                skip_count, self.instructions_per_path, ' '.join(
                    str(i) for i in skipped_refs)))

        model_score = np.mean(model_scores)
        score_summaries = []
        score_summary = {
            'model_score': model_score,
        }
        for t in range(20):
            assert len(all_refs[t])==len(all_hyps[t]), (len(all_refs[t]),len(all_hyps[t]))
            if len(all_refs[t])==0: continue
            bleu, unpenalized_bleu = multi_bleu(all_refs[t], all_hyps[t])
            score_summary["bleu"+str(t)] = bleu
            score_summary['unpenalized_bleu'+str(t)] = unpenalized_bleu
            print("t,bleu, unpenalized_bleu: %2d, %6f, %6f, %3d"%(t,bleu, unpenalized_bleu, len(all_hyps[t])))
        return score_summary, instruction_replaced_gt

def eval_simple_agents(args):
    ''' Run simple baselines on each split. '''
    img_features = ImageFeatures.from_args(args)
    for split in ['train', 'val_seen', 'val_unseen', 'test']:
        env = R2RBatch(img_features, batch_size=1, splits=[split])
        ev = Evaluation([split])

        for agent_type in ['Stop', 'Shortest', 'Random']:
            outfile = '%s%s_%s_agent.json' % (
                train.RESULT_DIR, split, agent_type.lower())
            agent = BaseAgent.get_agent(agent_type)(env, outfile)
            agent.test()
            agent.write_results()
            score_summary, _ = ev.score_file(outfile)
            print('\n%s' % agent_type)
            pp.pprint(score_summary)


def eval_seq2seq():
    ''' Eval sequence to sequence models on val splits (iteration selected from
    training error) '''
    outfiles = [
        train.RESULT_DIR + 'seq2seq_teacher_imagenet_%s_iter_5000.json',
        train.RESULT_DIR + 'seq2seq_sample_imagenet_%s_iter_20000.json'
    ]
    for outfile in outfiles:
        for split in ['val_seen', 'val_unseen']:
            ev = Evaluation([split])
            score_summary, _ = ev.score_file(outfile % split)
            print('\n%s' % outfile)
            pp.pprint(score_summary)


if __name__ == '__main__':
    from train import make_arg_parser
    utils.run(make_arg_parser(), eval_simple_agents)
    # eval_seq2seq()
