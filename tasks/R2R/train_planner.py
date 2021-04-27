import torch
from torch import optim

import os
import os.path
import time
import json
import numpy as np
from collections import defaultdict
import argparse
from pathlib import Path # mkdir
from collections import namedtuple

import glob
import re

import utils
from utils import read_vocab, Tokenizer, vocab_pad_idx, timeSince, try_cuda, rec_numpy2list, get_model_prefix
from env import R2RBatch, ImageFeatures
from model import EncoderLSTM, AttnDecoderLSTM
from model import SpeakerEncoderLSTM, SpeakerDecoderLSTM
#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from planner import Seq2SeqAgent
from speaker import Seq2SeqSpeaker
import eval

from vocab import SUBTRAIN_VOCAB, TRAINVAL_VOCAB, TRAIN_VOCAB

import logging
logger = logging.getLogger(__name__)

RESULT_DIR = 'tasks/R2R/results/'
SNAPSHOT_DIR = 'tasks/R2R/snapshots/'
OPTIMIZER_DIR = 'tasks/R2R/snapshots/'
PLOT_DIR = 'tasks/R2R/plots/'
JSON_DIR = 'tasks/R2R/json/'

# TODO: how much is this truncating instructions?
# MAX_INPUT_LENGTH = 80
MAX_INPUT_LENGTH = 100

BATCH_SIZE = 100
#max_episode_len = 10
word_embedding_size = 300
glove_path = 'tasks/R2R/data/train_glove.npy'
action_embedding_size = 2048+128
hidden_size = 512
dropout_ratio = 0.5
# feedback_method = 'sample' # teacher or sample
learning_rate = 0.0001
weight_decay = 0.0005
FEATURE_SIZE = 2048+128
#log_every = 100

Seq2SeqOptimizers = namedtuple("Seq2SeqOptimizers", "spkEnc, spkDec, flwEnc, flwDec")

def filter_param(param_list):
    return [p for p in param_list if p.requires_grad]

def make_path(args, train_env, split_string, n_iter):
    return os.path.join(
        SNAPSHOT_DIR, '%s_%s_iter_%d' % (
            get_model_prefix(args, train_env.image_features_list),
            split_string, n_iter))
def make_path_opt(args, train_env, split_string, n_iter):
    return os.path.join(
        OPTIMIZER_DIR, '%s_%s_iter_%d' % (
            get_model_prefix(args, train_env.image_features_list),
            split_string, n_iter))

def save_optimizer(optimizers, path, suffixs):
    for optimizer,suffix in zip(optimizers,suffixs):
        if optimizer:
            torch.save(optimizer.state_dict(), path+suffix)
def load_optimizer(optimizers, path, suffixs, **kwargs):
    for optimizer,suffix in zip(optimizers,suffixs):
        if optimizer:
            optimizer.load_state_dict(torch.load(path+suffix, **kwargs))
def load_speaker(agent,idx,prefix,load_args):
    try:
        agent.load_speaker(idx, prefix, **load_args)
    except FileNotFoundError:
        agent.load_speaker(idx, prefix, sep=True, **load_args)

def do_test(args, train_env, agent, optimizers, val_envs,
         data_log, loss_str, best_metrics, split_string, last_model_saved, n_iter,
         show_metrics=None):
    if show_metrics is None:
        show_metrics = ["success_rate"]
    assert len(show_metrics)>0
    save_log = []
    for env_name, (val_env, evaluator) in sorted(val_envs.items()):
        agent.env = val_env
        logger.info("Evaluating on {}".format(env_name))

        # Get validation loss under the same conditions as training
        if not args.wo_validation_loss:
            logger.info("Get validation loss under the same conditions as training")
            agent.test(args=args, use_dropout=True, feedback=args.feedback_method,
                       loss_type=args.loss_type,
                       debug_interval=args.debug_interval,
                       allow_cheat=True)
            val_losses = np.array(agent.losses)
            val_loss_avg = np.average(val_losses)
        else:
            val_loss_avg = 0.
        data_log['%s loss' % env_name].append(val_loss_avg)

        # Get validation distance from goal under evaluation conditions
        logger.info("Get validation distance from goal under evaluation conditions")
        agent.results_path = '%s%s_%s_iter_%d.json' % (
            RESULT_DIR, get_model_prefix(
                args, train_env.image_features_list),
            env_name, n_iter)
        if args.analysis_accuracy:  # Use gold transitions for analysis
            explore_log_path = '%s%s_%s_teacherLogPath_iter_%d.json' % (
                RESULT_DIR, get_model_prefix(
                    args, train_env.image_features_list),
                env_name, n_iter)
            agent.test(args=args, use_dropout=False, feedback='teacher' if args.analysis_gold else 'argmax',
                       allow_cheat=True,
                       loss_type=args.loss_type,
                       debug_interval=-1,
                       explore_log=[] if args.analysis_accuracy else None
                       )
            logger.info("Save explore_log_path="+explore_log_path)
            json.dump(rec_numpy2list(agent.explore_log),
                      open(explore_log_path,"w"))
            #if args.analysis_accuracy:
            #    import ipdb; ipdb.set_trace()
        elif args.analysis_gold:
            agent.test(args=args, use_dropout=False, feedback='teacher',
                       allow_cheat=True,
                       loss_type=args.loss_type,
                       debug_interval=-1,
                       )
        elif args.r4r_eval>=0:
            agent.results = {}
            for i in range(args.r4r_eval):
                load_path = args.r4r_result_prefix + "_" + args.r4r_small + "%02d.json"%i
                logger.info("Loading %d-th results from "%i+load_path)
                rlt = json.load(open(load_path, "r"))
                agent.results.update(rlt)
        else:
            agent.test(args=args, use_dropout=False, feedback='argmax',
                       loss_type=args.loss_type,
                       debug_interval=-1,
                       )

        if not args.no_save:
            agent.write_results()
        debug_minimal = True if args.debug_minimal or args.analysis_text_gen!=[] else False
        if args.r4r_eval_split>=0:
            save_path = args.r4r_result_prefix + "_" + args.r4r_small + "%02d.json"%args.r4r_eval_split
            logger.info("Saving results to "+save_path)
            def conv_results(results):
                for k,v in results.items():
                    v["observations"] = []
                    v['instr_encoding']=[vv.tolist() for vv in v['instr_encoding']]
                    v['score']=float(v['score']) if "score" in v else 0
                    v['scores']=[float(vv) for vv in v['scores']] if "scores" in v else [0]
                return results
            json.dump(conv_results(agent.results), open(save_path, "w"), indent = 4)
            quit()
        if args.analysis_text_bleu:
            score_summary, _ = evaluator.score_gen_instrcutions(agent.results, debug_minimal, verbose=args.analysis_text_bleu_verbose)
            logger.info(str(score_summary))
        else:
            score_summary, raw_scores = evaluator.score_results(agent.results, debug_minimal, close_look=args.analysis_examples)

            if args.save_scores!="":
                json.dump(dict(raw_scores), open(args.save_scores,"w"))

        loss_str += ', %s loss: %.4f' % (env_name, val_loss_avg)
        for metric, val in sorted(score_summary.items()):
            data_log['%s %s' % (env_name, metric)].append(val)
            if metric == show_metrics[0]: # main metric
                loss_str += ', %s: %.3f' % (metric, val)

                key = (env_name, metric)
                if key not in best_metrics or best_metrics[key] < val:
                    best_metrics[key] = val
                    if args.no_save: continue
                    if env_name=="val_seen" and val<0.50: continue
                    if env_name=="val_unseen" and val<0.40: continue
                    # Saving
                    model_path = make_path(args, train_env, split_string, n_iter) + "_%s-%s=%.3f" % (
                        env_name, metric, val)
                    save_log.append(
                        "new best %s, saved model to %s" %(env_name, model_path))
                    logger.info("Saving models to "+str(model_path))
                    agent.save(model_path)
                    path = make_path_opt(args, train_env, split_string, n_iter)
                    logger.info("Saving optimizers to "+str(path))
                    save_optimizer(optimizers, path, [suffix+"Op" for suffix in optimizers._fields])
                    if args.remove_last_saved:
                        if key in last_model_saved:
                            for old_model_path in \
                                    agent._encoder_and_decoder_paths(
                                        last_model_saved[key]):
                                try:
                                    os.remove(old_model_path)
                                except:
                                    logger.warning("Cannot remove "+old_model_path)
                    last_model_saved[key] = model_path
            if len(show_metrics)>1:
                if metric in show_metrics[1:]:
                    loss_str += ', %s: %.3f' % (metric, val)

        #logger.info("Fin Evaluating on {}".format(env_name))
        #logger.info("")
    return save_log, data_log, loss_str, best_metrics, split_string, last_model_saved

def do_train(args, train_env, agent, optimizers,
          n_iters, log_every, val_envs=None, begin_iters=0):
    ''' Train on training set, validating on both seen and unseen. '''

    if val_envs is None:
        val_envs = {}

    Path(JSON_DIR).mkdir(parents=True, exist_ok=True)
    Path(OPTIMIZER_DIR).mkdir(parents=True, exist_ok=True)

    logger.info('Training with %s feedback' % args.feedback_method)

    data_log = defaultdict(list)
    start = time.time()

    split_string = "-".join(train_env.splits)

    best_metrics = {}
    last_model_saved = {}
    begin = int(args.seq_eval[0])*log_every if args.seq_eval else begin_iters

    #begin = int(args.seq_eval[0]) if args.seq_eval else 0
    print(begin, n_iters)
    feedback_method = args.feedback_method
    if args.n_iters_teacher2sample>0:
        feedback_method = "teacher"
    fix_bert = args.fix_bert>0
    fix_emb = args.fix_emb>0
    logger.info('fix_bert : %s until iter %d' % (str(fix_bert),args.fix_bert))
    logger.info('fix_emb : %s until iter %d' % (str(fix_bert),args.fix_emb))
    for idx in range(begin, n_iters, log_every):

        load_name=""
        if args.seq_eval:
            if args.seq_eval_flw:
                load_name = args.follower_prefix+str(idx)
            else:
                load_name = args.speaker_prefix+str(idx)
            logger.info("Loading "+load_name)
            from time import sleep
            from scipy.linalg import lu
            while True:
                try:
                    if args.seq_eval_flw:
                        load_args = {}
                        if args.no_cuda:
                            load_args['map_location'] = 'cpu'
                        try:
                            print(0)
                            agent.load_follower(0,load_name, **load_args)
                        except FileNotFoundError:
                            try:
                                print(1)
                                agent.load_follower(0,load_name, sep=1, **load_args)
                            except FileNotFoundError:
                                print(2)
                                agent.load_follower(0,load_name, sep=2, **load_args)
                    else:
                        load_speaker(agent, 0, load_name,load_args={})
                    break
                except:
                    #logger.info("Not found:"+load_name)
                    #try:
                    for _ in range(10):
                        X=np.random.randn(10000,10000)
                        lu(X)
                    #except:
                    #    pass
                    sleep(10)

        #if args.n_iters_teacher2sample>0 and args.n_iters_teacher2sample<n_iters:
        if args.n_iters_teacher2sample>0 and args.n_iters_teacher2sample<idx:
            if feedback_method=="teacher":
                feedback_method = "sample"
                logger.info('Change feedback_method : Training with %s feedback at iter %d' % (feedback_method,idx))

        if args.fix_bert>0 and args.fix_bert<idx and (not fix_bert):
            logger.info('Change fix_bert : False at iter %d' % (idx))
            fix_bert = False
        if args.fix_emb>0 and args.fix_emb<idx and (not fix_emb):
            logger.info('Change fix_emb : False at iter %d' % (idx))
            fix_emb = False

        agent.env = train_env

        interval = min(log_every, n_iters-idx)
        if args.debug_interval > 0:
            interval = args.debug_interval
        n_iter = idx + interval
        data_log['iteration'].append(n_iter)

        delta = args.delta*(float(idx)/args.delta_linear) \
            if args.delta_linear>0 and idx<args.delta_linear \
            else args.delta

        # Train for log_every interval
        logging.info("Train on Epoch %d, n_iters=%d, log_every=%d, interval=%d, delta=%f"%(idx,n_iters,log_every,interval,delta))
        if not args.wo_train:
            agent.train(args, optimizers, interval,
                        feedback=feedback_method,
                        loss_type=args.loss_type,
                        fix_bert = fix_bert,
                        fix_emb = fix_emb,
                        delta = delta,
                        )
            train_losses = np.array(agent.losses)
            assert len(train_losses) == interval
            train_loss_avg = np.average(train_losses)
        else:
            train_loss_avg = 0.0
        data_log['train loss'].append(train_loss_avg)
        loss_str = 'train loss: %.4f' % train_loss_avg

        # Run validation
        #if (not ( args.wo_eval or args.wo_eval_until>n_iter ) ) or (args.save_every and n_iter % args.save_every == 0):
        save_log = []
        if not ( args.wo_eval or args.wo_eval_until>n_iter ):
            if args.r4r and args.wo_eval_until>n_iter:
                save_log = []
            #elif (not ( args.wo_eval or args.wo_eval_until>n_iter ) ) or (args.save_every and n_iter % args.save_every == 0):
            elif (not ( args.wo_eval or args.wo_eval_until>n_iter ) ) or (args.save_every and n_iter % args.save_every == 0):
                save_log, data_log, loss_str, best_metrics, split_string, last_model_saved = \
                    do_test(args, train_env, agent, optimizers, val_envs, data_log, loss_str, best_metrics, split_string, last_model_saved, n_iter, args.show_metrics)
            else:
                save_log = []

        if args.n_iters==1: # This is the final case. It presents all.
            jsonl = json.dumps({k:float(v[-1]) for k,v in data_log.items()})
            logger.info(jsonl)

        logger.info('%s (%d %d%%) %s %s' % (
            timeSince(start, float(n_iter)/n_iters),
            n_iter, float(n_iter)/n_iters*100, load_name, loss_str))
        for s in save_log:
            logger.info(s)

        if not args.no_save:
            if args.always_save or (args.save_every and n_iter % args.save_every == 0):
                path = make_path(args, train_env, split_string, n_iter)
                logger.info("Saving models to "+str(path))
                agent.save(path)
                logger.info("Saving optimizers to "+str(path))
                path = make_path_opt(args, train_env, split_string, n_iter)
                save_optimizer(optimizers, path, [suffix+"Op" for suffix in optimizers._fields])

        # pandas doen't work with data skipping by --wo_eval_until
        #df = pd.DataFrame(data_log)
        #df.set_index('iteration')
        #df_path = '%s%s_%s_log.csv' % (
        #    PLOT_DIR, get_model_prefix(
        #        args, train_env.image_features_list), split_string)
        #df.to_csv(df_path)

        jsonl = json.dumps({k:float(v[-1]) for k,v in data_log.items()})
        df_path = '%s%s_%s_log.jsonl' % (
            JSON_DIR, get_model_prefix(
                args, train_env.image_features_list), split_string)
        open(df_path,"a").write(jsonl+"\n")

        if args.seq_eval:
            if len(args.seq_eval)>1 and n_iter>int(args.seq_eval[1]):
                break


def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)


def make_more_train_env(args, train_vocab_path, train_splits,
                        batch_size=BATCH_SIZE):
    setup()
    image_features_list = ImageFeatures.from_args(args)
    vocab = read_vocab(train_vocab_path)
    tok = Tokenizer(vocab=vocab)
    if args.bert_follower:
        from huggingface import BertTokenizer
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert_follower)
    elif args.bert_follower_orig:
        from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert_follower_orig)
    else:
        bert_tokenizer = None
    train_env = R2RBatch(args, image_features_list, batch_size=batch_size,
                         splits=train_splits, tokenizer=tok, bert_tokenizer=bert_tokenizer,
                         debug_minimal=args.debug_minimal, r2r_dataset_path=args.r2r_dataset_path)
    return train_env


def make_env(args, train_vocab_path, train_splits, test_splits, batch_size=BATCH_SIZE):
    setup()
    logger.info("image")
    image_features_list = ImageFeatures.from_args(args)
    logger.info("vocab")
    vocab = read_vocab(train_vocab_path)
    tok = Tokenizer(vocab=vocab)
    logger.info("vocab size: %d"%len(vocab))
    if args.bert_follower:
        from huggingface import BertTokenizer
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert_follower)
    elif args.bert_follower_orig:
        from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert_follower_orig)
    else:
        bert_tokenizer = None
    train_env = R2RBatch(args, image_features_list, batch_size=batch_size,
                         splits=train_splits, tokenizer=tok, bert_tokenizer=bert_tokenizer,
                         debug_minimal=args.debug_minimal, r2r_dataset_path=args.r2r_dataset_path)
    test_envs = {
        split: (R2RBatch(args, image_features_list, batch_size=batch_size,
                         splits=[split], tokenizer=tok, bert_tokenizer=bert_tokenizer,
                         debug_minimal=args.debug_minimal, r2r_dataset_path=args.r2r_dataset_path),
                eval.Evaluation([split], args.instructions_per_path, r2r_dataset_path=args.r2r_dataset_path,
                                use_reference_path=(args.r4r or args.r4r_reward1 or args.r4r_reward2),
                                spl_based_on_annotated_length=args.spl_based_on_annotated_length))
        for split in test_splits}
    return train_env, test_envs, vocab

def make_follow_model(args,glove,vocab_size):
    enc_hidden_size = hidden_size//2 if args.bidirectional else hidden_size
    feature_size = FEATURE_SIZE
    if "follower" in args.loss_type:
        if args.bert_follower:
            from model_bert import EncoderBERT
            encoder = try_cuda(EncoderBERT(args.bert_follower))
        elif args.bert_follower_orig:
            from model_bert_orig import EncoderBERT
            encoder = try_cuda(EncoderBERT(args.bert_follower_orig))
        else:
            encoder = try_cuda(EncoderLSTM(
                vocab_size, word_embedding_size, enc_hidden_size, vocab_pad_idx,
                dropout_ratio, bidirectional=args.bidirectional, glove=glove))
        decoder = try_cuda(AttnDecoderLSTM(
            action_embedding_size, hidden_size, dropout_ratio,
            feature_size=feature_size))
    else:
        encoder = None
        decoder = None
    return encoder,decoder

def make_speak_model(args,glove,vocab_size):
    enc_hidden_size = hidden_size//2 if args.bidirectional else hidden_size
    feature_size = FEATURE_SIZE
    if args.speaker_type=="lstm":
        speak_encoder = try_cuda(SpeakerEncoderLSTM(
            action_embedding_size, feature_size, enc_hidden_size, dropout_ratio,
            bidirectional=args.bidirectional))
        speak_decoder = try_cuda(SpeakerDecoderLSTM(
            vocab_size, word_embedding_size, hidden_size, dropout_ratio,
            glove=glove))
    else:
        from model_transformer import SpeakerEncoderTransformer, SpeakerDecoderTransformer
        ntokens = vocab_size         # the size of vocabulary
        ninp = action_embedding_size+feature_size # embedding dimension
        nhid = action_embedding_size+feature_size # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 3   # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 4     # the number of heads in the multiheadattention models
        dropout = 0.5 # the dropout value
        # def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        speak_encoder = try_cuda(SpeakerEncoderTransformer(
            ntokens, ninp, nhead, nhid, nlayers, dropout,
            ))
        # def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        speak_decoder = try_cuda(SpeakerDecoderTransformer(
            ntokens, ninp, nhead, nhid, nlayers, dropout,
            ))
    return speak_encoder,speak_decoder

#def train_setup(args, batch_size=BATCH_SIZE):
def train_setup(args):
    train_splits = ['train']
    # val_splits = ['train_subset', 'val_seen', 'val_unseen']
    val_splits = ['val_seen', 'val_unseen']
    if args.r4r_eval>0:
        train_splits = []
        val_splits = ['val_unseen']
        args.show_metrics = ["success_rate", "spl", "lengths", "nav_error"]
        args.image_feature_type=["none"]
        args.r4r_eval_split = -1
        args.n_iters = 1
        args.wo_train = True
        args.wo_eval = False
    elif args.r4r_eval_split>=0:
        args.n_iters = 1
        args.wo_train = True
        args.wo_eval = False
        train_splits = []
        val_splits = [args.r4r_small+'%02d'%args.r4r_eval_split]
        #val_splits = ['val_unseen_split%02d'%args.r4r_eval_split]
    elif args.r4r_small:
        if args.r4r_small=="VAL_SEEN":
            val_splits = ['val_seen']
        else:
            val_splits = ['val_seen', args.r4r_small]
    elif args.use_test_set:
        val_splits.append('test')

    vocab_path = TRAIN_VOCAB
    batch_size = args.minibatch_size

    if args.use_train_subset:
        train_splits = ['sub_' + split for split in train_splits]
        val_splits = ['sub_' + split for split in val_splits]
        vocab_path = SUBTRAIN_VOCAB

    # make_env_and_models
    train_env, val_envs, vocab = make_env(args, vocab_path, train_splits, val_splits, batch_size)
    glove = np.load(glove_path)
    encoders,decoders=[],[]
    speak_encoders,speak_decoders,speakers=[],[],[]
    for _ in range(args.follower_num):
        enc, dec = make_follow_model(args, glove, len(vocab))
        encoders.append(enc)
        decoders.append(dec)
    for _ in range(args.speaker_num):
        speak_enc, speak_dec =  make_speak_model(args, glove, len(vocab))
        speaker = Seq2SeqSpeaker(
            args, train_env, "", speak_enc, speak_dec, args.max_input_length)
        speak_encoders.append(speak_enc)
        speak_decoders.append(speak_dec)
        speakers.append(speaker)
    if args.overall_speaker_prefix:
        oa_speak_encoder ,oa_speak_decoder = make_speak_model(args, glove, len(vocab))
        overall_speaker = Seq2SeqSpeaker(
            args, train_env, "", oa_speak_encoder, oa_speak_decoder, args.max_input_length)
    else:
        overall_speaker = None
    if args.use_pretraining:
        pretrain_splits = args.pretrain_splits
        assert len(pretrain_splits) > 0, \
            'must specify at least one pretrain split'
        pretrain_env = make_more_train_env(
            args, vocab_path, pretrain_splits, batch_size=batch_size)

    # Note that the agent is always unique while speakers or followers can be multiple.
    agent = Seq2SeqAgent(
        train_env, "", encoders, decoders, speakers, speak_encoders, speak_decoders, args.max_episode_len,
        max_instruction_length=args.max_input_length)

    load_args = {}
    if args.no_cuda:
        load_args['map_location'] = 'cpu'
    if args.follower_prefix!="" and not args.seq_eval_flw:
        try:
            print(0)
            agent.load_follower(0,args.follower_prefix, **load_args)
        except FileNotFoundError:
            try:
                print(1)
                agent.load_follower(0,args.follower_prefix, sep=1, **load_args)
            except FileNotFoundError:
                print(2)
                agent.load_follower(0,args.follower_prefix, sep=2, **load_args)
        #import ipdb; ipdb.set_trace()
    if args.speaker_prefix and (not (args.seq_eval or args.train_from_latest)):
        logger.info("Loading Speaker :"+args.speaker_prefix)
        load_speaker(agent,0,args.speaker_prefix,load_args)
    if args.overall_speaker_prefix:
        logger.info("Loading Overall Speaker :"+args.overall_speaker_prefix)
        raise NotImplemented
        load_speaker(overall_speaker,0,args.overall_speaker_prefix,load_args)
        agent.overall_speaker = overall_speaker

    if args.use_pretraining:
        return agent, train_env, val_envs, pretrain_env
    else:
        return agent, train_env, val_envs

def train_val(args):

    if args.seq_eval: # or args.wo_train:
        args.no_save = True
        args.wo_train = True
        args.wo_eval = False
        args.wo_eval_until = 0
        args.always_save = False
        args.save_every = 1e20
    #if args.always_save:
    #    args.save_every = 1e20

    ''' Train on the training set, and validate on seen and unseen splits. '''
    if args.use_pretraining:
        agent, train_env, val_envs, pretrain_env = train_setup(args)
    else:
        agent, train_env, val_envs = train_setup(args)

    if "follower" in args.loss_type:
        encoder_optimizer = optim.Adam(
            filter_param(agent.encoder[0].parameters()), lr=learning_rate,
            weight_decay=weight_decay)
        decoder_optimizer = optim.Adam(
            filter_param(agent.decoder[0].parameters()), lr=learning_rate,
            weight_decay=weight_decay)
    else:
        encoder_optimizer = None
        decoder_optimizer = None
    if "speaker" in args.loss_type:
        speaker_encoder_optimizer = optim.Adam(
            filter_param(agent.speak_encoder[0].parameters()), lr=learning_rate,
            weight_decay=weight_decay)
        speaker_decoder_optimizer = optim.Adam(
            filter_param(agent.speak_decoder[0].parameters()), lr=learning_rate,
            weight_decay=weight_decay)
    else:
        speaker_encoder_optimizer = None
        speaker_decoder_optimizer = None

    optimizers = Seq2SeqOptimizers(\
            spkEnc=speaker_encoder_optimizer,
            spkDec=speaker_decoder_optimizer,
            flwEnc=encoder_optimizer,
            flwDec=decoder_optimizer,
                             )

    if args.follower_op_prefix!="":
        raise NotImplemented
    if args.speaker_op_prefix!="":
        logger.info("Loading Speaker Optimizer"+args.speaker_op_prefix)
        load_optimizer(optimizers, args.speaker_op_prefix, [suffix+"Op" for suffix in optimizers._fields])


    dir1="tasks/R2R/snapshots/"
    #dir2="tasks/R2R/optimizers/"
    dir2="tasks/R2R/snapshots/"
    base = args.speaker_prefix
    if args.train_from_latest:
        files = glob.glob(base+"*")
        print(files)
        # Get number after base in ss
        def get_last_num(base,ss):
            ss=ss.replace(base,"")
            m=re.search(r'[0-9]+',ss)
            numstr=m.group()
            #print(numstr)
            return int(numstr)
        def get_largest_num(base,files):
            return max([get_last_num(base,f) for f in files])
        begin = get_largest_num(base,files)
        load_name = base+str(begin)
        logger.info("Loading "+load_name)
        load_speaker(agent, 0, load_name,load_args={})
        load_optimizer(optimizers, load_name.replace(dir1,dir2), [suffix+"Op" for suffix in optimizers._fields])
    elif args.train_from>0:
        begin = args.train_from
        #load_name = base+str(begin)
        #logger.info("Loading "+load_name)
        #load_speaker(agent, load_name,load_args={})
        #load_optimizer(optimizers, load_name.replace(dir1,dir2), [suffix+"Op" for suffix in optimizers._fields])
    else:
        begin=0

    if args.use_pretraining:
        if begin<args.n_pretrain_iters:
            do_train(args, pretrain_env, agent, optimizers,
                  args.n_pretrain_iters, args.log_every_pretrain, val_envs=val_envs, begin_iters=begin)
            begin = args.n_pretrain_iters

    do_train(args, train_env, agent, optimizers,
          args.n_iters, args.log_every, val_envs=val_envs,
          begin_iters=begin)

# Test set prediction will be handled separately
# def test_submission(args):
#     ''' Train on combined training and validation sets, and generate test
#     submission. '''
#     agent, train_env, test_envs = test_setup(args)
#     train(args, train_env, agent)
#
#     test_env = test_envs['test']
#     agent.env = test_env
#
#     agent.results_path = '%s%s_%s_iter_%d.json' % (
#         RESULT_DIR, get_model_prefix(args, train_env.image_features_list),
#         'test', args.n_iters)
#     agent.test(use_dropout=False, feedback='argmax')
#     if not args.no_save:
#         agent.write_results()


def make_arg_parser():
    parser = argparse.ArgumentParser()
    ImageFeatures.add_args(parser)
    parser.add_argument(
        "--feedback_method", choices=["sample", "teacher", "teacher+sample", "teacher2sample"],
        default="sample")
    parser.add_argument(
        "--loss_type", choices=["follower", "speaker", "follower+speaker"],
        default="follower")
    parser.add_argument(
        "--speaker_type", choices=["lstm", "transformer"],
        default="lstm")
    parser.add_argument(
        "--bert_follower", choices=["bert-base-uncased", "bert-large-uncased", ""],
        default="")
    parser.add_argument(
        "--bert_follower_orig", choices=["bert-base-uncased", "bert-large-uncased", ""],
        default="")
    parser.add_argument(
        "--r2r_dataset_path", choices=["tasks/R2R/data/R2R", "tasks/R2R/data/R4R", ""],
        default="tasks/R2R/data/R2R")

    parser.add_argument("--bidirectional", action='store_true')
    parser.add_argument("--use_pretraining", action='store_true')
    parser.add_argument("--pretrain_splits", nargs="+", default=[])
    parser.add_argument("--n_pretrain_iters", type=int, default=50000)
    parser.add_argument("--no_save", action='store_true')

    parser.add_argument("--n_iters", type=int, default=100000)
    parser.add_argument("--remove_last_saved", action='store_true')
    parser.add_argument("--train_from_latest", action='store_true')
    parser.add_argument("--train_from", type=int, default=-1)

    parser.add_argument("--speaker_num", type=int, default=1)
    parser.add_argument("--follower_num", type=int, default=1)
    parser.add_argument("--follower_prefix", type=str, default='')
    parser.add_argument("--speaker_prefix", type=str, default='')
    parser.add_argument("--follower_model_name", type=str, default='')
    parser.add_argument("--speaker_model_name", type=str, default='')
    parser.add_argument("--follower_op_prefix", type=str, default='')
    parser.add_argument("--speaker_op_prefix", type=str, default='')
    parser.add_argument("--model_name", type=str, default='planner')

    parser.add_argument("--debug_interval", type=int, default=-1)
    parser.add_argument("--wo_validation_loss", action='store_true')
    parser.add_argument("--wo_train", action='store_true')
    parser.add_argument("--wo_eval", action='store_true')
    parser.add_argument("--wo_eval_until", type=int, default=0)
    parser.add_argument("--always_save", action='store_true')
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--log_every_pretrain", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--delta", type=float, default=0.333333333)
    parser.add_argument("--delta_linear", type=int, default=-1)
    parser.add_argument("--beta", type=float, default=-1)
    parser.add_argument("--analysis_gold", action='store_true')
    parser.add_argument("--analysis_accuracy", action='store_true')
    parser.add_argument("--analysis_entropy", action='store_true')
    parser.add_argument("--analysis_text_bleu", action='store_true')
    parser.add_argument("--analysis_text_bleu_verbose", action='store_true')
    parser.add_argument('--analysis_examples', nargs='+', default=[])
    parser.add_argument("--analysis_text_gen", action='store_true')
    parser.add_argument("--analysis_text_score", action='store_true')
    parser.add_argument("--max_input_length", type=int, default=100)
    parser.add_argument("--max_episode_len", type=int, default=20)
    parser.add_argument("--minibatch_size", type=int, default=100)
    parser.add_argument("--fix_bert", type=int, default=-1)
    parser.add_argument("--fix_emb", type=int, default=-1)
    parser.add_argument("--n_iters_teacher2sample", type=int, default=-1)

    parser.add_argument("--seq_eval", nargs="+", type=int)
    parser.add_argument("--seq_eval_flw", action='store_true')

    parser.add_argument("--overall_speaker_prefix", type=str, default="")
    parser.add_argument(
        "--use_train_subset", action='store_true',
        help="use a subset of the original train data for validation for debug")
    parser.add_argument("--use_test_set", action='store_true')
    parser.add_argument("--save_scores", type=str, default='')

    # FAST
    parser.add_argument("--fast_greedy_search", action='store_true')
    parser.add_argument("--search_mean", action='store_true')
    # parser.add_argument("--logit", action='store_true')
    parser.add_argument("--allow_revisit", action='store_true')
    parser.add_argument("--show_loss", action='store_true')
    parser.add_argument("--inject_stop", help='Force injecting the stop action in any place.', action='store_true')


    # R4R
    parser.add_argument("--r4r", action='store_true')
    parser.add_argument("--r4r_follow_detailed_path", action='store_true')
    parser.add_argument("--r4r_reward1", action='store_true')
    parser.add_argument("--r4r_reward2", action='store_true')
    parser.add_argument("--instructions_per_path", type=int, default=3)
    parser.add_argument("--r4r_small", type=str, default='')
    parser.add_argument("--r4r_result_prefix", type=str, default='')
    parser.add_argument("--r4r_eval", type=int, default=-1)
    parser.add_argument("--r4r_eval_split", type=int, default=-1)
    parser.add_argument('--show_metrics', nargs='+', default=["success_rate", "spl"])
    parser.add_argument("--spl_based_on_annotated_length", action='store_true')



    return parser


if __name__ == "__main__":
    utils.run(make_arg_parser(), train_val)
