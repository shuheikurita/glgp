''' Utils for io, language, connectivity graphs etc '''

import os
import sys
import re
import string
import json
import time
import math
from collections import Counter
import numpy as np
import networkx as nx
import subprocess
import itertools
import base64

import torch
import logging

logger = logging.getLogger(__name__)

# padding, unknown word, end of sentence
base_vocab = ['<PAD>', '<UNK>', '<EOS>', '<BOS>']

vocab_pad_idx = base_vocab.index('<PAD>')
vocab_unk_idx = base_vocab.index('<UNK>')
vocab_eos_idx = base_vocab.index('<EOS>')
vocab_bos_idx = base_vocab.index('<BOS>')

def load_nav_graphs(scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


def load_datasets(splits, base_path="tasks/R2R/data/R2R"):
    data = []
    for split in splits:
        with open(base_path+'_%s.json' % split) as f:
            datum = json.load(f)
        print("load_datasets: ",split,len(datum))
        data+=datum
    return data

def decode_base64(string):
    if sys.version_info[0] == 2:
        return base64.decodestring(bytearray(string))
    elif sys.version_info[0] == 3:
        return base64.decodebytes(bytearray(string, 'utf-8'))
    else:
        raise ValueError("decode_base64 can't handle python version {}".format(sys.version_info[0]))

class Tokenizer(object):
    ''' Class to tokenize and encode a sentence. '''
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # Split on any non-alphanumeric character

    def __init__(self, vocab=None):
        self.vocab = vocab
        self.word_to_index = {}
        if vocab:
            for i,word in enumerate(vocab):
                self.word_to_index[word] = i

    @staticmethod
    def split_sentence(sentence):
        ''' Break sentence into a list of words and punctuation '''
        toks = []
        for word in [s.strip().lower() for s in Tokenizer.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def encode_sentence(self, sentence, include_tokens=False):
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')
        encoding = []
        words = Tokenizer.split_sentence(sentence)
        for word in words:
            if word in self.word_to_index:
                encoding.append(self.word_to_index[word])
            else:
                encoding.append(vocab_unk_idx)
        #encoding.append(vocab_eos_idx)
        #utterance_length = len(encoding)
        #if utterance_length < self.encoding_length:
            #encoding += [vocab_pad_idx] * (self.encoding_length - len(encoding))
        #encoding = encoding[:self.encoding_length] # leave room for unks
        arr = np.array(encoding)
        if include_tokens:
            return arr, len(encoding), words
        else:
            return arr, len(encoding)

    def decode_sentence(self, encoding, break_on_eos=False, join=True):
        sentence = []
        for ix in encoding:
            if ix == (vocab_eos_idx  if break_on_eos else vocab_pad_idx):
                break
            else:
                sentence.append(self.vocab[ix])
        if join:
            return " ".join(sentence)
        return sentence


def build_vocab(splits=['train'], min_count=5, start_vocab=base_vocab):
    ''' Build a vocab, starting with base vocab containing a few useful tokens. '''
    count = Counter()
    raise NotImplemented("No R2R compatible! For R2R, just remove me.")
    data = load_datasets(splits)
    for item in data:
        for instr in item['instructions']:
            count.update(Tokenizer.split_sentence(instr))
    vocab = list(start_vocab)
    for word,num in count.most_common():
        if num >= min_count:
            vocab.append(word)
        else:
            break
    return vocab


def write_vocab(vocab, path):
    print('Writing vocab of size %d to %s' % (len(vocab),path))
    with open(path, 'w') as f:
        for word in vocab:
            f.write("%s\n" % word)


def read_vocab(path):
    vocab = []
    with open(path) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def k_best_indices(arr, k, sorted=False):
    # https://stackoverflow.com/a/23734295
    if k >= len(arr):
        if sorted:
            return np.argsort(arr)
        else:
            return np.arange(0, len(arr))
    ind = np.argpartition(arr, -k)[-k:]
    if sorted:
        ind = ind[np.argsort(arr[ind])]
    return ind

def structured_map(function, *args, **kwargs):
    #assert all(len(a) == len(args[0]) for a in args[1:])
    nested = kwargs.get('nested', False)
    acc = []
    for t in zip(*args):
        if nested:
            mapped = [function(*inner_t) for inner_t in zip(*t)]
        else:
            mapped = function(*t)
        acc.append(mapped)
    return acc

def rec_numpy2list(obj):
    if isinstance(obj,np.ndarray):
        return obj.tolist()
    elif isinstance(obj,np.generic):
        return obj.item()
    elif isinstance(obj,list):
        return [rec_numpy2list(o) for o in obj]
    elif isinstance(obj,dict):
        return {k:rec_numpy2list(v) for k,v in obj.items()}
    elif isinstance(obj,set):
        return {rec_numpy2list(v) for v in obj}
    else:
        return obj
def rec_torch2numpy(obj):
    if isinstance(obj,torch.Tensor):
        return obj.cpu().data.numpy()
    elif isinstance(obj,list):
        return [rec_torch2numpy(o) for o in obj]
    elif isinstance(obj,dict):
        return {k:rec_torch2numpy(v) for k,v in obj.items()}
    elif isinstance(obj,set):
        return {rec_torch2numpy(v) for v in obj}
    else:
        return obj


def flatten(lol):
    return [l for lst in lol for l in lst]

def all_equal(lst):
    return all(x == lst[0] for x in lst[1:])

def try_cuda(pytorch_obj):
    import torch.cuda
    try:
        disabled = torch.cuda.disabled
    except:
        disabled = False
    if torch.cuda.is_available() and not disabled:
        return pytorch_obj.cuda()
    else:
        return pytorch_obj

def pretty_json_dump(obj, fp):
    json.dump(obj, fp, sort_keys=True, indent=4, separators=(',', ':'))

def spatial_feature_from_bbox(bboxes, im_h, im_w):
    # from Ronghang Hu
    # https://github.com/ronghanghu/cmn/blob/ff7d519b808f4b7619b17f92eceb17e53c11d338/models/spatial_feat.py

    # Generate 5-dimensional spatial features from the image
    # [xmin, ymin, xmax, ymax, S] where S is the area of the box
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    bboxes = bboxes.reshape((-1, 4))
    # Check the size of the bounding boxes
    assert np.all(bboxes[:, 0:2] >= 0)
    assert np.all(bboxes[:, 0] <= bboxes[:, 2])
    assert np.all(bboxes[:, 1] <= bboxes[:, 3])
    assert np.all(bboxes[:, 2] <= im_w)
    assert np.all(bboxes[:, 3] <= im_h)

    feats = np.zeros((bboxes.shape[0], 5), dtype=np.float32)
    feats[:, 0] = bboxes[:, 0] * 2.0 / im_w - 1  # x1
    feats[:, 1] = bboxes[:, 1] * 2.0 / im_h - 1  # y1
    feats[:, 2] = bboxes[:, 2] * 2.0 / im_w - 1  # x2
    feats[:, 3] = bboxes[:, 3] * 2.0 / im_h - 1  # y2
    feats[:, 4] = (feats[:, 2] - feats[:, 0]) * (feats[:, 3] - feats[:, 1]) # S
    return feats

def run(arg_parser, entry_function):
    arg_parser.add_argument("--pdb", action='store_true')
    arg_parser.add_argument("--ipdb", action='store_true')
    arg_parser.add_argument("--no_cuda", action='store_true')
    arg_parser.add_argument("--git", action='store_true')
    arg_parser.add_argument("--debug", action='store_true')
    arg_parser.add_argument("--debug_minimal", type=int, default=-1)

    args = arg_parser.parse_args()

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.DEBUG if args.debug else logging.INFO)
                        #level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    #logger = logging.getLogger(__name__)

    import torch.cuda
    # todo: yuck
    torch.cuda.disabled = args.no_cuda

    def log(out_file):
        subprocess.call("git rev-parse HEAD", shell=True, stdout=out_file)
        subprocess.call("git --no-pager diff", shell=True, stdout=out_file)
        out_file.write('\n\n')
        out_file.write(' '.join(sys.argv))
        out_file.write('\n\n')
        json.dump(vars(args), out_file)
        out_file.write('\n\n')

    if args.git:
        log(sys.stdout)
    # if 'save_dir' in vars(args) and args.save_dir:
    #     with open(os.path.join(args.save_dir, 'invoke.log'), 'w') as f:
    #         log(f)

    if args.ipdb:
        import ipdb
        ipdb.runcall(entry_function, args)
    elif args.pdb:
        import pdb
        pdb.runcall(entry_function, args)
    else:
        entry_function(args)

def assert2(A,B):
    assert A==B,(A,B)


def get_model_prefix(args, image_feature_list):
    image_feature_name = "+".join(
        [featurizer.get_name() for featurizer in image_feature_list])
    model_prefix = '{}_{}_{}_{}'.format(
        args.model_name, args.feedback_method, image_feature_name, args.loss_type)
    if args.use_train_subset:
        model_prefix = 'trainsub_' + model_prefix
    if args.bidirectional:
        model_prefix = model_prefix + "_bidirectional"
    if args.use_pretraining:
        #model_prefix = model_prefix.replace(
        #    'follower', 'follower_with_pretraining', 1)
        model_prefix += "_with_pretraining"
    if args.follower_prefix!="":
        model_prefix += "_fp="+args.follower_model_name
    if args.speaker_prefix!="":
        model_prefix += "_sp="+args.speaker_model_name
    #if args.follower_prefix!="":
    #    model_prefix += "_fp="+args.follower_prefix.split("/")[-1]
    #if args.speaker_prefix!="":
    #    model_prefix += "_sp="+args.speaker_prefix.split("/")[-1]
    return model_prefix


# Partially update the models' state_dict
def update_state_dict(model, pretrained_dict):
    # 0. get state_dict
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model