''' Batched Room-to-Room navigation environment '''

import sys
sys.path.append('build')
import MatterSim
import csv
import numpy as np
import math
import json
import random
import networkx as nx
import functools
import os.path
import time
import paths
import pickle
import os
import os.path
import sys
import itertools
import logging

from collections import namedtuple, defaultdict

from utils import load_datasets, load_nav_graphs, structured_map, vocab_pad_idx, decode_base64, k_best_indices, try_cuda, spatial_feature_from_bbox

import torch
from torch.autograd import Variable

csv.field_size_limit(sys.maxsize)

logger = logging.getLogger(__name__)

# Not needed for panorama action space
# FOLLOWER_MODEL_ACTIONS = ['left', 'right', 'up', 'down', 'forward', '<end>', '<start>', '<ignore>']
#
# LEFT_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("left")
# RIGHT_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("right")
# UP_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("up")
# DOWN_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("down")
# FORWARD_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("forward")
# END_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("<end>")
# START_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("<start>")
# IGNORE_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("<ignore>")


# FOLLOWER_ENV_ACTIONS = [
#     (0,-1, 0), # left
#     (0, 1, 0), # right
#     (0, 0, 1), # up
#     (0, 0,-1), # down
#     (1, 0, 0), # forward
#     (0, 0, 0), # <end>
#     (0, 0, 0), # <start>
#     (0, 0, 0)  # <ignore>
# ]

# assert len(FOLLOWER_MODEL_ACTIONS) == len(FOLLOWER_ENV_ACTIONS)

angle_inc = np.pi / 6.


def _build_action_embedding(adj_loc_list, features, init_emb=None):
    feature_dim = features.shape[-1]
    embedding = np.zeros((len(adj_loc_list), feature_dim + 128), np.float32)
    for a, adj_dict in enumerate(adj_loc_list):
        if a == 0:
            # the embedding for the first action ('stop') is left as zero
            pass
        elif a == -1:
            embedding = init_emb
        else:
            embedding[a, :feature_dim] = features[adj_dict['absViewIndex']]
            loc_embedding = embedding[a, feature_dim:]
            rel_heading = adj_dict['rel_heading']
            rel_elevation = adj_dict['rel_elevation']
            loc_embedding[0:32] = np.sin(rel_heading)
            loc_embedding[32:64] = np.cos(rel_heading)
            loc_embedding[64:96] = np.sin(rel_elevation)
            loc_embedding[96:] = np.cos(rel_elevation)
    return embedding

def _initial_action_embedding(feature_dim):
    logger.info("feature_dim %d"%feature_dim)
    embedding = np.random.rand(1, feature_dim + 128)*0.01 #, np.float32)*0.01
    return embedding

def build_viewpoint_loc_embedding(viewIndex):
    """
    Position embedding:
    heading 64D + elevation 64D
    1) heading: [sin(heading) for _ in range(1, 33)] +
                [cos(heading) for _ in range(1, 33)]
    2) elevation: [sin(elevation) for _ in range(1, 33)] +
                  [cos(elevation) for _ in range(1, 33)]
    """
    embedding = np.zeros((36, 128), np.float32)
    for absViewIndex in range(36):
        relViewIndex = (absViewIndex - viewIndex) % 12 + (absViewIndex // 12) * 12
        rel_heading = (relViewIndex % 12) * angle_inc
        rel_elevation = (relViewIndex // 12 - 1) * angle_inc
        embedding[absViewIndex,  0:32] = np.sin(rel_heading)
        embedding[absViewIndex, 32:64] = np.cos(rel_heading)
        embedding[absViewIndex, 64:96] = np.sin(rel_elevation)
        embedding[absViewIndex,   96:] = np.cos(rel_elevation)
    return embedding


# pre-compute all the 36 possible paranoram location embeddings
_static_loc_embeddings = [
    build_viewpoint_loc_embedding(viewIndex) for viewIndex in range(36)]


def _loc_distance(loc):
    return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)


def _canonical_angle(x):
    ''' Make angle in (-pi, +pi) '''
    return x - 2 * np.pi * round(x / (2 * np.pi))


def _adjust_heading(sim, heading):
    heading = (heading + 6) % 12 - 6  # minimum action to turn (e.g 11 -> -1)
    ''' Make possibly more than one heading turns '''
    for _ in range(int(abs(heading))):
        sim.makeAction(0, np.sign(heading), 0)


def _adjust_elevation(sim, elevation):
    for _ in range(int(abs(elevation))):
        ''' Make possibly more than one elevation turns '''
        sim.makeAction(0, 0, np.sign(elevation))


def _navigate_to_location(sim, nextViewpointId, absViewIndex):
    state = sim.getState()
    if state.location.viewpointId == nextViewpointId:
        return  # do nothing

    # 1. Turn to the corresponding view orientation
    _adjust_heading(sim, absViewIndex % 12 - state.viewIndex % 12)
    _adjust_elevation(sim, absViewIndex // 12 - state.viewIndex // 12)
    # find the next location
    state = sim.getState()
    assert state.viewIndex == absViewIndex
    a, next_loc = None, None
    for n_loc, loc in enumerate(state.navigableLocations):
        if loc.viewpointId == nextViewpointId:
            a = n_loc
            next_loc = loc
            break
    assert next_loc is not None

    # 3. Take action
    sim.makeAction(a, 0, 0)


def _get_panorama_states(sim):
    '''
    Look around and collect all the navigable locations

    Representation of all_adj_locs:
        {'absViewIndex': int,
         'relViewIndex': int,
         'nextViewpointId': int,
         'rel_heading': float,
         'rel_elevation': float}
        where relViewIndex is normalized using the current heading

    Concepts:
        - absViewIndex: the absolute viewpoint index, as returned by
          state.viewIndex
        - nextViewpointId: the viewpointID of this adjacent point
        - rel_heading: the heading (radians) of this adjacent point
          relative to looking forward horizontally (i.e. relViewIndex 12)
        - rel_elevation: the elevation (radians) of this adjacent point
          relative to looking forward horizontally (i.e. relViewIndex 12)

    Features are 36 x D_vis, ordered from relViewIndex 0 to 35 (i.e.
    feature[12] is always the feature of the patch forward horizontally)
    '''
    state = sim.getState()
    initViewIndex = state.viewIndex
    # 1. first look down, turning to relViewIndex 0
    elevation_delta = -(state.viewIndex // 12)
    _adjust_elevation(sim, elevation_delta)

    # 2. scan through the 36 views and collect all navigable locations
    adj_dict = {}
    for relViewIndex in range(36):
        # Here, base_rel_heading and base_rel_elevation are w.r.t
        # relViewIndex 12 (looking forward horizontally)
        # (i.e. the relative heading and elevation
        # adjustment needed to switch from relViewIndex 12
        # to the current relViewIndex)
        base_rel_heading = (relViewIndex % 12) * angle_inc
        base_rel_elevation = (relViewIndex // 12 - 1) * angle_inc

        state = sim.getState()
        absViewIndex = state.viewIndex
        # get adjacent locations
        for loc in state.navigableLocations[1:]:
            distance = _loc_distance(loc)
            # if a loc is visible from multiple view, use the closest
            # view (in angular distance) as its representation
            if (loc.viewpointId not in adj_dict or
                    distance < adj_dict[loc.viewpointId]['distance']):
                rel_heading = _canonical_angle(
                    base_rel_heading + loc.rel_heading)
                rel_elevation = base_rel_elevation + loc.rel_elevation
                adj_dict[loc.viewpointId] = {
                    'absViewIndex': absViewIndex,
                    'nextViewpointId': loc.viewpointId,
                    'rel_heading': rel_heading,
                    'rel_elevation': rel_elevation,
                    'distance': distance}
        # move to the next view
        if (relViewIndex + 1) % 12 == 0:
            sim.makeAction(0, 1, 1)  # Turn right and look up
        else:
            sim.makeAction(0, 1, 0)  # Turn right
    # 3. turn back to the original view
    _adjust_elevation(sim, - 2 - elevation_delta)
    state = sim.getState()
    assert state.viewIndex == initViewIndex  # check the agent is back
    # collect navigable location list
    stop = {
        'absViewIndex': -1,
        'nextViewpointId': state.location.viewpointId}
    adj_loc_list = [stop] + sorted(
            adj_dict.values(), key=lambda x: abs(x['rel_heading']))

    return state, adj_loc_list


WorldState = namedtuple("WorldState", ["scanId", "viewpointId", "heading", "elevation"])

BottomUpViewpoint = namedtuple("BottomUpViewpoint", ["cls_prob", "image_features", "attribute_indices", "object_indices", "spatial_features", "no_object_mask"])

def load_world_state(sim, world_state):
    sim.newEpisode(*world_state)

def get_world_state(sim):
    state = sim.getState()
    return WorldState(scanId=state.scanId,
                      viewpointId=state.location.viewpointId,
                      heading=state.heading,
                      elevation=state.elevation)

def make_sim(image_w, image_h, vfov):
    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(True)
    sim.setCameraResolution(image_w, image_h)
    sim.setCameraVFOV(math.radians(vfov))
    sim.init()
    return sim

# def encode_action_sequence(action_tuples):
#     encoded = []
#     reached_end = False
#     if action_tuples[0] == (0, 0, 0):
#         # this method can't handle a <start> symbol
#         assert all(t == (0, 0, 0) for t in action_tuples)
#     for tpl in action_tuples:
#         if tpl == (0, 0, 0):
#             if reached_end:
#                 ix = IGNORE_ACTION_INDEX
#             else:
#                 ix = END_ACTION_INDEX
#                 reached_end = True
#         else:
#             ix = FOLLOWER_ENV_ACTIONS.index(tpl)
#         encoded.append(ix)
#     return encoded

# Not needed for panorama action space
# def index_action_tuple(action_tuple):
#     ix, heading_chg, elevation_chg = action_tuple
#     if heading_chg > 0:
#         return FOLLOWER_MODEL_ACTIONS.index('right')
#     elif heading_chg < 0:
#         return FOLLOWER_MODEL_ACTIONS.index('left')
#     elif elevation_chg > 0:
#         return FOLLOWER_MODEL_ACTIONS.index('up')
#     elif elevation_chg < 0:
#         return FOLLOWER_MODEL_ACTIONS.index('down')
#     elif ix > 0:
#         return FOLLOWER_MODEL_ACTIONS.index('forward')
#     else:
#         return FOLLOWER_MODEL_ACTIONS.index('<end>')

class ImageFeatures(object):
    NUM_VIEWS = 36
    MEAN_POOLED_DIM = 2048
    feature_dim = MEAN_POOLED_DIM

    IMAGE_W = 640
    IMAGE_H = 480
    VFOV = 60

    @staticmethod
    def from_args(args):
        feats = []
        for image_feature_type in sorted(args.image_feature_type):
            if image_feature_type == "none":
                feats.append(NoImageFeatures())
            elif image_feature_type == "bottom_up_attention":
                # feats.append(BottomUpImageFeatures(
                #     args.bottom_up_detections,
                #     #precomputed_cache_path=paths.bottom_up_feature_cache_path,
                #     precomputed_cache_dir=paths.bottom_up_feature_cache_dir,
                # ))
                raise NotImplementedError('bottom_up_attention has not been implemented for panorama environment')
            elif image_feature_type == "convolutional_attention":
                feats.append(ConvolutionalImageFeatures(
                    args.image_feature_datasets,
                    split_convolutional_features=True,
                    downscale_convolutional_features=args.downscale_convolutional_features
                ))
                #raise NotImplementedError('convolutional_attention has not been implemented for panorama environment')
            else:
                assert image_feature_type == "mean_pooled"
                feats.append(MeanPooledImageFeatures(args.image_feature_datasets))
        return feats

    @staticmethod
    def add_args(argument_parser):
        argument_parser.add_argument("--image_feature_type", nargs="+", choices=["none", "mean_pooled", "convolutional_attention", "bottom_up_attention"], default=["mean_pooled"])
        argument_parser.add_argument("--image_attention_size", type=int)
        argument_parser.add_argument("--image_feature_datasets", nargs="+", choices=["imagenet", "places365"], default=["imagenet"], help="only applicable to mean_pooled or convolutional_attention options for --image_feature_type")
        argument_parser.add_argument("--bottom_up_detections", type=int, default=20)
        argument_parser.add_argument("--bottom_up_detection_embedding_size", type=int, default=20)
        argument_parser.add_argument("--downscale_convolutional_features", action='store_true')

    def get_name(self):
        raise NotImplementedError("get_name")

    def batch_features(self, feature_list):
        features = np.stack(feature_list)
        return try_cuda(Variable(torch.from_numpy(features), requires_grad=False))

    def get_features(self, state):
        raise NotImplementedError("get_features")

class NoImageFeatures(ImageFeatures):
    feature_dim = ImageFeatures.MEAN_POOLED_DIM

    def __init__(self):
        logger.info('Image features not provided')
        self.features = np.zeros((ImageFeatures.NUM_VIEWS, self.feature_dim), dtype=np.float32)

    def get_features(self, state):
        return self.features

    def get_name(self):
        return "none"

class MeanPooledImageFeatures(ImageFeatures):
    def __init__(self, image_feature_datasets):
        image_feature_datasets = sorted(image_feature_datasets)
        self.image_feature_datasets = image_feature_datasets

        self.mean_pooled_feature_stores = [paths.mean_pooled_feature_store_paths[dataset]
                                           for dataset in image_feature_datasets]
        self.feature_dim = MeanPooledImageFeatures.MEAN_POOLED_DIM * len(image_feature_datasets)
        logger.info('Loading image features from %s' % ', '.join(self.mean_pooled_feature_stores))
        tsv_fieldnames = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']
        self.features = defaultdict(list)
        for mpfs in self.mean_pooled_feature_stores:
            with open(mpfs, "rt") as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = tsv_fieldnames)
                for item in reader:
                    assert int(item['image_h']) == ImageFeatures.IMAGE_H
                    assert int(item['image_w']) == ImageFeatures.IMAGE_W
                    assert int(item['vfov']) == ImageFeatures.VFOV
                    long_id = self._make_id(item['scanId'], item['viewpointId'])
                    features = np.frombuffer(decode_base64(item['features']), dtype=np.float32).reshape((ImageFeatures.NUM_VIEWS, ImageFeatures.MEAN_POOLED_DIM))
                    self.features[long_id].append(features)
            #import ipdb; ipdb.set_trace()
        assert all(len(feats) == len(self.mean_pooled_feature_stores) for feats in self.features.values())
        self.features = {
            long_id: np.concatenate(feats, axis=1)
            for long_id, feats in self.features.items()
        }
        logger.info('Number of Loaded image features %d'%len(self.features))
        assert "VzqfbhrpDEA_0331b320ae1f41d0b42fdf0100e56bd2" in self.features

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def get_features(self, state):
        long_id = self._make_id(state.scanId, state.location.viewpointId)
        # Return feature of all the 36 views
        return self.features[long_id]

    def get_name(self):
        name = '+'.join(sorted(self.image_feature_datasets))
        name = "{}_mean_pooled".format(name)
        return name

class ConvolutionalImageFeatures(ImageFeatures):
    feature_dim = ImageFeatures.MEAN_POOLED_DIM

    def __init__(self, image_feature_datasets, split_convolutional_features=True, downscale_convolutional_features=True):
        self.image_feature_datasets = image_feature_datasets
        self.split_convolutional_features = split_convolutional_features
        self.downscale_convolutional_features = downscale_convolutional_features

        self.convolutional_feature_stores = [paths.convolutional_feature_store_paths[dataset]
                                             for dataset in image_feature_datasets]

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    @functools.lru_cache(maxsize=3000)
    def _get_convolutional_features(self, scanId, viewpointId, viewIndex):
        feats = []
        for cfs in self.convolutional_feature_stores:
            if self.split_convolutional_features:
                path = os.path.join(cfs, scanId, "{}_{}{}.npy".format(viewpointId, viewIndex, "_downscaled" if self.downscale_convolutional_features else ""))
                this_feats = np.load(path)
            else:
                # memmap for loading subfeatures
                path = os.path.join(cfs, scanId, "%s.npy" % viewpointId)
                mmapped = np.load(path, mmap_mode='r')
                this_feats = mmapped[viewIndex,:,:,:]
            feats.append(this_feats)
        import ipdb; ipdb.set_trace()
        if len(feats) > 1:
            return np.concatenate(feats, axis=1)
        return feats[0]

    def get_features(self, state):
        return self._get_convolutional_features(state.scanId, state.location.viewpointId, state.viewIndex)

    def get_name(self):
        name = '+'.join(sorted(self.image_feature_datasets))
        name = "{}_convolutional_attention".format(name)
        if self.downscale_convolutional_features:
            name = name + "_downscale"
        return name

class BottomUpImageFeatures(ImageFeatures):
    PAD_ITEM = ("<pad>",)
    feature_dim = ImageFeatures.MEAN_POOLED_DIM

    def __init__(self, number_of_detections, precomputed_cache_path=None, precomputed_cache_dir=None, image_width=640, image_height=480):
        self.number_of_detections = number_of_detections
        self.index_to_attributes, self.attribute_to_index = BottomUpImageFeatures.read_visual_genome_vocab(paths.bottom_up_attribute_path, BottomUpImageFeatures.PAD_ITEM, add_null=True)
        self.index_to_objects, self.object_to_index = BottomUpImageFeatures.read_visual_genome_vocab(paths.bottom_up_object_path, BottomUpImageFeatures.PAD_ITEM, add_null=False)

        self.num_attributes = len(self.index_to_attributes)
        self.num_objects = len(self.index_to_objects)

        self.attribute_pad_index = self.attribute_to_index[BottomUpImageFeatures.PAD_ITEM]
        self.object_pad_index = self.object_to_index[BottomUpImageFeatures.PAD_ITEM]

        self.image_width = image_width
        self.image_height = image_height

        self.precomputed_cache = {}
        def add_to_cache(key, viewpoints):
            assert len(viewpoints) == ImageFeatures.NUM_VIEWS
            viewpoint_feats = []
            for viewpoint in viewpoints:
                params = {}
                for param_key, param_value in viewpoint.items():
                    if param_key == 'cls_prob':
                        # make sure it's in descending order
                        assert np.all(param_value[:-1] >= param_value[1:])
                    if param_key == 'boxes':
                        # TODO: this is for backward compatibility, remove it
                        param_key = 'spatial_features'
                        param_value = spatial_feature_from_bbox(param_value, self.image_height, self.image_width)
                    assert len(param_value) >= self.number_of_detections
                    params[param_key] = param_value[:self.number_of_detections]
                viewpoint_feats.append(BottomUpViewpoint(**params))
            self.precomputed_cache[key] = viewpoint_feats

        if precomputed_cache_dir:
            self.precomputed_cache = {}
            import glob
            for scene_dir in glob.glob(os.path.join(precomputed_cache_dir, "*")):
                scene_id = os.path.basename(scene_dir)
                pickle_file = os.path.join(scene_dir, "d={}.pkl".format(number_of_detections))
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)
                    for (viewpoint_id, viewpoints) in data.items():
                        key = (scene_id, viewpoint_id)
                        add_to_cache(key, viewpoints)
        elif precomputed_cache_path:
            self.precomputed_cache = {}
            with open(precomputed_cache_path, 'rb') as f:
                data = pickle.load(f)
                for (key, viewpoints) in data.items():
                    add_to_cache(key, viewpoints)

    @staticmethod
    def read_visual_genome_vocab(fname, pad_name, add_null=False):
        # one-to-many mapping from indices to names (synonyms)
        index_to_items = []
        item_to_index = {}
        start_ix = 0
        items_to_add = [pad_name]
        if add_null:
            null_tp = ()
            items_to_add.append(null_tp)
        for item in items_to_add:
            index_to_items.append(item)
            item_to_index[item] = start_ix
            start_ix += 1

        with open(fname) as f:
            for index, line in enumerate(f):
                this_items = []
                for synonym in line.split(','):
                    item = tuple(synonym.split())
                    this_items.append(item)
                    item_to_index[item] = index + start_ix
                index_to_items.append(this_items)
        assert len(index_to_items) == max(item_to_index.values()) + 1
        return index_to_items, item_to_index

    def batch_features(self, feature_list):
        def transform(lst, wrap_with_var=True):
            features = np.stack(lst)
            x = torch.from_numpy(features)
            if wrap_with_var:
                x = Variable(x, requires_grad=False)
            return try_cuda(x)

        return BottomUpViewpoint(
            cls_prob=transform([f.cls_prob for f in feature_list]),
            image_features=transform([f.image_features for f in feature_list]),
            attribute_indices=transform([f.attribute_indices for f in feature_list]),
            object_indices=transform([f.object_indices for f in feature_list]),
            spatial_features=transform([f.spatial_features for f in feature_list]),
            no_object_mask=transform([f.no_object_mask for f in feature_list], wrap_with_var=False),
        )

    def parse_attribute_objects(self, tokens):
        parse_options = []
        # allow blank attribute, but not blank object
        for split_point in range(0, len(tokens)):
            attr_tokens = tuple(tokens[:split_point])
            obj_tokens = tuple(tokens[split_point:])
            if attr_tokens in self.attribute_to_index and obj_tokens in self.object_to_index:
                parse_options.append((self.attribute_to_index[attr_tokens], self.object_to_index[obj_tokens]))
        assert parse_options, "didn't find any parses for {}".format(tokens)
        # prefer longer objects, e.g. "electrical outlet" over "electrical" "outlet"
        return parse_options[0]

    @functools.lru_cache(maxsize=20000)
    def _get_viewpoint_features(self, scan_id, viewpoint_id):
        if self.precomputed_cache:
            return self.precomputed_cache[(scan_id, viewpoint_id)]

        fname = os.path.join(paths.bottom_up_feature_store_path, scan_id, "{}.p".format(viewpoint_id))
        with open(fname, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        viewpoint_features = []
        for viewpoint in data:
            top_indices = k_best_indices(viewpoint['cls_prob'], self.number_of_detections, sorted=True)[::-1]

            no_object = np.full(self.number_of_detections, True, dtype=np.bool) # will become torch Byte tensor
            no_object[0:len(top_indices)] = False

            cls_prob = np.zeros(self.number_of_detections, dtype=np.float32)
            cls_prob[0:len(top_indices)] = viewpoint['cls_prob'][top_indices]
            assert cls_prob[0] == np.max(cls_prob)

            image_features = np.zeros((self.number_of_detections, ImageFeatures.MEAN_POOLED_DIM), dtype=np.float32)
            image_features[0:len(top_indices)] = viewpoint['features'][top_indices]

            spatial_feats = np.zeros((self.number_of_detections, 5), dtype=np.float32)
            spatial_feats[0:len(top_indices)] = spatial_feature_from_bbox(viewpoint['boxes'][top_indices], self.image_height, self.image_width)

            object_indices = np.full(self.number_of_detections, self.object_pad_index)
            attribute_indices = np.full(self.number_of_detections, self.attribute_pad_index)

            for i, ix in enumerate(top_indices):
                attribute_ix, object_ix = self.parse_attribute_objects(list(viewpoint['captions'][ix].split()))
                object_indices[i] = object_ix
                attribute_indices[i] = attribute_ix

            viewpoint_features.append(BottomUpViewpoint(cls_prob, image_features, attribute_indices, object_indices, spatial_feats, no_object))
        return viewpoint_features

    def get_features(self, state):
        viewpoint_features = self._get_viewpoint_features(state.scanId, state.location.viewpointId)
        return viewpoint_features[state.viewIndex]

    def get_name(self):
        return "bottom_up_attention_d={}".format(self.number_of_detections)

class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, batch_size, beam_size, jagged=False):
        assert jagged==False or len(jagged)==batch_size
        self.sims = []
        self.batch_size = batch_size
        self.beam_size = beam_size if not jagged else jagged
        self.jagged = jagged
        for i in range(batch_size):
            beam = []
            if not self.jagged:
                for j in range(self.beam_size):
                    sim = make_sim(ImageFeatures.IMAGE_W, ImageFeatures.IMAGE_H, ImageFeatures.VFOV)
                    beam.append(sim)
            else:
                for j in range(self.jagged[i]):
                    sim = make_sim(ImageFeatures.IMAGE_W, ImageFeatures.IMAGE_H, ImageFeatures.VFOV)
                    beam.append(sim)
            self.sims.append(beam)

    def sims_view(self, beamed):
        if beamed:
            return [itertools.cycle(sim_list) for sim_list in self.sims]
        else:
            return (s[0] for s in self.sims)

    def newEpisodes(self, scanIds, viewpointIds, headings, beamed=False):
        assert len(scanIds) == len(viewpointIds)
        assert len(headings) == len(viewpointIds)
        assert len(scanIds) == len(self.sims)
        world_states = []
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            world_state = WorldState(scanId, viewpointId, heading, 0)
            if beamed:
                world_states.append([world_state])
            else:
                world_states.append(world_state)
            load_world_state(self.sims[i][0], world_state)
        assert len(world_states) == len(scanIds)
        return world_states

    def getStates(self, world_states, beamed=False):
        ''' Get list of states. '''
        def f(sim, world_state):
            load_world_state(sim, world_state)
            return _get_panorama_states(sim)
        return structured_map(f, self.sims_view(beamed), world_states, nested=beamed)

    def makeActions(self, world_states, actions, last_obs, beamed=False):
        ''' Take an action using the full state dependent action interface (with batched input).
            Each action is an index in the adj_loc_list,
            0 means staying still (i.e. stop)
        '''
        def f(sim, world_state, action, last_ob):
            load_world_state(sim, world_state)
            # load the location attribute corresponding to the action
            loc_attr = last_ob['adj_loc_list'][action]
            _navigate_to_location(
                sim, loc_attr['nextViewpointId'], loc_attr['absViewIndex'])
            # sim.makeAction(index, heading, elevation)
            return get_world_state(sim)
        return structured_map(f, self.sims_view(beamed), world_states, actions, last_obs, nested=beamed)

    def makeActions_and_getStates(self, world_states, actions, last_obs, beamed=False):
        def f(sim, world_state, action, last_ob):
            load_world_state(sim, world_state)
            # load the location attribute corresponding to the action
            loc_attr = last_ob['adj_loc_list'][action]
            _navigate_to_location(
                sim, loc_attr['nextViewpointId'], loc_attr['absViewIndex'])
            # sim.makeAction(index, heading, elevation)
            next_world_state = get_world_state(sim)
            next_panorama_state = _get_panorama_states(sim)
            return next_world_state, next_panorama_state
        return structured_map(f, self.sims_view(beamed), world_states, actions, last_obs, nested=beamed)

    # def makeSimpleActions(self, simple_indices, beamed=False):
    #     ''' Take an action using a simple interface: 0-forward, 1-turn left, 2-turn right, 3-look up, 4-look down.
    #         All viewpoint changes are 30 degrees. Forward, look up and look down may not succeed - check state.
    #         WARNING - Very likely this simple interface restricts some edges in the graph. Parts of the
    #         environment may not longer be navigable. '''
    #     def f(sim, index):
    #         if index == 0:
    #             sim.makeAction(1, 0, 0)
    #         elif index == 1:
    #             sim.makeAction(0,-1, 0)
    #         elif index == 2:
    #             sim.makeAction(0, 1, 0)
    #         elif index == 3:
    #             sim.makeAction(0, 0, 1)
    #         elif index == 4:
    #             sim.makeAction(0, 0,-1)
    #         else:
    #             sys.exit("Invalid simple action %s" % index)
    #     structured_map(f, self.sims_view(beamed), simple_indices, nested=beamed)
    #     return None

class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, args, image_features_list, batch_size=-1, seed=10, splits=['train'],
                 tokenizer=None, beam_size=1, instruction_limit=None,
                 bert_tokenizer=None, debug_minimal=False, r2r_dataset_path=None):
        assert batch_size>0
        self.image_features_list = image_features_list
        self.data = []
        self.scans = []
        self.gt = {}
        data = load_datasets(splits, base_path=r2r_dataset_path)
        stat_length = defaultdict(int)
        for i,item in enumerate(data):
            # Split multiple instructions into separate entries
            if not args.r4r_small:
                assert item['path_id'] not in self.gt
            self.gt[item['path_id']] = item
            instructions = item['instructions']
            if instruction_limit:
                instructions = instructions[:instruction_limit]
            for j,instr in enumerate(instructions):
                self.scans.append(item['scan'])
                new_item = dict(item)
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                new_item['instructions'] = instr
                if tokenizer:
                    self.tokenizer = tokenizer
                    #new_item['instr_encoding'], new_item['instr_length'] = tokenizer.encode_sentence(instr)
                    new_item['instr_encoding'], new_item['instr_length'], new_item['words'] = self.tokenizer.encode_sentence(instr,include_tokens=True)
                    stat_length[new_item['instr_length']]+=1
                else:
                    self.tokenizer = None
                if bert_tokenizer:
                    self.bert_tokenizer = bert_tokenizer
                    new_item['bert_words'] = ["[CLS]"]+new_item['words']+["[SEP]"]
                    new_item['bert_subwords'] = bert_tokenizer.tokenize(" ".join(new_item['bert_words']))
                    new_item['instr_encoding_bert'] = bert_tokenizer.convert_tokens_to_ids(new_item['bert_subwords'])
                    #tokens_tensor = torch.tensor([new_item['instr_encoding_bert']])
                    #segments_tensors = torch.tensor([1]*len(new_item['instr_encoding_bert']))
                    #with torch.no_grad():
                    #    encoded_layers, _ = bert_model(try_cuda(tokens_tensor), try_cuda(segments_tensors))
                    #new_item['bert_embedding'] = encoded_layers
                    #print(new_item["words"])
                self.data.append(new_item)
            if debug_minimal>0 and i+1==debug_minimal:
                break
        self.scans = set(self.scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)
        # FAST
        self.instr_id_to_idx = {}
        for i,item in enumerate(self.data):
            self.instr_id_to_idx[item['instr_id']] = i
        #
        self.ix = 0
        self.batch_size = batch_size
        logger.info("R2R batch_size %d"%self.batch_size)
        self._load_nav_graphs()
        self.set_beam_size(beam_size)
        self.print_progress = False
        logger.info('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

        # test
        logger.debug(str({k:d for k,d in stat_length.items() if d!=0}))
        logger.debug("len(self.image_features_list) %d"%len(self.image_features_list))
        #starting_world_states = self.reset(load_next_minibatch=False)
        #path_obs, path_actions = self.shortest_paths_to_goals(starting_world_states, max_steps)
        #feature = self.image_features_list[0].get_features(starting_world_states[0])
        self.init_action_embedding = _initial_action_embedding(image_features_list[0].feature_dim)

        # R4R
        self.r4r_follow_detailed_path = args.r4r_follow_detailed_path
        self.r4r_reward1 = args.r4r_reward1
        self.r4r_reward2 = args.r4r_reward2

    def set_beam_size(self, beam_size, force_reload=False):
        # warning: this will invalidate the environment, self.reset() should be called afterward!
        try:
            invalid = (beam_size != self.beam_size)
        except:
            invalid = True
        if force_reload or invalid:
            self.beam_size = beam_size
            self.env = EnvBatch(self.batch_size, beam_size)

    def _load_nav_graphs(self):
        ''' Load connectivity graph for each scan, useful for reasoning about shortest paths '''
        logger.info('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, sort_instr_length):
        batch = self.data[self.ix:self.ix+self.batch_size]
        if self.print_progress:
            sys.stderr.write("\rix {} / {}".format(self.ix, len(self.data)))
        if len(batch) < self.batch_size:
            random.shuffle(self.data)
            # FAST
            for i,item in enumerate(self.data):
                self.instr_id_to_idx[item['instr_id']] = i
            #
            self.ix = self.batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += self.batch_size
        if sort_instr_length:
            batch = sorted(batch, key=lambda item: item['instr_length'], reverse=True)
        self.batch = batch

    def reset_epoch(self):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        self.ix = 0

    def _shortest_path_action(self, state, adj_loc_list, goalViewpointId):
        '''
        Determine next action on the shortest path to goal,
        for supervised training.
        '''
        if state.location.viewpointId == goalViewpointId:
            return 0  # do nothing
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        for n_a, loc_attr in enumerate(adj_loc_list):
            if loc_attr['nextViewpointId'] == nextViewpointId:
                return n_a

        # Next nextViewpointId not found! This should not happen!
        print('adj_loc_list:', adj_loc_list)
        print('nextViewpointId:', nextViewpointId)
        long_id = '{}_{}'.format(state.scanId, state.location.viewpointId)
        print('longId:', long_id)
        raise Exception('Bug: nextViewpointId not in adj_loc_list')

    def observe(self, world_states, beamed=False, include_teacher=True, status=None, instr_id=None, history_path=None, is_train=None):
        # history_path is required only when self.follow_detailed_path and include_teacher
        #start_time = time.time()
        if status is None:
            status = self.env.getStates(world_states, beamed=beamed)
        obs = []
        for i_batch,states_beam in enumerate(status):
            item = self.batch[i_batch]
            obs_batch = []
            for state, adj_loc_list in states_beam if beamed else [states_beam]:
                # FAST
                if item['scan'] != state.scanId:
                    item = self.data[self.instr_id_to_idx[instr_id]]
                    assert item['scan'] == state.scanId
                # /FAST
                assert item['scan'] == state.scanId, (item['scan'], state.scanId)
                feature = [featurizer.get_features(state) for featurizer in self.image_features_list]
                assert len(feature) == 1, 'for now, only work with MeanPooled feature'
                #print(feature[0].shape)
                #print(_static_loc_embeddings[state.viewIndex].shape)
                feature_with_loc = np.concatenate((feature[0], _static_loc_embeddings[state.viewIndex]), axis=-1)
                action_embedding = _build_action_embedding(adj_loc_list, feature[0], self.init_action_embedding)
                ob = {
                    'instr_id' : item['instr_id'],
                    'scan' : state.scanId,
                    'viewpoint' : state.location.viewpointId,
                    'viewIndex' : state.viewIndex,
                    'heading' : state.heading,
                    'elevation' : state.elevation,
                    'feature' : [feature_with_loc],
                    'step' : state.step,
                    'adj_loc_list' : adj_loc_list,
                    'action_embedding': action_embedding,
                    'navigableLocations' : state.navigableLocations,
                    'instructions' : item['instructions'],
                }
                if include_teacher:
                    path = item['path']
                    if self.r4r_follow_detailed_path:
                        assert not beamed
                        if not is_train:
                            ob['teacher'] = 0
                        else:
                            #if i_batch==0:
                            #    import ipdb; ipdb.set_trace()
                            if history_path is None: # the first time
                                t = 0
                            else:
                                t = len(history_path[i_batch])
                            #ob['teacher'] = self._shortest_path_action(state, adj_loc_list, item['path'][-1])
                            #assert t<=len(item["path"]), (i_batch,t+1,len(item["path"]))
                            #if t>0:
                            #    assert history_path[i_batch][t-1]==item["path"][t-1]
                            if t+1>=len(item["path"]):
                                ob['teacher'] = 0 # last time
                            else:
                                next_place = item['path'][t+1]
                                try: # assert the next place is neibouring
                                    assert next_place in [loc_attr['nextViewpointId'] for loc_attr in adj_loc_list]
                                except:
                                    import ipdb; ipdb.set_trace()
                                #if state.location.viewpointId == goalViewpointId:
                                #    return 0  # do nothing
                                #path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
                                #nextViewpointId = path[1]
                                #for n_a, loc_attr in enumerate(adj_loc_list):
                                #    if loc_attr['nextViewpointId'] == nextViewpointId:

                                ob['teacher'] = self._shortest_path_action(state, adj_loc_list, next_place)

                    elif self.r4r_reward1:
                        if history_path: # the first time
                            t = len(history_path[i_batch])

                            current_place = ob["viewpoint"]
                            history_current_path = [current_place] if history_path is None else history_path[i_batch]+[current_place]
                            def when_last_you_were_on_the_way(reference,real):
                                for j,place in enumerate(real[::-1]):
                                    if place in reference:
                                        how_many_visited = real.count(place)
                                        how_many_should_visit = reference.count(place)
                                        you_are_at_n_th_visit_on_reference =  how_many_visited if how_many_visited<how_many_should_visit else how_many_should_visit
                                        visited = [i for i,v in enumerate(reference) if v==place]
                                        timestep_on_reference = visited[you_are_at_n_th_visit_on_reference-1]
                                        #return i,place,reference.index(place)
                                        leap_from_reference = j
                                        return leap_from_reference,place,timestep_on_reference
                                import ipdb; ipdb.set_trace()
                                return len(real),None,-1
                            leap_from_reference,_,timestep_on_reference = when_last_you_were_on_the_way(path,history_current_path)
                            if timestep_on_reference<0:
                                import ipdb; ipdb.set_trace()
                            if leap_from_reference==0:
                                if len(item['path'])==timestep_on_reference+1:
                                    next_goal = item['path'][-1]
                                else:
                                    try:
                                        next_goal = item['path'][timestep_on_reference+1]
                                    except:
                                        import ipdb; ipdb.set_trace()
                            else:
                                allowed_skip = leap_from_reference+1
                                candidate_to_move = []
                                allowed_to_move = timestep_on_reference+allowed_skip+1 if timestep_on_reference+allowed_skip+1 < len(path) else len(path)
                                for i in range(timestep_on_reference,allowed_to_move):
                                    candidate_to_move.append(path[i])
                                try:
                                    next_goal = min(candidate_to_move,key=lambda x: self.distances[item['scan']][current_place][x])
                                except:
                                    import ipdb; ipdb.set_trace()
                            ob['teacher'] = self._shortest_path_action(state, adj_loc_list, next_goal)
                            #if i_batch==0:
                            #    import ipdb; ipdb.set_trace()
                        else:
                            ob['teacher'] = self._shortest_path_action(state, adj_loc_list, path[1])

                    elif self.r4r_reward1 or self.r4r_reward2:
                        if history_path: # the first time
                            t = len(history_path[i_batch])

                            current_place = ob["viewpoint"]
                            history_current_path = [current_place] if history_path is None else history_path[i_batch]+[current_place]
                            def when_last_you_were_on_the_way(reference,real):
                                for j,place in enumerate(real[::-1]):
                                    if place in reference:
                                        how_many_visited = real.count(place)
                                        how_many_should_visit = reference.count(place)
                                        you_are_at_n_th_visit_on_reference =  how_many_visited if how_many_visited<how_many_should_visit else how_many_should_visit
                                        visited = [i for i,v in enumerate(reference) if v==place]
                                        timestep_on_reference = visited[you_are_at_n_th_visit_on_reference-1]
                                        #return i,place,reference.index(place)
                                        leap_from_reference = j
                                        return leap_from_reference,place,timestep_on_reference
                                import ipdb; ipdb.set_trace()
                                return len(real),None,-1
                            leap_from_reference,_,timestep_on_reference = when_last_you_were_on_the_way(path,history_current_path)
                            if timestep_on_reference<0:
                                import ipdb; ipdb.set_trace()
                            if leap_from_reference==0:
                                if len(item['path'])==timestep_on_reference+1:
                                    next_goal = item['path'][-1]
                                else:
                                    try:
                                        next_goal = item['path'][timestep_on_reference+1]
                                    except:
                                        import ipdb; ipdb.set_trace()
                            else:
                                allowed_skip = leap_from_reference+1
                                candidate_to_move = []
                                allowed_to_move = timestep_on_reference+allowed_skip+1 if timestep_on_reference+allowed_skip+1 < len(path) else len(path)
                                if self.r4r_reward1:
                                    for i in range(timestep_on_reference,allowed_to_move):
                                        candidate_to_move.append(path[i])
                                    try:
                                        next_goal = min(candidate_to_move,key=lambda x: self.distances[item['scan']][current_place][x])
                                    except:
                                        import ipdb; ipdb.set_trace()
                                elif self.r4r_reward2:
                                    cumurative_distance = 0
                                    cumurative_distances = [self.distances[item['scan']][current_place][path[allowed_to_move-1]]]
                                    candidate_to_move.append(path[allowed_to_move-1])
                                    for i in range(allowed_to_move-2,timestep_on_reference,-1):
                                        cumurative_distance += self.distances[item['scan']][path[i]][path[i+1]]
                                        cumurative_distances.append(cumurative_distance + self.distances[item['scan']][current_place][path[i]])
                                        candidate_to_move.append(path[i])
                                    try:
                                        goal_th = timestep_on_reference + np.argmin(cumurative_distances[::-1])
                                        next_goal = path[goal_th]
                                    except:
                                        import ipdb; ipdb.set_trace()
                            ob['teacher'] = self._shortest_path_action(state, adj_loc_list, next_goal)
                            #if i_batch==0:
                            #    import ipdb; ipdb.set_trace()
                        else:
                            ob['teacher'] = self._shortest_path_action(state, adj_loc_list, path[1])

                    else:
                        ob['teacher'] = self._shortest_path_action(state, adj_loc_list, path[-1])

                if 'instr_encoding' in item:
                    ob['instr_encoding'] = item['instr_encoding']
                if 'instr_encoding_bert' in item:
                    ob['instr_encoding_bert'] = item['instr_encoding_bert']
                if 'instr_length' in item:
                    ob['instr_length'] = item['instr_length']
                obs_batch.append(ob)
                #import ipdb; ipdb.set_trace()
            if beamed:
                obs.append(obs_batch)
            else:
                assert len(obs_batch) == 1
                obs.append(obs_batch[0])

        #end_time = time.time()
        #print("get obs in {} seconds".format(end_time - start_time))
        return obs

    def get_starting_world_states(self, instance_list, beamed=False):
        scanIds = [item['scan'] for item in instance_list]
        viewpointIds = [item['path'][0] for item in instance_list]
        headings = [item['heading'] for item in instance_list]
        return self.env.newEpisodes(scanIds, viewpointIds, headings, beamed=beamed)

    def reset(self, sort=False, beamed=False, load_next_minibatch=True):
        ''' Load a new minibatch / episodes. '''
        if load_next_minibatch:
            self._next_minibatch(sort)
        assert len(self.batch) == self.batch_size, (len(self.batch), self.batch_size)
        return self.get_starting_world_states(self.batch, beamed=beamed)

    def step(self, world_states, actions, last_obs, beamed=False):
        ''' Take action (same interface as makeActions) '''
        return self.env.makeActions(world_states, actions, last_obs, beamed=beamed)

    #def shortest_paths_to_goals(self, starting_world_states, max_steps):
    def referenced_paths_to_goals(self, starting_world_states, max_steps):
        world_states = starting_world_states
        obs = self.observe(world_states, is_train=True)
        batch_size = len(obs)
        history_path = [[ob['viewpoint']] for ob in obs] # required when follow_detailed_path

        all_obs = []
        all_actions = []
        for ob in obs:
            all_obs.append([ob])
            all_actions.append([])

        ended = np.array([False] * len(obs))
        for t in range(max_steps):
            actions = [ob['teacher'] for ob in obs]
            world_states = self.step(world_states, actions, obs)
            assert len(world_states)==len(history_path), (len(world_states)==len(history_path))
            obs = self.observe(world_states, is_train=True, history_path=history_path)
            for i in range(batch_size):
                if not ended[i]:
                    history_path[i].append(obs[i]['viewpoint'])
            for i,ob in enumerate(obs):
                if not ended[i]:
                    all_obs[i].append(ob)
            for i,a in enumerate(actions):
                if not ended[i]:
                    all_actions[i].append(a)
                    if a == 0:
                        ended[i] = True
            if ended.all():
                break
        # print("len of all_actions",[len(_) for _ in all_actions])
        return all_obs, all_actions

    def gold_obs_actions_and_instructions(self, max_steps, load_next_minibatch=True,
                                          bert=False, r4r_follow_detailed_path=False):
        starting_world_states = self.reset(load_next_minibatch=load_next_minibatch)
        path_obs, path_actions = self.referenced_paths_to_goals(starting_world_states, max_steps)
        for i, (obs, actions) in enumerate(zip(path_obs, path_actions)): # kurita
            # don't include the last state, which should result after the stop action
            assert len(obs) == len(actions) + 1 , "%d %d %d"%(i, len(obs) , len(actions) + 1)
        if bert:
            encoded_instructions = [obs[0]['instr_encoding_bert'] for obs in path_obs]
        else:
            encoded_instructions = [obs[0]['instr_encoding'] for obs in path_obs]
        return path_obs, path_actions, encoded_instructions
