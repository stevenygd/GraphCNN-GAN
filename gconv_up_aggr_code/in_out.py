import six
import warnings
import numpy as np
import os
import os.path as osp
import re
from six.moves import cPickle
from multiprocessing import Pool

# from . general_utils import rand_rotation_matrix
# from .. external.python_plyfile.plyfile import PlyElement, PlyData
from general_utils import rand_rotation_matrix
from plyfile import PlyElement, PlyData

snc_synth_id_to_category = {
    '02691156': 'airplane',  '02773838': 'bag',        '02801938': 'basket',
    '02808440': 'bathtub',   '02818832': 'bed',        '02828884': 'bench',
    '02834778': 'bicycle',   '02843684': 'birdhouse',  '02871439': 'bookshelf',
    '02876657': 'bottle',    '02880940': 'bowl',       '02924116': 'bus',
    '02933112': 'cabinet',   '02747177': 'can',        '02942699': 'camera',
    '02954340': 'cap',       '02958343': 'car',        '03001627': 'chair',
    '03046257': 'clock',     '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table',     '04401088': 'telephone',  '02946921': 'tin_can',
    '04460130': 'tower',     '04468005': 'train',      '03085013': 'keyboard',
    '03261776': 'earphone',  '03325088': 'faucet',     '03337140': 'file',
    '03467517': 'guitar',    '03513137': 'helmet',     '03593526': 'jar',
    '03624134': 'knife',     '03636649': 'lamp',       '03642806': 'laptop',
    '03691459': 'speaker',   '03710193': 'mailbox',    '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano',     '03938244': 'pillow',     '03948459': 'pistol',
    '03991062': 'pot',       '04004475': 'printer',    '04074963': 'remote_control',
    '04090263': 'rifle',     '04099429': 'rocket',     '04225987': 'skateboard',
    '04256520': 'sofa',      '04330267': 'stove',      '04530566': 'vessel',
    '04554684': 'washer',    '02858304': 'boat',       '02992529': 'cellphone'
}


def snc_category_to_synth_id():
    d = snc_synth_id_to_category
    inv_map = {v: k for k, v in six.iteritems(d)}
    return inv_map


def create_dir(dir_path):
    ''' Creates a directory (or nested directories) if they don't exist.
    '''
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def pickle_data(file_name, *args):
    '''Using (c)Pickle to save multiple python objects in a single file.
    '''
    myFile = open(file_name, 'wb')
    cPickle.dump(len(args), myFile, protocol=2)
    for item in args:
        cPickle.dump(item, myFile, protocol=2)
    myFile.close()


def unpickle_data(file_name):
    '''Restore data previously saved with pickle_data().
    '''
    inFile = open(file_name, 'rb')
    size = cPickle.load(inFile)
    for _ in xrange(size):
        yield cPickle.load(inFile)
    inFile.close()


def files_in_subdirs(top_dir, search_pattern):
    regex = re.compile(search_pattern)
    for path, _, files in os.walk(top_dir):
        for name in files:
            full_name = osp.join(path, name)
            if regex.search(full_name):
                yield full_name


def load_ply(file_name, with_faces=False, with_color=False):
    ply_data = PlyData.read(file_name)
    points = ply_data['vertex']
    points = np.vstack([points['x'], points['y'], points['z']]).T
    ret_val = [points]

    if with_faces:
        faces = np.vstack(ply_data['face']['vertex_indices'])
        ret_val.append(faces)

    if with_color:
        r = np.vstack(ply_data['vertex']['red'])
        g = np.vstack(ply_data['vertex']['green'])
        b = np.vstack(ply_data['vertex']['blue'])
        color = np.hstack((r, g, b))
        ret_val.append(color)

    if len(ret_val) == 1:  # Unwrap the list
        ret_val = ret_val[0]

    return ret_val


def pc_loader(f_name):
    ''' loads a point-cloud saved under ShapeNet's "standar" folder scheme:
    i.e. /syn_id/model_name.ply
    '''
    tokens = f_name.split('/')
    model_id = tokens[-1].split('.')[0]
    synet_id = tokens[-2]
    return load_ply(f_name), model_id, synet_id


def pc_npy_loader(f_name):
    ''' loads a point-cloud saved under ShapeNet's "standar" folder scheme:
    i.e. /syn_id/model_name.npy
    '''
    tokens = f_name.split('/')
    model_id = tokens[-1].split('.')[0]
    synet_id = tokens[-2]
    out = np.load(f_name)
    return out, model_id, synet_id


def load_all_point_clouds_under_folder(
        top_dir, n_threads=20, file_ending='.ply', max_num_points=None, verbose=False,
        normalize=False, rotation_axis=None, file_names=None):
    if file_names is None:
        file_names = [f for f in files_in_subdirs(top_dir, file_ending)]
    if file_ending == '.ply':
        pclouds, model_ids, syn_ids = load_point_clouds_from_filenames(
                file_names, n_threads, loader=pc_loader, verbose=verbose)
        return PointCloudDataSet(
                pclouds, labels=syn_ids + '_' + model_ids, init_shuffle=False, max_num_points=max_num_points,
                rotation_axis=rotation_axis)
    elif file_ending == '.npy':
        pclouds, model_ids, syn_ids = load_point_clouds_from_numpy_filenames(
                file_names, n_threads, loader=pc_npy_loader, verbose=verbose, normalize=normalize)
        return PointCloudDataSet(
                pclouds, labels=syn_ids + '_' + model_ids, init_shuffle=False, max_num_points=max_num_points,
                rotation_axis=rotation_axis)


def load_all_point_clouds_under_folders(
        top_dirs, n_threads=20, file_ending='.ply', max_num_points=None, verbose=False,
        normalize=False, rotation_axis=None, file_names=None):
    if file_names is None:
        file_names = [f for top_dir in top_dirs for f in files_in_subdirs(top_dir, file_ending)]
    if file_ending == '.ply':
        pclouds, model_ids, syn_ids = load_point_clouds_from_filenames(
                file_names, n_threads, loader=pc_loader, verbose=verbose)
        return PointCloudDataSet(
                pclouds, labels=syn_ids + '_' + model_ids,
                init_shuffle=False, max_num_points=max_num_points,
                rotation_axis=rotation_axis)
    elif file_ending == '.npy':
        pclouds, model_ids, syn_ids = load_point_clouds_from_numpy_filenames(
                file_names, n_threads, loader=pc_npy_loader, verbose=verbose, normalize=normalize)
        return PointCloudDataSet(
                pclouds, labels=syn_ids + '_' + model_ids,
                init_shuffle=False, max_num_points=max_num_points,
                rotation_axis=rotation_axis)



def load_point_clouds_from_numpy_filenames(file_names, n_threads, loader, verbose=False, normalize=False):
    pc = loader(file_names[0])[0]
    pclouds = np.empty([len(file_names), pc.shape[0], pc.shape[1]], dtype=np.float32)
    model_names = np.empty([len(file_names)], dtype=object)
    class_ids = np.empty([len(file_names)], dtype=object)
    pool = Pool(n_threads)

    for i, data in enumerate(pool.imap(loader, file_names)):
        pclouds[i, :, :], model_names[i], class_ids[i] = data

    pool.close()
    pool.join()

    if normalize:
        B, N = pclouds.shape[0], pclouds.shape[1]
        pclouds_mean = pclouds.mean(axis=1).reshape(-1, 1, 3)
        pclouds_std = pclouds.reshape(B, N*3).std(axis=1).reshape(-1, 1, 1)
        pclouds = (pclouds  - pclouds_mean) / pclouds_std

    if len(np.unique(model_names)) != len(pclouds):
        warnings.warn('Point clouds with the same model name were loaded.')

    if verbose:
        print('{0} pclouds were loaded. They belong in {1} shape-classes.'.format(len(pclouds), len(np.unique(class_ids))))

    return pclouds, model_names, class_ids



def load_point_clouds_from_filenames(file_names, n_threads, loader, verbose=False):
    pc = loader(file_names[0])[0]
    pclouds = np.empty([len(file_names), pc.shape[0], pc.shape[1]], dtype=np.float32)
    model_names = np.empty([len(file_names)], dtype=object)
    class_ids = np.empty([len(file_names)], dtype=object)
    pool = Pool(n_threads)

    for i, data in enumerate(pool.imap(loader, file_names)):
        pclouds[i, :, :], model_names[i], class_ids[i] = data

    pool.close()
    pool.join()

    if len(np.unique(model_names)) != len(pclouds):
        warnings.warn('Point clouds with the same model name were loaded.')

    if verbose:
        print('{0} pclouds were loaded. They belong in {1} shape-classes.'.format(len(pclouds), len(np.unique(class_ids))))

    return pclouds, model_names, class_ids


class PointCloudDataSet(object):
    '''
    See https://github.com/tensorflow/tensorflow/blob/a5d8217c4ed90041bea2616c14a8ddcf11ec8c03/tensorflow/examples/tutorials/mnist/input_data.py
    '''

    def __init__(self, point_clouds, noise=None, labels=None, copy=True, init_shuffle=True, max_num_points=None,
                 rotation_axis=None):
        '''Construct a DataSet.
        Args:
            init_shuffle, shuffle data before first epoch has been reached.
        Output:
            original_pclouds, labels, (None or Feed) # TODO Rename
        '''

        if max_num_points is not None:
            point_clouds = point_clouds[:,:max_num_points,:]
        self.num_examples = point_clouds.shape[0]
        self.n_points = point_clouds.shape[1]
        self.rotation_axis = rotation_axis

        if labels is not None:
            assert point_clouds.shape[0] == labels.shape[0], \
                    ('points.shape: %s labels.shape: %s' % (point_clouds.shape, labels.shape))
            if copy:
                self.labels = labels.copy()
            else:
                self.labels = labels

        else:
            self.labels = np.ones(self.num_examples, dtype=np.int8)

        if noise is not None:
            assert (type(noise) is np.ndarray)
            if copy:
                self.noisy_point_clouds = noise.copy()
            else:
                self.noisy_point_clouds = noise
        else:
            self.noisy_point_clouds = None

        if copy:
            self.point_clouds = point_clouds.copy()
        else:
            self.point_clouds = point_clouds

        self.epochs_completed = 0
        self._index_in_epoch = 0
        if init_shuffle:
            self.shuffle_data()


    def apply_random_rotation(self, pc, rot=None, theta=None):
        # import pdb; pdb.set_trace()

        B = pc.shape[0]
        if rot is not None and theta is not None:
            assert rot.shape[0] == B
            assert rot.shape[1] == 3
            assert rot.shape[2] == 3
            rot = rot
            theta = theta

        elif self.rotation_axis is None:
            zeros = np.zeros(B)
            ones  = np.ones(B)
            rot = np.stack([
                ones,   zeros,  zeros,
                zeros,  ones,   zeros,
                zeros,  zeros,  ones
            ]).T.reshape(B, 3, 3)
            theta = np.zeros(B)

            return pc, rot, theta
        else:
            theta = np.random.rand(B) * 2 * np.pi
            zeros = np.zeros(B)
            ones  = np.ones(B)
            cos   = np.cos(theta)
            sin   = np.sin(theta)
            if self.rotation_axis == 0:
                rot = np.stack([
                    cos,   -sin,  zeros,
                    sin,   cos,   zeros,
                    zeros, zeros, ones
                ]).T.reshape(B, 3, 3)
            elif self.rotation_axis == 1:
                rot = np.stack([
                    cos,   zeros, -sin,
                    zeros, ones,  zeros,
                    sin,   zeros, cos
                ]).T.reshape(B, 3, 3)
            elif self.rotation_axis == 2:
                rot = np.stack([
                    ones,  zeros, zeros,
                    zeros, cos,   -sin,
                    zeros, sin,   cos
                ]).T.reshape(B, 3, 3)
            else:
                raise Exception("Invalid rotation axis")

            # (B, N, 3) mul (B, 3, 3) -> (B, N, 3)
            pc_rotated = np.matmul(pc, rot)
            return pc_rotated, rot, theta


    def shuffle_data(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        self.point_clouds = self.point_clouds[perm]
        self.labels = self.labels[perm]
        if self.noisy_point_clouds is not None:
            self.noisy_point_clouds = self.noisy_point_clouds[perm]
        return self


    def next_batch(self, batch_size, seed=None):
        '''Return the next batch_size examples from this data set.
        '''
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            self.epochs_completed += 1  # Finished epoch.
            self.shuffle_data(seed)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch

        pcl = self.point_clouds[start:end][:, :2048, ...]
        pcl, rot, theta = self.apply_random_rotation(pcl, rot=None, theta=None)
        lbl = self.labels[start:end]

        if self.noisy_point_clouds is None:
            # return self.point_clouds[start:end], self.labels[start:end], None
            return pcl, lbl, None
        else:
            noisy_pcl = self.noisy_point_clouds[start:end]
            noisy_pcl = self.apply_random_rotation(noisy_pcl, rot=rot, theta=theta)
            # return self.point_clouds[start:end], self.labels[start:end], self.noisy_point_clouds[start:end]
            return pcl, lbl, noisy_pcl


    def full_epoch_data(self, shuffle=True, seed=None):
        '''Returns a copy of the examples of the entire data set (i.e. an epoch's data), shuffled.
        '''
        if shuffle and seed is not None:
            np.random.seed(seed)
        perm = np.arange(self.num_examples)  # Shuffle the data.
        if shuffle:
            np.random.shuffle(perm)
        pc = self.point_clouds[perm]
        pc, rot, theta = self.apply_random_rotation(pc, rot=None, theta=None)
        lb = self.labels[perm]
        ns = None
        if self.noisy_point_clouds is not None:
            ns = self.noisy_point_clouds[perm]
            ns = self.apply_random_rotation(ns, rot=rot, theta=theta)
        return pc, lb, ns


    def merge(self, other_data_set):
        self._index_in_epoch = 0
        self.epochs_completed = 0
        self.point_clouds = np.vstack((self.point_clouds, other_data_set.point_clouds))

        labels_1 = self.labels.reshape([self.num_examples, 1])  # TODO = move to init.
        labels_2 = other_data_set.labels.reshape([other_data_set.num_examples, 1])
        self.labels = np.vstack((labels_1, labels_2))
        self.labels = np.squeeze(self.labels)

        if self.noisy_point_clouds is not None:
            self.noisy_point_clouds = np.vstack((self.noisy_point_clouds, other_data_set.noisy_point_clouds))

        self.num_examples = self.point_clouds.shape[0]

        return self


