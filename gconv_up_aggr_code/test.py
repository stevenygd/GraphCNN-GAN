#!/home/valsesia/tensorflow-python2.7/bin/python
import os
import os.path as osp
import numpy as np
import shutil
import sys

from config import Config
from in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, load_all_point_clouds_under_folder
from gan import GAN
from general_utils import *
from PIL import Image
import tqdm
import scipy.io as sio

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--class_name', default='', help='Shapenet class')
parser.add_argument('--render_dir', default='', help='Renders directory')
parser.add_argument('--N_test', default=1000, help='NUmber of shapes')
# parser.add_argument('--N_test', default=100, help='NUmber of shapes')
parser.add_argument('--save_dir', default='', help='Trained model directory')
param = parser.parse_args()


# import config
config = Config()
config.render_dir = param.render_dir
config.save_dir = param.save_dir

#class_name = raw_input('Give me the class name (e.g. "chair"): ').lower()
class_name = param.class_name

model = GAN(config)
model.do_variables_init()
model.restore_model(config.save_dir+'model.ckpt')


# testing
all_shapes = []
ttl_cnt = 0
# for test_no in range(args.N_test):
# while True:
for _ in tqdm.tqdm(range(0, param.N_test, config.batch_size)):
    noise = np.random.normal(size=[config.batch_size, config.z_size], scale=0.2)
    pc_gen = model.generate(noise)
    # sio.savemat('%srender.mat' % (config.render_dir,),{'X_hat':pc_gen})
    all_shapes.append(pc_gen)
all_shapes = np.concatenate(all_shapes)

T_yz = np.array([[1,0,0],[0,0,1],[0,1,0]])
T_xz = np.array([[0,0,1],[0,1,0],[1,0,0]])
T_zinv = np.array([[1,0,0],[0,1,0],[0,0,-1]])
T = np.matmul(np.matmul(T_yz, T_xz), T_zinv)
gen = np.matmul(pc_gen, T)
np.save("%s/pc_gen.npy" % config.render_dir, pc_gen)
np.save("%s/gen.npy" % config.render_dir, gen)
