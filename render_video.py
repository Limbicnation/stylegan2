# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import time
import re
import bisect
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import scipy.ndimage
import scipy.misc
import imageio

import pickle
import training.misc as misc
import dnnlib.tflib.tfutil as tfutil

from PIL import Image

def random_latents(num_latents, G, random_state=None):
    if random_state is not None:
        return random_state.randn(num_latents, *G.input_shape[1:]).astype(np.float32)
    else:
        return np.random.randn(num_latents, *G.input_shape[1:]).astype(np.float32)

def load_pkl(network_pkl):
    with open(network_pkl, 'rb') as file:
        return pickle.load(file, encoding='latin1')

def get_id_string_for_network_pkl(network_pkl):
    p = network_pkl.replace('.pkl', '').replace('\\', '/').split('/')
    longname = '-'.join(p[max(len(p) - 2, 0):])
    return '-'.join(longname.split('-')[2:])

#----------------------------------------------------------------------------
# Generate MP4 video of random interpolations using a previously trained network.
# To run, uncomment the appropriate line in config.py and launch train.py.

def generate_interpolation_video(network_pkl = None, grid_size=[1,1], png_sequence=False, image_shrink=1, image_zoom=1, duration_sec=60.0, smoothing_sec=1.0, filename=None, mp4_fps=30, mp4_codec='libx265', mp4_bitrate='16M', random_seed=1000, minibatch_size=8):
    
    if network_pkl == None:
        print('Please enter pkl path')
    num_frames = int(np.rint(duration_sec * mp4_fps))
    random_state = np.random.RandomState(random_seed)
    if filename is None:
        filename = get_id_string_for_network_pkl(network_pkl) + '-seed-' + str(random_seed) + '.mp4'

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = load_pkl(network_pkl)

    print('Generating latent vectors...')
    shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:] # [frame, image, channel, component]
    print(shape)
    print(len(shape))
    all_latents = random_state.randn(*shape).astype(np.float32)
    all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape), mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))
    print(all_latents[0].shape)


    # Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latents = all_latents[frame_idx]
        labels = np.zeros([latents.shape[0], 0], np.float32)
        images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=1, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8, truncation_psi=1, randomize_noise=False)
        grid = misc.create_image_grid(images, grid_size).transpose(1, 2, 0) # HWC
        if image_zoom > 1:
            grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2) # grayscale => RGB
        return grid

    if png_sequence: 
        result_subdir = "results/videos/" + filename
        os.makedirs(result_subdir)
        for png_idx in range(num_frames):
            print('Generating png %d / %d...' % (png_idx, num_frames))
            latents = latents = all_latents[png_idx]
            labels = np.zeros([latents.shape[0], 0], np.float32)
            images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=1, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8, truncation_psi=1, randomize_noise=False)
            misc.save_image_grid(images, os.path.join(result_subdir, '%06d.png' % (png_idx)), [0,255], grid_size)
    else:
        # Generate video.
        import moviepy.editor # pip install moviepy
        result_subdir = "results/videos"
        os.makedirs(result_subdir)
        moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(os.path.join(result_subdir, filename + ".mp4"), fps=mp4_fps, codec='mpeg4', bitrate=mp4_bitrate)


if __name__ == "__main__":
    import datetime
    import time
    print(datetime.datetime.now(), int(time.time()))
    np.random.seed(int(time.time()))
    tfutil.init_tf()

    generate_interpolation_video("./results/00042-stylegan2-covidfaces1024-1gpu-config-f/network-snapshot-003274.pkl", grid_size=[1,1], png_sequence=True, random_seed=int(time.time()), mp4_fps=60, duration_sec=60.0, smoothing_sec=2)

    print('Exiting...')
    print(datetime.datetime.now())
