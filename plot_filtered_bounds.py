# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 12:44:26 2023

@author: ADIL003
"""

# %% Imports

from pathlib import Path
import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import rasterio.warp
from rasterio.windows import Window
import pandas as pd
#import geowombat as gw

import scipy
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_dilation

import skimage
from skimage import morphology
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from sklearn.preprocessing import minmax_scale
from sklearn.neighbors import NearestNeighbors

from enum import Enum
from tqdm import tqdm
import gc
import os

import uuid
from adjustText import adjust_text
import itertools

# %%

os.chdir('E:/Tree-Classification')

dirpaths = [
    R'E:/Tree-Classification/casuarina//93deg',
    R'E:/Tree-Classification/casuarina//183deg',
    R'E:/Tree-Classification/chestnut//10May2021',
    R'E:/Tree-Classification/chestnut//18Dec2020'
]
dirpaths = [Path(i) for i in dirpaths]

# %% Load each image and its corresponding bounds

images = []
for dirpath in dirpaths:
    if (adjusted := dirpath / 'adjusted.tif').exists():
        image = rio.open(adjusted)
        unadjusted = None
    else:
        image = rio.open(unadjusted := dirpath / 'result.tif')
        adjusted = None
    filtered_bounds = pd.read_csv(bounds := dirpath / '{}_filtered_bounds.csv'.format(dirpath.name))
    print('Loaded {}'.format(adjusted or unadjusted))
    print('Loaded {}'.format(bounds))
    images.append((image, adjusted or unadjusted, filtered_bounds, bounds))

# %% Draw the bounds for each image
for image, image_path, bounds, bounds_path in images:
    px_height, px_width = image.height, image.width
    dpi = 100
    in_height, in_width = px_height/dpi, px_width/dpi

    fig, ax = plt.subplots(figsize=(in_width, in_height))
    img = image.read([1, 2, 3]).transpose(1, 2, 0)
    idx = (img[...,:3] == np.array((0,0,0))).all(axis=-1)
    img[idx] = 127
    plt.imshow(img, interpolation='none')
    
    texts = []
    for idx, props in bounds.iterrows():
        minr, minc, maxr, maxc = props[['minr', 'minc', 'maxr', 'maxc']]
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-', color='white', linewidth=10)
        texts.append(ax.text(minc+5, maxr-5, '{}: {}'.format(str(idx), props['uuid']), fontsize=20))
    #adjust_text(texts)

    plt.savefig(bounds_path.parent / 'bboxes.jpg')

# %% Coregister the images
for pair in list(itertools.pairwise(images))[::2]:
    image1, image_path1, bounds1, bounds_path1 = pair[0]
    image2, image_path2, bounds2, bounds_path2 = pair[1]
    
    image1_bounds = image1.bounds
    image2_bounds = image2.bounds
    
    union_bounds = rio.coords.BoundingBox(min(image1_bounds[0], image2_bounds[0]),
                                          min(image1_bounds[1], image2_bounds[1]),
                                          max(image1_bounds[2], image2_bounds[2]),
                                          max(image1_bounds[3], image2_bounds[3]))
    
    union_px_topleft_image1 = image1.index(union_bounds[0], union_bounds[3])
    union_px_botright_image1 = image1.index(union_bounds[2], union_bounds[1])
    union_px_topleft_image2 = image2.index(union_bounds[0], union_bounds[3])
    union_px_botright_image2 = image2.index(union_bounds[2], union_bounds[1])

    window_im1 = Window(col_off=union_px_topleft_image1[1],
                        row_off=union_px_topleft_image1[0],
                        width=union_px_botright_image1[1]-union_px_topleft_image1[1],
                        height=union_px_botright_image1[0]-union_px_topleft_image1[0])
    window_im2 = Window(col_off=union_px_topleft_image2[1],
                        row_off=union_px_topleft_image2[0],
                        width=union_px_botright_image2[1]-union_px_topleft_image2[1],
                        height=union_px_botright_image2[0]-union_px_topleft_image2[0])
    
    im1_lined_up = image1.read([1, 2, 3], window=window_im1, boundless=True)
    im2_lined_up = image2.read([1, 2, 3], window=window_im2, boundless=True)
    
    with rasterio.open(
            str(bounds_path1.parent.parent / 'superimposed.tif'),
            'w',
            driver='GTiff',
            height=im1_lined_up.shape[1],
            width=im1_lined_up.shape[2],
            count=2,
            dtype=im1_lined_up.dtype,
            crs=image1.crs,
            transform=rio.windows.transform(window_im1, image1.transform),
    ) as dst:
        dst.write(im1_lined_up.mean(axis=0), 1)
        dst.write(im2_lined_up.mean(axis=0), 2)
        
# %% Now, get the nearest neighbours 
for pair in list(itertools.pairwise(images))[::2]:
    image1, image_path1, bounds1, bounds_path1 = pair[0]
    image2, image_path2, bounds2, bounds_path2 = pair[1]
    
    image1_bounds = image1.bounds
    image2_bounds = image2.bounds
    
    union_bounds = rio.coords.BoundingBox(min(image1_bounds[0], image2_bounds[0]),
                                          min(image1_bounds[1], image2_bounds[1]),
                                          max(image1_bounds[2], image2_bounds[2]),
                                          max(image1_bounds[3], image2_bounds[3]))
    
    union_px_topleft_image1 = image1.index(union_bounds[0], union_bounds[3])
    union_px_botright_image1 = image1.index(union_bounds[2], union_bounds[1])
    union_px_topleft_image2 = image2.index(union_bounds[0], union_bounds[3])
    union_px_botright_image2 = image2.index(union_bounds[2], union_bounds[1])

    window_im1 = Window(col_off=union_px_topleft_image1[1],
                        row_off=union_px_topleft_image1[0],
                        width=union_px_botright_image1[1]-union_px_topleft_image1[1],
                        height=union_px_botright_image1[0]-union_px_topleft_image1[0])
    window_im2 = Window(col_off=union_px_topleft_image2[1],
                        row_off=union_px_topleft_image2[0],
                        width=union_px_botright_image2[1]-union_px_topleft_image2[1],
                        height=union_px_botright_image2[0]-union_px_topleft_image2[0])
    
    bounds1[['maxr_fixed', 'minr_fixed']] = bounds1[['maxr', 'minr']] - window_im1.row_off
    bounds1[['maxc_fixed', 'minc_fixed']] = bounds1[['maxc', 'minc']] - window_im1.col_off
    bounds2[['maxr_fixed', 'minr_fixed']] = bounds2[['maxr', 'minr']] - window_im2.row_off
    bounds2[['maxc_fixed', 'minc_fixed']] = bounds2[['maxc', 'minc']] - window_im2.col_off

    for bounds in [bounds1, bounds2]:
        bounds['center_r'] = (bounds['minr_fixed'] + bounds['maxr_fixed']) / 2
        bounds['center_c'] = (bounds['minc_fixed'] + bounds['maxc_fixed']) / 2
        bounds['row_rad'] = (bounds['maxr_fixed'] - bounds['minr_fixed']) / 2
        bounds['col_rad'] = (bounds['maxc_fixed'] - bounds['minc_fixed']) / 2
        bounds['rad'] = bounds[['row_rad', 'col_rad']].min(axis=1)
    
    new = bounds1
    old = bounds2
    
    nn = NearestNeighbors()
    nn.fit(X=old[['center_r', 'center_c']])
    nn_out = nn.kneighbors(new[['center_r', 'center_c']], 1)
    dists, preds = nn_out[0].squeeze(), nn_out[1].squeeze()
    
    matched = pd.concat((new['uuid'].reset_index(drop=True),
                         old['uuid'].loc[preds].reset_index(drop=True)),
                        axis=1)\
        .reset_index(drop=True)
    matched.columns = [bounds_path1.parent.name, bounds_path2.parent.name]
    matched['distance_px'] = dists
    
    preds_filtered = preds[dists < new['rad']]
    dists_filtered = dists[dists < new['rad']]

    temp = pd.Series(dists < new['rad'])
    temp.index = new.index
    new_uuids_filtered = new['uuid'][temp]

    matched_filtered = pd.concat((new_uuids_filtered.reset_index(drop=True),
                         old['uuid'].loc[preds_filtered].reset_index(drop=True)),
                        axis=1)\
        .reset_index(drop=True)
    matched_filtered.columns = [bounds_path1.parent.name, bounds_path2.parent.name]
    matched_filtered['distance_px'] = dists_filtered

    matched.to_csv(bounds_path1.parent.parent / '{}_{}_{}_bounding_boxes_match.csv'
                   .format(bounds_path1.parent.parent.name, bounds_path1.parent.name, bounds_path2.parent.name))
    matched_filtered.to_csv(bounds_path1.parent.parent / '{}_{}_{}_bounding_boxes_match_filtered.csv'
                   .format(bounds_path1.parent.parent.name, bounds_path1.parent.name, bounds_path2.parent.name))


# %% Redraw the bounds with the nearest-neighbours visualised
for pair in list(itertools.pairwise(images))[::2]:
    image1, image_path1, bounds1, bounds_path1 = pair[0]
    image2, image_path2, bounds2, bounds_path2 = pair[1]
    
    with rasterio.open(
            str(bounds_path1.parent.parent / 'superimposed.tif'),
            'r') as dst:
        px_height, px_width = dst.height, dst.width
        dpi = 100
        in_height, in_width = px_height/dpi, px_width/dpi

        fig, ax = plt.subplots(figsize=(in_width, in_height))
        
        image = dst.read([1, 2])
        image = np.stack((image[0], image[1], np.zeros_like(image[0]))).transpose(1, 2, 0)
        
        idx = (image[...,:3] == np.array((0,0,0))).all(axis=-1)
        image[idx] = 127
        
        plt.imshow(image, interpolation='none')

        bounds1['origin'] = bounds_path1.parent.name
        bounds2['origin'] = bounds_path2.parent.name
        

        matched_filtered = pd.read_csv(bounds_path1.parent.parent / '{}_{}_{}_bounding_boxes_match_filtered.csv'
                                       .format(bounds_path1.parent.parent.name, bounds_path1.parent.name,
                                               bounds_path2.parent.name),
                                       index_col = 0)

        bounds1['in_matched_filtered'] = bounds1['uuid'].isin(matched_filtered[bounds_path1.parent.name])
        bounds2['in_matched_filtered'] = True
        
        bounds = pd.concat((bounds1, bounds2))

        texts = []
        for idx, props in bounds.iterrows():
            minr, minc, maxr, maxc = props[['minr_fixed', 'minc_fixed', 'maxr_fixed', 'maxc_fixed']]
            
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            ax.plot(bx, by, '-',
                    color='blue' if props['origin'] == bounds_path2.parent.name else \
                                ('red' if props['in_matched_filtered'] else 'white'),
                    linewidth=10)
            texts.append(ax.text(minc+5, maxr-5, '{}: {}'.format(str(idx), props['uuid']), fontsize=12))
        #adjust_text(texts)
        
        matched_filtered = matched_filtered.merge(bounds1[['uuid', 'center_r', 'center_c']],
                                                  left_on=bounds_path1.parent.name,
                                                  right_on='uuid',
                                                  suffixes=('_m', '_n'))
        matched_filtered = matched_filtered.merge(bounds2[['uuid', 'center_r', 'center_c']],
                                                  left_on=bounds_path2.parent.name,
                                                  right_on='uuid',
                                                  suffixes=('_n', '_o'))
        
        for idx, row in matched_filtered.iterrows():
            r_n, c_n = row[['center_r_n', 'center_c_n']]
            r_o, c_o = row[['center_r_o', 'center_c_o']]
            ax.arrow(c_n, r_n, c_o-c_n, r_o-r_n, width=10, color='blue')
                
        print("plotted bounds for {}".format(bounds_path1.parent.parent))

        plt.savefig(str(bounds_path1.parent.parent / 'bboxes.jpg'))

