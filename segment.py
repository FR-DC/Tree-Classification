# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:05:11 2023

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

from enum import Enum
from tqdm import tqdm
import gc
import os

import uuid

# %% Data Ingest

# =============================================================================
# For reusability later one, we want to separate the format we store our 2D and 3D imagery data from the manner in which we read it in -- the same idea followed by FRModel. We will have separate classes/methods for representing the input data storage format and the in-memory format; the former will inherit from the latter.
#==============================================================================

class Bands(Enum):
    wr = 1
    wg = 2
    wb = 3
    nr = 4
    ng = 5
    nb = 6
    nir = 7
    red_edge = 8
    dsm = 9

class CapData():
    def __init__(self, ndarrs, labels):
        self.ndarrs = ndarrs
        self.labels = labels
    
    def get_band(self, band):
        return self.ndarrs[band]

    def get_bands(self, bands):
        return np.ma.stack([self.get_band(band) for band in bands],
                           axis=-1)
    
    def get_all_bands(self):
        return self.get_bands(self.labels)

class chestnut_input_dat(CapData):    
    input_files = [
        {'filename': 'result.tif', 'bands': [Bands.wr, Bands.wg, Bands.wb]},
        {'filename': 'result_Red.tif', 'bands': [Bands.nr]},
        {'filename': 'result_Green.tif', 'bands': [Bands.ng]},
        {'filename': 'result_Blue.tif', 'bands': [Bands.nb]},
        {'filename': 'result_NIR.tif', 'bands': [Bands.nir]},
        {'filename': 'result_RedEdge.tif', 'bands': [Bands.red_edge]},
        {'filename': 'dsm.tif', 'bands': [Bands.dsm]}
    ]
    
    def __init__(self, dirpath):
        dirpath = Path(dirpath)
        self.dirpath = dirpath
        self.labels = list(Bands)
        self.write_like = dirpath / chestnut_input_dat.input_files[0]['filename']
        read_bands = dict()
        for file in chestnut_input_dat.input_files:
            for band, idx in zip(file['bands'], 
                                 range(1, len(file['bands'])+1)):
                read_bands[band] = (dirpath / file['filename'], idx)
        self.ndarrs = read_bands
        self.datasets = dict()
        
    def get_band(self, band):
        if band == Bands.dsm:
            if not isinstance(self.ndarrs[Bands.nir], np.ndarray):
                self.get_band(Bands.nir)

        if not isinstance(self.ndarrs[band], np.ndarray):
            path, idx = self.ndarrs[band]
            with rio.open(path, mode='r+') as of:
                self.ndarrs[band] = of.read(idx)

                nodata = of.nodata
                # special case
                if nodata == None:
                    nodata = float(0)
                
                self.ndarrs[band] = np.ma.masked_equal(self.ndarrs[band], nodata)

                self.datasets[band] = of

            if band == Bands.dsm:
                dsm_reproj = np.empty_like(self.ndarrs[Bands.nir])
                rio.warp.reproject(
                    self.ndarrs[band],
                    dsm_reproj,
                    src_transform = self.datasets[Bands.dsm].transform,
                    src_crs = self.datasets[Bands.dsm].crs,
                    dst_transform = self.datasets[Bands.nir].transform,
                    dst_crs = self.datasets[Bands.nir].crs,
                    resampling  = rio.warp.Resampling.nearest)
                self.ndarrs[band] = dsm_reproj
                
        if band != Bands.nir:
            self.ndarrs[band] = self.ndarrs[band].astype('uint8')

        return self.ndarrs[band]
    
        
# %% Meaningless Segmentation (adapted from FRModel)
FIG_SIZE = 10
NIR_THRESHOLD = 90 / 256

BLOB_CONNECTIVITY = 2
BLOB_MIN_SIZE = 1000
TEXT_X = 0.5
TEXT_Y = 1.02

PEAKS_FOOTPRINT = 200
CANNY_THICKNESS = 5

def BIN_FILTER(inp: CapData):
    # noinspection PyTypeChecker
    return inp.get_band(Bands.nir) < NIR_THRESHOLD * (2 ** 14)

def meaningless_segmentation(inp: CapData,
                             bin_filter=BIN_FILTER,
                             blob_connectivity=BLOB_CONNECTIVITY,
                             blob_min_size=BLOB_MIN_SIZE,
                             peaks_footprint=PEAKS_FOOTPRINT,
                             canny_thickness=CANNY_THICKNESS):
    """ Runs the Meaningless Segmentation as depicted in the journal

    The output_dir will be automatically created if it doesn't exist.
    Default: "mnl/"

    :param inp: Input Frame2D, can be MaskedData
    :param bin_filter: A function that takes in Frame2D and returns a boolean mask
    :param blob_connectivity: Connectivity of morphology.remove_small_objects
    :param blob_min_size: Min Size of morphology.remove_small_objects
    :param peaks_footprint: Footprint of Local Peak Max
    :param canny_thickness: Thickness of Canny line
    :param output_dir: Output directory, will be created if doesn't exist
    :return: Dictionary of "frame": Frame2D, "peaks": np.ndarray
    """
    
    # ============ BINARIZATION ============
    print("Binarizing Image...", end=" ")

    fig, ax = plt.subplots(1, 3, figsize=(FIG_SIZE,
                                          FIG_SIZE // 2),
                           sharey=True)
    #produce binary array with mask preserved
    binary = np.ma.where(bin_filter(inp), 0, 1).squeeze()
    
    #turn mask into NaN for skimage
    print("Masked array...")
    mask = binary.mask
    binary = binary.filled(fill_value=0)
    
    # ============ BLOB REMOVAL ============
    print("Removing Small Blobs...", end=" ")
    ax[0].imshow(binary, cmap='gray')
    ax[0].text(TEXT_X, TEXT_Y, 'ORIGINAL',
               horizontalalignment='center', transform=ax[0].transAxes)
    binary = morphology.remove_small_objects(binary.astype(bool),
                                             min_size=blob_min_size,
                                             connectivity=blob_connectivity)
    ax[1].imshow(binary, cmap='gray')
    ax[1].text(TEXT_X, TEXT_Y, 'REMOVE MEANINGLESS',
               horizontalalignment='center', transform=ax[1].transAxes)
    binary = ~morphology.remove_small_objects(~binary,
                                              min_size=blob_min_size,
                                              connectivity=blob_connectivity)
    ax[2].imshow(binary, cmap='gray')
    ax[2].text(TEXT_X, TEXT_Y, 'PATCH MEANINGFUL',
               horizontalalignment='center', transform=ax[2].transAxes)
    fig.tight_layout()
    fig.savefig('blob_removal_path.jpg')

    print("Binarized.")

    print(f"Removed Blobs with size < {blob_min_size}, "
          f"connectivity = {blob_connectivity}.")


    # ============ DISTANCE ============
    print("Creating Distance Image...", end=" ")
    distances = distance_transform_edt(binary.astype(bool))

    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))

    i = ax.imshow(-distances, cmap='gray')
    fig: plt.Figure
    fig.colorbar(i, ax=ax)
    fig.tight_layout()
    fig.savefig('euclidean_distance_transform_path.jpg')

    # ============ PEAK FINDING ============
    print("Finding Peaks...", end=" ")
    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))

    peaks = peak_local_max(distances,
                           footprint=np.ones((peaks_footprint, peaks_footprint)),
                           exclude_border=0,
                           labels=binary)

    ax.imshow(-distances, cmap='gray')
    ax: plt.Axes
    ax.scatter(peaks[..., 1], peaks[..., 0], c='red', s=1)
    ax.text(x=TEXT_X, y=TEXT_Y, s=f"FOOTPRINT {peaks_footprint}", size=10,
            horizontalalignment='center', transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig('peaks_path.jpg')

    print(f"Found {peaks.shape[0]} peaks with Footprint {peaks_footprint}.")

    # ============ WATERSHED ============
    print("Running Watershed...", end=" ")
    markers = np.zeros(distances.shape, dtype=bool)
    markers[tuple(peaks.T)] = True
    markers, _ = ndi.label(markers)
    water = watershed(-distances, markers, mask=binary)

    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
    ax.imshow(water, cmap="magma")
    ax.scatter(peaks[..., 1], peaks[..., 0], c='red', s=1)

    fig.tight_layout()
    fig.savefig('watershed_path.jpg')

    print("Created Watershed Image.")

    # ============ CANNY EDGE ============
    print("Running Canny Edge Detection...", end=" ")
    canny = skimage.feature.canny(water.astype('float32'))
    fig, ax = plt.subplots(figsize=(FIG_SIZE*3, FIG_SIZE*3))
    ax.axis('off')
    ax.imshow(minmax_scale(inp.get_bands([Bands.nr, Bands.ng, Bands.nb]).reshape(-1, 3)).reshape(binary.shape + (3,)))
    ax.imshow(binary_dilation(canny, structure=np.ones((canny_thickness*3, canny_thickness*3))),
              cmap='gray', alpha=0.20)
    fig.savefig('canny_path.tif')
    
    print("Created Canny Edge Image.")
    
    labels = ["BINARY", "DISTANCE", "WATER", "CANNY"]
    ndarrs = dict(BINARY = binary,
                  DISTANCES = distances,
                  WATER = water,
                  CANNY = canny)
    for key, value in ndarrs.items():
        ndarrs[key] = np.ma.MaskedArray(data=value,
                                        mask=mask)
    cap_data_ = CapData(ndarrs, labels)

    return dict(cap_data=cap_data_, peaks=peaks)

# %% Actually implement our cropping-out.
def load(dirpath, class_):
    ch = class_(dirpath)
    mnls = meaningless_segmentation(ch)
    return ch, mnls

def label(ch, mnls):
    cmap = plt.get_cmap('nipy_spectral')
    cmap = mpl.colors.ListedColormap([cmap(i) for i in np.random.rand(256)])
    cmap.set_bad('black', 1.0)
    
    dilated = binary_dilation(mnls['cap_data'].get_band('CANNY').filled(),
                              structure=np.ones((CANNY_THICKNESS*3, CANNY_THICKNESS*3)))
    
    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
    ax.imshow(dilated, cmap='gray', alpha=1.0, interpolation='none')
    fig.savefig('edges_dilated.jpg')

    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))    
    dilated_mask_1 = np.copy(dilated)
    dilated_mask_1[ch.get_band(Bands.wr).mask] = 1
    dilated_mask_1[mnls['cap_data'].get_band('WATER') == 0] = 1
    ax.imshow(dilated_mask_1, cmap='gray', alpha=1.0, interpolation='none')
    fig.savefig('dilated_mask_1.jpg')
    
    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
    labelled = skimage.measure.label(dilated_mask_1, background=1)
    labelled = np.ma.MaskedArray(labelled,
                                 labelled == 0)
    plt.imshow(labelled, interpolation='none', cmap=cmap)
    plt.savefig('labelled.jpg')
    
    all_bands_masked_coregistered = [
        np.ma.MaskedArray(i,
                          labelled.mask).filled(fill_value=0)
        for i in [ch.get_band(j) for j in list(Bands)]
    ]
    
    with rio.open(ch.write_like, mode='r') as write_like:
        with rio.open('all_bands_masked_coregistered.tif',
                      mode='w',
                      driver="GTiff",
                      height=labelled.shape[0],
                      width=labelled.shape[1],
                      count=len(all_bands_masked_coregistered),
                      dtype='int16',
                      crs=write_like.crs,
                      transform=write_like.transform) as out:
            for i, idx in zip(all_bands_masked_coregistered, range(1, len(all_bands_masked_coregistered)+1)):
                out.write(i, idx)
    
    fig, ax = plt.subplots(figsize=(FIG_SIZE*4, FIG_SIZE*4))
    plt.imshow(labelled, interpolation='none', cmap=cmap)
    regions = skimage.measure.regionprops(labelled)
    for props, color_idx in zip(regions, range(len(regions))):
        y0, x0 = props.centroid
        #ax.plot(x0, y0, '.g', markersize=1)
        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-', color='white', linewidth=1)
        #ax.plot(bx, by, '-', color=cmap(color_idx % 256), linewidth=1)
    plt.savefig('labelled_bboxes.jpg')
    
    with rio.open(ch.write_like, mode='r') as write_like:
        canny = mnls['cap_data'].get_band('CANNY')
        water = mnls['cap_data'].get_band('WATER')
        with rio.open('labels.tif',
                      mode='w',
                      driver="GTiff",
                      height=canny.shape[0],
                      width=canny.shape[1],
                      count=4,
                      dtype='int16',
                      crs=write_like.crs,
                      transform=write_like.transform) as out:
            out.write(canny, 1)
            out.write(labelled, 2)
            out.write(water, 3)
    
    bboxes = []
    for props in regions:
        bboxes.append((uuid.uuid4(),) + props.bbox)
    bounds = pd.DataFrame(bboxes, columns=['uuid', 'minr', 'minc', 'maxr', 'maxc'])

    return ch, mnls, labelled, bounds


# %% Do the cropping out.

os.chdir('/home/pitter/Tree-Classification')

dirpaths = [
    '/home/pitter/Tree-Classification/chestnut//10May2021',
    '/home/pitter/Tree-Classification/chestnut//18Dec2020'
]

for dirpath in dirpaths:
    os.chdir(dirpath)
    load_out = load(dirpath, chestnut_input_dat)
    label_out = label(*load_out)
    label_out[3].to_csv('bounds.csv')
    del load_out
    del label_out
    gc.collect()

# %% Same but for casuarina

os.chdir(r"/home/pitter/Tree-Classification")
class casuarina_input_dat(chestnut_input_dat):        
    input_files = [
        {'filename': '{} result.tif', 'bands': [Bands.wr, Bands.wg, Bands.wb]},
        {'filename': 'result_Red.tif', 'bands': [Bands.nr]},
        {'filename': 'result_Green.tif', 'bands': [Bands.ng]},
        {'filename': 'result_Blue.tif', 'bands': [Bands.nb]},
        {'filename': 'result_NIR.tif', 'bands': [Bands.nir]},
        {'filename': 'result_RedEdge.tif', 'bands': [Bands.red_edge]},
        {'filename': '{} dsm.tif', 'bands': [Bands.dsm]}
    ]
    
    def __init__(self, dirpath):
        dirpath = Path(dirpath)

        self.input_files = copy.deepcopy(casuarina_input_dat.input_files)
        name = dirpath.name
        for dict_ in self.input_files:
            dict_['filename'] = dict_['filename'].format(name)

        self.dirpath = dirpath
        self.labels = list(Bands)
        self.write_like = dirpath / self.input_files[0]['filename']
        read_bands = dict()
        for file in self.input_files:
            for band, idx in zip(file['bands'], 
                                 range(1, len(file['bands'])+1)):
                read_bands[band] = (dirpath / file['filename'], idx)
        self.ndarrs = read_bands
        self.datasets = dict()

dirpaths = [
    R'/home/pitter/Tree-Classification/casuarina//93deg',
    R'/home/pitter/Tree-Classification/casuarina//183deg'
]

for dirpath in dirpaths:
    os.chdir(dirpath)
    load_out = load(dirpath, casuarina_input_dat)
    label_out = label(*load_out)
    label_out[3].to_csv('bounds.csv')
    del load_out
    del label_out
    gc.collect()
    
# %% Filter the bounds produced
os.chdir(r"/home/pitter/Tree-Classification")

dirpaths = [
    R'/home/pitter/Tree-Classification/casuarina//93deg',
    R'/home/pitter/Tree-Classification/casuarina//183deg',
    R'/home/pitter/Tree-Classification/chestnut//10May2021',
    R'/home/pitter/Tree-Classification/chestnut//18Dec2020'
]


for dirpath in dirpaths:
    filepath = dirpath + '/bounds.csv'
    csv = pd.read_csv(filepath, index_col=0)
    csv = csv[(csv['maxr'] - csv['minr'] > 224) & (csv['maxc'] - csv['minc'] > 224)]
    filepath = dirpath + '/{}_filtered_bounds.csv'.format(Path(dirpath).name)
    csv.to_csv(filepath)


























