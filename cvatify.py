# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:21:34 2023

@author: ADIL003
"""
import os
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
import shutil
import datetime
import rasterio as rio

os.chdir('/home/pitter/Tree-Classification')

dirpaths = [
    R'/home/pitter/Tree-Classification/casuarina//93deg',
    R'/home/pitter/Tree-Classification/casuarina//183deg',
    R'/home/pitter/Tree-Classification/chestnut//10May2021',
    R'/home/pitter/Tree-Classification/chestnut//18Dec2020'
]
dirpaths = [Path(dirpath) for dirpath in dirpaths]
for dirpath in dirpaths:
    os.chdir(dirpath)
    bounds = pd.read_csv(dirpath.name + '_filtered_bounds.csv')

    dirname = '{}_cvat_task'.format(dirpath.name)
    os.makedirs(dirname, exist_ok=True)
    os.makedirs(dirname + '/images', exist_ok=True)
    
    with rio.open('result.tif' if 'chestnut' in str(dirpath) else '{} result.tif'.format(dirpath.name)) as result_tif:
        width, height = result_tif.width, result_tif.height

    #if not os.path.isfile(dirname + '/images/result.tif'):
    #shutil.copy('result.tif', dirname + '/images/result.tif')
    os.chdir(dirname)

    os.makedirs('images', exist_ok=True)

    timestamp = datetime.datetime.now().isoformat()

    annotations = ET.Element('annotations')
    version = ET.SubElement(annotations, 'version').text = '1.1'
    meta = ET.SubElement(annotations, 'meta')
    task = ET.SubElement(meta, 'task')
    ET.SubElement(task, 'id').text = '1'
    ET.SubElement(task, 'name').text = 'segmentation'
    ET.SubElement(task, 'size').text = '1'
    ET.SubElement(task, 'mode').text = 'annotation'
    ET.SubElement(task, 'overlap').text = '0'
    ET.SubElement(task, 'bugtracker')
    ET.SubElement(task, 'flipped').text = 'False'
    ET.SubElement(task, 'created').text = timestamp
    ET.SubElement(task, 'updated').text = timestamp
    labels = ET.SubElement(task, 'labels')
    label = ET.SubElement(labels, 'label')
    ET.SubElement(label, 'name').text = 'tree'
    ET.SubElement(label, 'attributes')
    ET.SubElement(task, 'segments')
    ET.SubElement(task, 'dumped').text = timestamp

    image = ET.SubElement(annotations, 'image', id='0', name='result.tif',
                          width=str(width), height=str(height))
    for idx, row in bounds.iterrows():
        ET.SubElement(image, 'box', label='tree',
                      xtl=str(row['minc']), ytl=str(row['minr']),
                      xbr=str(row['maxc']), ybr=str(row['maxr']))
    
    tree = ET.ElementTree(annotations)
    tree.write('annotations.xml')

    os.chdir('..')

    shutil.make_archive('{}.zip'.format(dirname), 'zip', dirname)
