# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:21:34 2023

@author: ADIL003
"""
import os
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree  as ET
import shutil
import datetime
import rasterio as rio

os.chdir('E:\Tree-Classification')

dirpaths = [
    'E:\Tree-Classification\chestnut\\10May2021',
    'E:\Tree-Classification\chestnut\\18Dec2020'
]
dirpaths = [Path(dirpath) for dirpath in dirpaths]
for dirpath in dirpaths:
    os.chdir(dirpath)
    bounds = pd.read_csv(dirpath.name + '_bounds.csv')
    
    dirname = '{}_cvat_task'.format(dirpath.name)
    os.makedirs(dirname, exist_ok=True)
    os.chdir(dirname)
    
    os.makedirs('images', exist_ok=True)
    shutil.copy('result.tif', './images/result.tif')
    
    timestamp = datetime.datetime.now().isoformat()
    
    annotations = ET.Element('annotations')
    version = ET.SubElement(annotations, 'version', text='1.1')
    meta = ET.SubElement(annotations, 'meta')
    task = ET.SubElement(meta, 'task')
    ET.SubElement(task, 'id', text=1)
    ET.SubElement(task, 'name', text='segmentation')
    ET.SubElement(task, 'size', text=1)
    ET.SubElement(task, 'mode', text='annotation')
    ET.SubElement(task, 'overlap', text='0')
    ET.SubElement(task, 'bugtracker')
    ET.SubElement(task, 'flipped', text=False)
    ET.SubElement(task, 'created', text=timestamp)
    ET.SubElement(task, 'updated', text=timestamp)
    labels = ET.SubElement(task, 'labels')
    label = ET.SubElement(labels, 'label')
    ET.SubElement(label, 'name', text='tree')
    ET.SubElement(label, 'attributes')
    ET.SubElement(task, 'segments')
    ET.SubElement(task, 'dumped', text=timestamp)
    
    with rio.open('images/result.tif') as result_tif:
        image = ET.SubElement(annotations, 'image', id=0, name='result.tif',
                              width=result_tif.width, height=result_tif.height)
    for row in bounds.iterrows():
        ET.SubElement(image, 'box', label='tree',
                      xtl=row['minr'], ytl=row['minc'],
                      xbr=row['maxr'], ybr=row['maxc'])
    
    annotations.write('annotations.xml')
    
    os.chdir('..')
    
    shutil.make_archive('{}.zip'.format(dirname), 'zip', dirname)