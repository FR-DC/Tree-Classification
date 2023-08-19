#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

#%%

files_for_inner_join =\
[
 [
  r"/home/pitter/Tree-Classification/chestnut/18Dec2020/18Dec2020_filtered_bounds.csv",
  r"/home/pitter/Tree-Classification/chestnut/10May2021/10May2021_filtered_bounds.csv",
  r"/home/pitter/Tree-Classification/chestnut/chestnut_18Dec2020_10May2021_bounding_boxes_match_filtered.csv",
 ],
 [
  r"/home/pitter/Tree-Classification/casuarina/183deg/183deg_filtered_bounds.csv",
  r"/home/pitter/Tree-Classification/casuarina/93deg/93deg_filtered_bounds.csv",
  r"/home/pitter/Tree-Classification/casuarina/casuarina_183deg_93deg_bounding_boxes_match_filtered.csv",
 ]
]

for files in files_for_inner_join:
    matches = pd.read_csv(files[2], index_col=0)
    col1 = pd.read_csv(files[0], index_col=0)
    col2 = pd.read_csv(files[1], index_col=0)
    col1 = col1[col1['uuid'].isin(matches.iloc[:,0])]
    col2 = col2[col2['uuid'].isin(matches.iloc[:,1])]
    col1.to_csv(files[0][:-4] + '_without_unmatched.csv')
    col2.to_csv(files[1][:-4] + '_without_unmatched.csv')
                           
#%%

files =\
[
 r"/home/pitter/Tree-Classification/chestnut/18Dec2020/18Dec2020_filtered_bounds.csv",
 r"/home/pitter/Tree-Classification/chestnut/10May2021/10May2021_filtered_bounds.csv",
 r"/home/pitter/Tree-Classification/chestnut/18Dec2020/18Dec2020_filtered_bounds_without_unmatched.csv",
 r"/home/pitter/Tree-Classification/chestnut/10May2021/10May2021_filtered_bounds_without_unmatched.csv",
 r"/home/pitter/Tree-Classification/chestnut/chestnut_18Dec2020_10May2021_bounding_boxes_match_filtered.csv",
 r"/home/pitter/Tree-Classification/chestnut/chestnut_18Dec2020_10May2021_bounding_boxes_match.csv",
 r"/home/pitter/Tree-Classification/casuarina/183deg/183deg_filtered_bounds.csv",
 r"/home/pitter/Tree-Classification/casuarina/93deg/93deg_filtered_bounds.csv"
 r"/home/pitter/Tree-Classification/casuarina/183deg/183deg_filtered_bounds_without_unmatched.csv",
 r"/home/pitter/Tree-Classification/casuarina/93deg/93deg_filtered_bounds_without_unmatched.csv"
 r"/home/pitter/Tree-Classification/casuarina/casuarina_183deg_93deg_bounding_boxes_match_filtered.csv",
 r"/home/pitter/Tree-Classification/casuarina/casuarina_183deg_93deg_bounding_boxes_match.csv",
]

for file in files:
    df = pd.read_csv(file)
    df.to_json(file[:-4] + '.json')