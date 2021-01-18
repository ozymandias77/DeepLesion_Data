#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 16:16:11 2021

to prepare your dataset for tf obj detection api

@author: ztao
"""

"""
Process DeepLesion truth csv to save focus slice with gt boxes
"""

import csv
import cv2
import random
import numpy as np
import tensorflow as tf
import sys
import os
import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#from object_detection.utils import dataset_util
#import contextlib2
#from object_detection.dataset_tools import tf_record_creation_util

datadir = '/media/ztao/New Volume/DeepLesion/Images_png/'
lesion_dict = {1 : 'bone',
               2 : 'abdomen',
               3 : 'mediastinum',
               4 : 'liver',
               5 : 'lung',
               6 : 'kidney',
               7 : 'soft tissue',
               8 : 'pelvis',
               9 : 'unknown',
               -1 : 'test'}


HALF_THICK = 0 #mm
MAX_NUM = 100000000

def GetPngName(sliceID):
    return (str(sliceID + 100000))[-3:] + '.png'

def GetAllInfo(csvfilename):
    allTrus = {}
    num_png = 0
    num_bbox = 0
    with open(csvfilename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        idx = 0
        for tru in reader:
            class_id = int(tru['Coarse_lesion_type'])
#            if class_id == -1: 
#                num_test += 1
#                continue
#            else:
#                class_id = 1
            class_id = 1
            
            name = tru['File_name']
            print(name)
            split_idx = name.rfind('_')
            folder_name = datadir + name[:split_idx] + '/'
            
            # image_data = cv2.imread(png_file_name) #cv2 default read RGB
            # dict['encoded_image_data'] = image_data.tostring()
                  
            key_slice = int(tru['Key_slice_index'])
            slice1 = int(tru['Slice_range'].split(',')[0])
            slice2 = int(tru['Slice_range'].split(',')[1])
            width = int(tru['Image_size'].split(',')[0])
            height = int(tru['Image_size'].split(',')[1])      
            dicom_win = [float(i) for i in tru['DICOM_windows'].split(',')]
            bbox = np.array([float(i) for i in (tru['Bounding_boxes'].split(','))])
            x1 = min(bbox[0], bbox[2])/width
            x2 = max(bbox[0], bbox[2])/width
            y1 = min(bbox[1], bbox[3])/height
            y2 = max(bbox[1], bbox[3])/height
            class_text = lesion_dict[class_id]
                
#            print(folder_name)
#            print('image shape = ({}, {})'.format(width, height))
#            print('key slice = {}, slice range = ({}, {}), window width = {}'.format(key_slice, slice1, slice2, dicom_win))
#            print('lesion type = ', class_text)
#            break
            
            start_slice = slice1
            end_slice = slice2 + 1
            for slice_id in range(start_slice, end_slice):
                dict = {}
                
                png_name = GetPngName(slice_id)                
                png_file_name = folder_name + png_name
                key_name = name[:split_idx + 1] + png_name
                
                if not os.path.exists(png_file_name):
                    print('{} not exist!'.format(png_file_name))
                    continue
                
                num_bbox += 1
                dict['filename'] = png_file_name 
                dict['height'] = height
                dict['width'] = width
                dict['win_min'] = dicom_win[0]
                dict['win_max'] = dicom_win[1]
                dict['xmins'] = [x1]
                dict['xmaxs'] = [x2]
                dict['ymins'] = [y1]
                dict['ymaxs'] = [y2]
                dict['classes'] = [class_id]
                dict['classes_text'] = [class_text.encode()] # use bytes representation
                            
                if key_name not in allTrus:
                    allTrus[key_name] = dict
                    num_png += 1
                else:
                    #need to combone info
                    #print('{} already exists'.format(key_name)) 
                    allTrus[key_name]['xmins'].append(dict['xmins'][0])
                    allTrus[key_name]['xmaxs'].append(dict['xmaxs'][0])
                    allTrus[key_name]['ymins'].append(dict['ymins'][0])
                    allTrus[key_name]['ymaxs'].append(dict['ymaxs'][0])
                    allTrus[key_name]['classes_text'].append(dict['classes_text'][0])
                    allTrus[key_name]['classes'].append(dict['classes'][0])
    #                break             
            idx += 1
            if (idx == MAX_NUM): break
                        
    fp = open('data_stats.txt', 'w')
    print('total_lesion = {}, total_image = {}'.format(num_bbox, num_png))
    fp.write('total_lesion = {}, total_image = {}'.format(num_bbox, num_png))
    
    if HALF_THICK >= 0:
        selectTrus = {}
        num_png = 0
        num_bbox = 0
        with open(csvfilename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for tru in reader:           
                name = tru['File_name']             
                split_idx = name.rfind('_')
                key_slice = int(tru['Key_slice_index'])
                slice_thickness = float(tru['Spacing_mm_px_'].split(',')[-1])
                half_range = int(HALF_THICK / slice_thickness)
                
                for slice_id in range(key_slice - half_range, key_slice + half_range + 1):
                    png_name = GetPngName(slice_id)          
                    key_name = name[:split_idx + 1] + png_name
                    if key_name in allTrus:
                        num_png += 1
                        num_bbox += len(allTrus[key_name]['xmins'])
                        selectTrus[key_name] = allTrus[key_name]
                        
        print('select_lesion = {}, select_image = {}'.format(num_bbox, num_png))
        fp.write('select_lesion = {}, select_image = {}'.format(num_bbox, num_png))
        allTrus = selectTrus
    
    fp.close()
    #sys.exit()
    return allTrus

def SaveFocusDeepLesion(csvfilename):
    
    with open(csvfilename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        imagenum = 0
        for tru in reader:
            is_test = False
            class_id = int(tru['Coarse_lesion_type'])
            if class_id == -1: 
                is_test = True
            
            name = tru['File_name']
#            print(name)
            split_idx = name.rfind('_')
            folder_name = datadir + name[:split_idx] + '/'
            
            # image_data = cv2.imread(png_file_name) #cv2 default read RGB
            # dict['encoded_image_data'] = image_data.tostring()
                  
            key_slice = int(tru['Key_slice_index'])
            width = int(tru['Image_size'].split(',')[0])
            height = int(tru['Image_size'].split(',')[1])    
            
            if (height != 512) or (width != 512):
                print('skip image with size other than 512')
                continue
            
            if is_test:
                print('skip lesion with id -1')
                continue
            
            bbox = np.array([float(i) for i in (tru['Bounding_boxes'].split(','))])
            x1 = min(bbox[0], bbox[2])/width
            x2 = max(bbox[0], bbox[2])/width
            y1 = min(bbox[1], bbox[3])/height
            y2 = max(bbox[1], bbox[3])/height
#            class_text = lesion_dict[class_id]
            
            save_base = name[:-3]
            
            png_name = GetPngName(key_slice)
            png_file_name = folder_name + png_name
            if not os.path.exists(png_file_name):
                print('{} not exist!'.format(png_file_name))
                continue
            
            save_tru_name = save_base + 'tru'
            save_img_name = save_base + 'jpg'
            if is_test:
                save_tru_name = 'test/' + save_tru_name
                save_img_name = 'test/' + save_img_name
            
            if not os.path.exists(save_img_name):
                #pre-process image
                image = cv2.imread(png_file_name, -1)
                if image is None:
                    continue
                image = (image.astype(np.int32) - 32768)#.astype(np.int16)
                minval = -1024.0
                maxval = 3071.0
                factor = 255.0/(maxval - minval)
                image = (image - minval) * factor
                image = np.clip(image, 0, 255)
                image = image.astype(np.uint8)
                image_rgb = cv2.merge([image, image, image])
                cv2.imwrite(save_img_name, image_rgb)
                imagenum += 1
            
            print(save_img_name)
            fp = open(save_tru_name, 'a')
            fp.write("{:3d} {:8.4f} {:8.4f} {:8.4f} {:8.4f}\n".format(class_id, y1, x1, y2, x2))

            fp.close()
            
            if (imagenum == 2000): break  

if __name__ == '__main__':
    allTrus = SaveFocusDeepLesion('/media/ztao/New Volume/DeepLesion/DL_info.csv')

