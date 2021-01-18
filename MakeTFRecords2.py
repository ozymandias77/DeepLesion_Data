# -*- coding: utf-8 -*-
"""
Created on Sun May 10 11:24:11 2020

@author: zhong
"""

import csv
import cv2
# from PIL import Image
# import matplotlib.image as mpimage
import random
import numpy as np
import tensorflow as tf
#import matplotlib.image as mpimg 
#import matplotlib.pyplot as plt

from object_detection.utils import dataset_util
import contextlib2
from six.moves import range

import sys
import os
import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

flags = tf.compat.v1.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('input_csv', '/media/ztao/New Volume/DeepLesion/DL_info.csv', 'Path to output TFRecord')
FLAGS = flags.FLAGS


#def samplecode__create_tf_example():
#  # TODO(user): Populate the following variables from your example.
#  height = None # Image height
#  width = None # Image width
#  filename = None # Filename of the image. Empty if image is not from file
#  encoded_image_data = None # Encoded image bytes
#  image_format = None # b'jpeg' or b'png'
#
#  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
#  xmaxs = [] # List of normalized right x coordinates in bounding box
#             # (1 per box)
#  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
#  ymaxs = [] # List of normalized bottom y coordinates in bounding box
#             # (1 per box)
#  classes_text = [] # List of string class name of bounding box (1 per box)
#  classes = [] # List of integer class id of bounding box (1 per box)
#
#  tf_example = tf.train.Example(features=tf.train.Features(feature={
#      'image/height': dataset_util.int64_feature(height),
#      'image/width': dataset_util.int64_feature(width),
#      'image/filename': dataset_util.bytes_feature(filename),
#      'image/source_id': dataset_util.bytes_feature(filename),
#      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
#      'image/format': dataset_util.bytes_feature(image_format),
#      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
#      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
#      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
#      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
#      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
#      'image/object/class/label': dataset_util.int64_list_feature(classes),
#  }))
#  return tf_example


#def samplecode__main(_):
#  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
#
#  # TODO(user): Write code to read in your dataset to examples variable
#
#  for example in examples:
#    tf_example = create_tf_example(example)
#    writer.write(tf_example.SerializeToString())
#
#  writer.close()
  
""" 
the following function is same as object_detection/tf_record_creation_util.py
except for the python_io part, which is a bug that I have fixed here, for tf2 use
"""
def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
  tf_record_output_filenames = [
      '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
      for idx in range(num_shards)
  ]

  tfrecords = [
      exit_stack.enter_context(tf.io.TFRecordWriter(file_name))
      for file_name in tf_record_output_filenames
  ]

  return tfrecords

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

#lesion_dict = {1 : 'lesion'}

HALF_THICK = 0 #mm
MAX_NUM = 100000000

def GetPngName(sliceID):
    return (str(sliceID + 100000))[-3:] + '.png'

def GetAllInfo(csvfilename):
    allTrus = {}
    num_png = 0
    num_bbox = 0
    num_test = 0
    num_noise = 0
    with open(csvfilename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for tru in reader:
            class_id = int(tru['Coarse_lesion_type'])
            if class_id == -1: 
                num_test += 1
                continue
            elif class_id == 9:
                num_noise += 1
                continue
            
            name = tru['File_name']
            print(name)
            split_idx = name.rfind('_')
            folder_name = datadir + name[:split_idx] + '/'
            
            key_slice = int(tru['Key_slice_index'])
            slice1 = int(tru['Slice_range'].split(',')[0])
            slice2 = int(tru['Slice_range'].split(',')[1])
            width = int(tru['Image_size'].split(',')[0])
            height = int(tru['Image_size'].split(',')[1])      
            dicom_win = [float(i) for i in tru['DICOM_windows'].split(',')]
            bbox = np.array([float(i) for i in (tru['Bounding_boxes'].split(','))])
            x1 = min(bbox[0], bbox[2])/width
            x2 = max(bbox[0], bbox[2])/height
            y1 = min(bbox[1], bbox[3])/width
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
    
            if (num_png >= MAX_NUM): break
                        
    fp = open('data_stats.txt', 'w')
    print('total_lesion = {}, total_image = {}, num_test = {}, num_noise = {}'.format(num_bbox, num_png, num_test, num_noise))
    fp.write('total_lesion = {}, total_image = {}, num_test = {}, num_noise = {}\n'.format(num_bbox, num_png, num_test, num_noise))
    
    if HALF_THICK >= 0:
        selectTrus = {}
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
                        
        num_bbox = 0
        for key_name in selectTrus:
            num_bbox += len(selectTrus[key_name]['xmins'])            
        print('HALF_THICK = {}, select_lesion = {}, select_image = {}'.format(HALF_THICK, num_bbox, len(selectTrus)))
        fp.write('HALF_THICK = {}, select_lesion = {}, select_image = {}'.format(HALF_THICK, num_bbox, len(selectTrus)))        
        allTrus = selectTrus
    
    fp.close()
    #sys.exit()
    return allTrus


def create_tf_example(trudict):
    image_format = b'png'
    # with tf.gfile.GFile(trudict['filename'], 'rb') as fid:
    #     image_data = fid.read()
    # if image_data is None:
    #     return None
    
    image = cv2.imread(trudict['filename'], -1)
    if image is None:
        return None
    image = (image.astype(np.int32) - 32768)#.astype(np.int16)
    minval = -1024.0
    maxval = 3071.0
    if (maxval > minval):
        factor = 255.0/(maxval - minval)
        image = (image - minval) * factor
        image = np.clip(image, 0, 255)
    else:
        print('error: maxval = {}, minval = {}'.format(maxval, minval))
        sys.exit()
    image = image.astype(np.uint8)
    image_rgb = cv2.merge([image, image, image])
    _, encoded_image = cv2.imencode('.png', image_rgb)
    image_data = encoded_image.tobytes()
    # sys.exit()
    
    filename = trudict['filename'].encode()
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(trudict['height']),
        'image/width': dataset_util.int64_feature(trudict['width']),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(trudict['xmins']),
        'image/object/bbox/xmax': dataset_util.float_list_feature(trudict['xmaxs']),
        'image/object/bbox/ymin': dataset_util.float_list_feature(trudict['ymins']),
        'image/object/bbox/ymax': dataset_util.float_list_feature(trudict['ymaxs']),
        'image/object/class/text': dataset_util.bytes_list_feature(trudict['classes_text']),
        'image/object/class/label': dataset_util.int64_list_feature(trudict['classes']),
        }))
    return tf_example


def ShuffleDict(inputdict):
    keys =  list(inputdict.keys())
    random.shuffle(keys)
    shuffledDict = dict()
    for key in keys:
        shuffledDict.update({key:inputdict[key]})
    return shuffledDict


def WriteShardedTFR(input_dict, filenamebase, num_shards):
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = open_sharded_output_tfrecords(tf_record_close_stack, filenamebase, num_shards)
        index = 0
        for item in input_dict:
            print(item)
            tf_example = create_tf_example(input_dict[item])
            if tf_example is not None:
                output_shard_index = index % num_shards
                output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
                index += 1


def WriteTFR(input_dict, output_path):
    with tf.io.TFRecordWriter(output_path) as file_writer:
        for item in input_dict:
            print(item)
            tf_example = create_tf_example(input_dict[item])
            if tf_example is not None:
                file_writer.write(tf_example.SerializeToString())
            
def main(_):
    #output_filebase = FLAGS.output_path + 'train_dataset.record'
    truDict = GetAllInfo(FLAGS.input_csv)
    truDict = ShuffleDict(truDict)
    
    # len(truDict) is the total number of images
    # use num_eval_or_test as the number of images for evaluation during training
    # use num_eval_or_test as the number of images for final test
    num_eval_or_test = min(1000, len(truDict)//20)
    num_val2 = num_eval_or_test * 2
    tru_dict_eval = dict(list(truDict.items())[:num_eval_or_test]) 
    tru_dict_test = dict(list(truDict.items())[num_eval_or_test:num_val2]) 
    tru_dict_train = dict(list(truDict.items())[num_val2:]) 
    output_train_filebase = FLAGS.output_path + 'train_deeplesion.record'
    output_eval_filebase = FLAGS.output_path + 'eval_deeplesion.record'
    output_test_filebase = FLAGS.output_path + 'test_deeplesion.record'
    
    num_shards = len(tru_dict_train) // num_eval_or_test # each shard contains same number of images as test/eval 
    WriteShardedTFR(tru_dict_train, output_train_filebase, num_shards)
 
    WriteShardedTFR(tru_dict_eval, output_eval_filebase, 1)
    # WriteTFR(tru_dict_eval, output_eval_filebase)

    WriteShardedTFR(tru_dict_test, output_test_filebase, 1)
    # WriteTFR(tru_dict_test, output_test_filebase)
    
    with open('training_images_stats.txt', 'w') as fp:            
        print('training data size = {}; eval data size = {}, test data size = {}'.format(len(tru_dict_train), len(tru_dict_eval), len(tru_dict_test))) 
        fp.write('training data size = {}; eval data size = {}, test data size = {}\n'.format(len(tru_dict_train), len(tru_dict_eval), len(tru_dict_test))) 
        fp.write('HALF_THICK = {}'.format(HALF_THICK))
        
if __name__ == '__main__':
    tf.compat.v1.app.run()