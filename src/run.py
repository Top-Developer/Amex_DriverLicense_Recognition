# import packages
import os
import cv2
import sys
import argparse
import numpy as np
import tensorflow as tf

# import utilities
from utils import label_map_util
from utils import visualization_utils as vis_util

# import const parameters for this project
from common import utils as U
from common import const as C

import pandas

# global parameters
models  = []            # model list for detection stage
cwdPath = os.getcwd()   # full current path


def main(args):
    # arguments
    testImgDir = args.testImgDir
    
    # loaded models
    for idx in range(C.MODEL_CNT):
        models.append(load_model_into_memory(idx))

    # loop in image file for test
    imgFileList = U.get_files_match_ext( [testImgDir], ['jpg','png'] )
    for imgName in imgFileList:

        # load image file and check none or not
        img = U.read_image_with_opencv(testImgDir, imgName)
        if img is None: continue

        # running detector to detect credit card and driver license
        preResult   = run_session(img, models[C.MODEL_IDX_CARD_LICENSE])
        if preResult is None: continue

        # running detector to detect info
        infoResult  = detect_info(img, preResult)

        # running detector to recognize info
        recogResult = recognize_info(img, preResult, infoResult)
        
        # draw detected result on image
        # draw_result_on_image(img, preResult, infoResult)
        print('=======================')

    return


# recognize detected info
def recognize_info(img, preResult, infoResult):

    drawImg = img.copy()

    # loop in preResult that detects credit card or driver license
    _, preClasses, preCategories = preResult
    for i in range(len(preClasses)):

        print(C.CARD_LICENSE_NAME[preCategories[preClasses[i]]['name']])
        
        # loop in infoResult that detects info 
        # such as card number, expiration date, issued date, and so on
        # draw detected area
        infoBoxes, infoClasses, infoCategories = infoResult[i]
        drawImg = draw_detected_object(drawImg, infoBoxes)

        for j in range(len(infoBoxes)):
            
            # print category name
            categoryName = infoCategories[infoClasses[j]]['name']

            if categoryName in C.BANK_STATE_NAME:
                print(C.BANK_STATE_NAME[categoryName])

            # if category name doesn't exist in MODEL_IDX_FROM_CATEGORY,
            # continue
            elif categoryName in C.MODEL_IDX_FROM_CATEGORY:
                
                # detected object image and model index
                infoImg  = U.get_detected_image(img, infoBoxes[j])
                modelIdx = C.MODEL_IDX_FROM_CATEGORY[categoryName]

                # new detection
                boxes, classes, categories = run_session(infoImg, models[modelIdx], infoBoxes[j])

                # detected digits and draw detection result
                digits  = digits_recognition(boxes, classes, categories)
                drawImg = draw_detected_object(drawImg, boxes)

                print(categoryName, digits)

        cv2.imshow('img', drawImg)
        U.delay()

    return

def digits_recognition(boxes, classes, categories):
    
    idxList = list(range(len(boxes)))       # detected index list
    
    digits = ''
    # if detected index exists, then loop
    while len(idxList) > 0:
        
        i = idxList[0]                                  # current index
        x_i = int( (boxes[i][1] + boxes[i][3]) / 2 )    # x cordinate of current object

        # loop the rest of points
        for j in idxList:
            x_j = int( (boxes[j][1] + boxes[j][3]) / 2 )
            if x_i > x_j:
                x_i = x_j
                i = j
        
        digits = digits + categories[classes[i]]['name']
        idxList.remove(i)

    return digits


# draw detected result on image
def draw_detected_object(img, boxes):

    for k in range(len(boxes)):
    
        # detected area position
        ymin, xmin, ymax, xmax = boxes[k]
        
        # draw detected area of credit card or driver license from preResult
        img = cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,255,0), 1)

    return img


# draw detected result on image
def draw_result_on_image(img, preResult, infoResult):

    # loop in preResult that detects credit card or driver license
    preBoxes, preClasses, preCategories = preResult
    for i in range(len(preBoxes)):

        # detected area position
        ymin, xmin, ymax, xmax = preBoxes[i]

        # draw detected area of credit card or driver license from preResult
        cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,255,0), 2)

        # print category name
        print('--------------------')
        print(preCategories[preClasses[i]]['name'])

        # loop in infoResult that detects info 
        # such as card number, expiration date, issued date, and so on
        infoBoxes, infoClasses, infoCategories = infoResult[i]
        for j in range(len(infoBoxes)):
            
            # print category name
            print(infoCategories[infoClasses[j]]['name'])
            
            # detected area position
            ymin, xmin, ymax, xmax = infoBoxes[j]
            
            # draw detected area of credit card or driver license from preResult
            cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,255,0), 1)
    
    cv2.imshow('detectedImg', img)
    U.delay()

    return

# detect info in credit card and driver license detected from origin image
def detect_info(img, preResult):
    
    # get each parameters of detected result for credit card or driver license
    boxes, classes, category_index = preResult

    # new parameters for new detection
    result = []

    # loop in detected images
    for idx in range(len(boxes)):
        # get detected area from image
        detectedImg = U.get_detected_image(img, boxes[idx])
        if detectedImg is None: continue

        # running detector to detect info, and then append result.
        subResult = run_session(detectedImg, models[classes[idx]], boxes[idx])
        result.append(subResult)
        
    return result


# tensorflow model and its prameters 
def load_model_into_memory(stageIdx):
    
    # path to frozen detection graph .pb file, which contains the model
    # path to label map file
    # number of classes the area detector can identify
    ckptPath   = os.path.join(cwdPath, C.MODEL_DIR_PATH[stageIdx], C.FROZEN_GRAPH_NAME)
    labelsPath = os.path.join(cwdPath, C.MODEL_DIR_PATH[stageIdx], C.LABEL_MAP_FILE_NAME)
    classNum   = C.CLASSES_NUM[stageIdx]

    # label map and category index
    labelMap     = label_map_util.load_labelmap(labelsPath)
    categories   = label_map_util.convert_label_map_to_categories(labelMap, max_num_classes=classNum, use_display_name=True)
    category_idx = label_map_util.create_category_index(categories)

    # load the tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(ckptPath, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='') 

        sess = tf.Session(graph=detection_graph)

    ### define input and output tensors for the area detecttion classifier ###

    # input tensor: image
    imgTensor = detection_graph.get_tensor_by_name('image_tensor:0')
    
    # output tensor: detection boxies, scores and class
    # each box represents a info part in image
    # each score represents confidence level for each info
    # each class represents the class of info
    detection_boxes   = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores  = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # number of detected area
    detectedNum   = detection_graph.get_tensor_by_name('num_detections:0')

    # detected model
    detectedModel = [ sess, 
                      detection_boxes, 
                      detection_scores, 
                      detection_classes, 
                      category_idx, 
                      detectedNum, 
                      imgTensor
                    ]

    return detectedModel


# perform detection by running the model
def run_session(img, model, box_offset=(0,0,0,0)):

    # get model parameters for detection
    sess              = model[C.SESS]
    detection_boxes   = model[C.DETECTION_BOXES]
    detection_scores  = model[C.DETECTION_SCORES]
    detection_classes = model[C.DETECTION_CLASSES]
    category_index    = model[C.CATEGORY_INDEX]
    detectedNum       = model[C.DETECTED_NUM]
    imgTensor         = model[C.IMG_TENSOR]

    # convert source image to flattend/expanded image
    expandedImg = np.expand_dims(img, axis=0)

    # perform detection by running the model
    (boxes, scores, classes, _) = sess.run( [ detection_boxes, detection_scores, detection_classes, detectedNum],
                                            feed_dict={imgTensor:expandedImg}
                                          )
    
    # detectio results
    boxes   = np.squeeze(boxes)
    scores  = np.squeeze(scores)
    classes = np.squeeze(classes).astype(np.int32)

    result = normalize_detected_results(img, boxes, scores, classes, category_index, box_offset)

    return result


# normalize results:
# pop up result if score is less than min_score_thresh
# convert ratio position to absolute position of boxes
def normalize_detected_results(img, boxes, scores, classes, category_index, box_offset=(0,0,0,0)):

    if U.isImage_opencv(img) is False: 
        return None

    # index list whose scores less than min_score_thresh
    # this is a list of index that is no area
    idxList = [idx for idx in range(boxes.shape[0]) if scores[idx] < C.MIN_SCORE_THRESH]
    
    # pop up no area from the results
    # only real areas remains
    boxes   = np.delete(boxes, idxList, axis=0)
    scores  = np.delete(scores, idxList, axis=0)
    classes = np.delete(classes, idxList, axis=0)

    # image size, multiplication ratio vector
    height, width  = img.shape[:2]
    mult_ratio_vec = [height, width, height, width]

    # convert ratio position to absolute position of boxes
    offsetVec = (box_offset[0], box_offset[1], box_offset[0], box_offset[1])
    boxes = [list(np.add(np.multiply(boxes[idx], mult_ratio_vec).astype(int), offsetVec)) for idx in range(boxes.shape[0])]

    result = [ boxes, 
               classes, 
               category_index
             ]    

    return result


# define argument parser
def parse_arguments():
    
    # create argument parser
    parser = argparse.ArgumentParser('initial procession')

    # add argument key and default value
    parser.add_argument('--testImgDir',     type=str,   default='./testImage',   help='source directory of images')
    # parser.add_argument('--testImgDir',   	type=str, 	default='./images_card_license/test', 	help='source directory of images')

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_arguments()	# get arguments

    main(args)
    # main_process(args)

# destroy all window from memory
cv2.destroyAllWindows()