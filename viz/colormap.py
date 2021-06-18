import numpy as np

color_mapping = {
     0 : [192,192,192], 
     1 : [105,105,105], 
     2 : [160, 82, 45], 
     3 : [244,164, 96], 
     4 : [ 60,179,113],
     5 : [ 34,139, 34],
     6 : [154,205, 50],
     7 : [  0,128,  0],
     8 : [  0,100,  0],
     9 : [  0,250,154],
    10 : [139, 69, 19],
    11 : [  1, 51, 73],
    12 : [190,153,153],
    13 : [  0,132,111],
    14 : [  0,  0,142],
    15 : [  0, 60,100],
    16 : [135,206,250],
    17 : [128,  0,128],
    18 : [153,153,153],
    19 : [255,255,  0],
    20 : [220, 20, 60],
    21 : [255,182,193],
    22 : [220,220,220],
    255 : [  0,  0,  0]
}

semantic_map = {
     0 : 'asphalt', 
     1 : 'gravel', 
     2 : 'soil', 
     3 : 'sand', 
     4 : 'bush',
     5 : 'forest',
     6 : 'low grass',
     7 : 'high grass',
     8 : 'misc. vegetation',
     9 : 'tree crown',
    10 : 'tree trunk',
    11 : 'building',
    12 : 'fence',
    13 : 'wall',
    14 : 'car',
    15 : 'bus',
    16 : 'sky',
    17 : 'misc. object',
    18 : 'pole',
    19 : 'traffic sign',
    20 : 'person',
    21 : 'animal',
    22 : 'ego vehicle',
    255 : 'undefined'
}

def visualize_prediction(prediction):
    color_image = np.zeros((prediction.shape[0], prediction.shape[1], 3))
    for color_id in color_mapping.keys():
        color_image[prediction == color_id] = color_mapping[color_id]
    return color_image.astype(np.uint8)

def get_class_colors(class_ids):
    return [color_mapping[i] for i in class_ids]

def get_class_names(class_ids):
    return [semantic_map[i] for i in class_ids]