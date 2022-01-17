cimport numpy as np 
from cpython cimport array 
from numpy cimport uint8_t, uint16_t

cdef extern from "features_extraction.hpp":
    ctypedef struct Thumbnail:
       uint16_t x_min
       uint16_t y_min
       uint16_t width
       uint16_t height
       uint16_t x_obj
       uint16_t y_obj

cdef extern from "features_extraction.cpp":
    void get_features(const uint8_t *const input, const Thumbnail *const T, const uint16_t input_row_stride, const uint8_t threshold, float *const features)

def py_get_features(img, region, threshold):

    # flatten array
    flat_img = img.flatten() # C++ actually sees all arrays/tables in 1D
    
    # input de type array.array
    cdef array.array input = array.array('B', flat_img) # B for uint8
    
     # structure
    cdef Thumbnail tmb
    tmb.x_min = region.bbox[1]
    tmb.y_min = region.bbox[0]
    tmb.width = region.bbox[3] - region.bbox[1]
    tmb.height = region.bbox[2] - region.bbox[0]
    tmb.x_obj = region.coords[0][1] # coords will give the 'filled' pixel, ie. belonging to the object
    tmb.y_obj = region.coords[0][0]
    
    # row stride (if no padding, it is equivalent to the width of the input image)
    cdef int input_row_stride = img.shape[1]
    
    # treshold
    cdef int thresh = threshold

    # features
    cdef float features[55] # need to specify the length for C++
    
    get_features(input.data.as_uchars, &tmb, input_row_stride, thresh, features)
    
    return [x for x in features[:55]]
