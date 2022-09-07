#DEF number_classes = 40 # /!\ This must be known before the compilation, see also here: https://stackoverflow.com/questions/46306737/cython-compilation-error-not-allowed-in-a-constant-expression. 40 classes is the max number of classes for a UVP6 model hence 40 here (it allocates a static number of 40 in the memory, which is needed by C (based on Fabio's code float* const pred_scores_per_cat.

cimport numpy as np 
from cpython cimport array 
from numpy cimport uint8_t, uint16_t, uint32_t, float
from libcpp.string cimport string # https://stackoverflow.com/questions/3870772/cython-c-and-stdstring
import os

## features extraction
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

## xgboost_import 
cdef extern from "boost_prediction.hpp":
    ctypedef struct decision_tree:
       float *value
       uint8_t *idx
       uint8_t *next

    ctypedef struct predictor:
     uint32_t category_nb
     uint32_t trees_per_cat
     uint32_t total_nb_nodes
     uint32_t *nodes_in_tree
     decision_tree *trees

cdef extern from "xgboost_import.cpp":
     int load_xgboost_predictor(predictor *const pred, const string &filename, const uint8_t nb_categories)
     int dump_predictor(const predictor *const pred, const string &filename)
    
def py_convert_txt_model_to_binary_model(num_classes, xgboost_model_filename):
    cdef predictor pred  
    cdef int nb_categories = num_classes
    cdef string xgb_filename = xgboost_model_filename
    cdef string output_filename = os.path.splitext(xgboost_model_filename)[0]

    print('Loading xgboost prediction model...')
    tmp = load_xgboost_predictor(&pred, xgb_filename, nb_categories)
    
    if tmp == 0:
        print('Success !')
        print('Conversion of model from txt to binary...')
        dump_predictor(&pred, output_filename)
    else:
        print('Failed !')
        return
    
cdef extern from "boost_prediction.cpp":
    int load_predictor(predictor *const pred, const char *const filename)
    void release_predictor(predictor *const pred)
    void predict(const predictor *const pred, const float *const input_vec, uint32_t *const best_pred_id, float *const best_pred_score, float *const pred_scores_per_cat)
    
## load predictor
def py_load_model_and_predict(binary_filename, input_features, nb_classes):

    cdef predictor pred # structure is empty at this time, it will be filled by the C++ module
    
    cdef char* uvp6_filename = binary_filename # https://stackoverflow.com/questions/28002214/cython-typeerror-an-integer-is-required
    
    size_model = load_predictor(&pred, uvp6_filename) # no need to reference (&) binary_filename because a char is already a pointer to a table in C/C++ (like an array)
    
    if size_model < 0:
        print('Failed to load UVP6 model')
        return
    #else:
    #    print('Model size (in bytes): '+str(size_model)) 
    
    cdef array.array features_in = input_features # https://cython.readthedocs.io/en/stable/src/tutorial/array.html AND https://docs.python.org/3/library/array.html
    #cdef cnp.ndarray features_in = input_features # https://www.futurelearn.com/info/courses/python-in-hpc/0/steps/65126
    
    #print(features_in)
    
    cdef uint32_t best_pred_id
    
    cdef float best_pred_score
    
    #cdef float pred_scores_per_cat[20] # could also be used with a fixed number here and in the return function below but then let's just define it at the very beginning of this file.
    #cdef float pred_scores_per_cat[number_classes] # can be used like this but 'number_classes' needs to be defined BEFORE the compilation otherwise C will not be happy.
    cdef float pred_scores_per_cat[40] # needs to be known and cannot be a constant so I take the max number of UVP6 classes allowed.

    predict(&pred, features_in.data.as_floats, &best_pred_id, &best_pred_score, pred_scores_per_cat)
    
    release_predictor(&pred)
    
    #return(best_pred_id, best_pred_score, [x for x in pred_scores_per_cat[:20]])
    return(best_pred_id, best_pred_score, [x for x in pred_scores_per_cat[:nb_classes]])
    
