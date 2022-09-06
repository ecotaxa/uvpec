#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

//
// Structure predictor contains the basic info and pointers to data buffers, describing a
//      random forest prediction model, as created by the XGBoost library. It can be populated
//      by the "load_predictor" function, or by the "load_xgboost_predictor" function from
//      "xgboost_import" library. It is given to the "predict" function to perform vignette
//      classification from a given input features vector.
//
typedef struct {
    float *value;
    uint8_t *idx;
    uint8_t *next;
} decision_tree;

typedef struct {
    uint32_t category_nb;
    uint32_t trees_per_cat;
    uint32_t total_nb_nodes;
    uint32_t *nodes_in_tree;
    decision_tree *trees;
} predictor;


//
// load_predictor function :
//      Loads a predictor from a binary file containing a prediction model in UVP6 format
//          (previously created using the "dump_predictor" function from "xgboost_import" library)
//
// PARAMETERS :
//      pred : pointer to an empty/new predictor structure, which will be filled
//
//      filename : "path to/name of" the binary file containing the prediction model to import, in UVP6 format
//
// RETURN VALUE : number of bytes read from the binary file if success, negative error_code if fails
//
int load_predictor(predictor *const pred, const char *const filename);


//
// release_predictor function :
//      Releases data buffers and clear the predictor structure, which can be discarded or
//          re-used to load a new model, avoiding memory leak
//
// PARAMETERS :
//      pred : pointer to a populated predictor structure to release
//
void release_predictor(predictor *const pred);


//
// predict function :
//      Performs automatic classification for a sample, using a prediction model previously loaded in
//          the predictor structure, and an input features vector
//
// PARAMETERS :
//      pred : pointer to a populated predictor structure, to be used as prediction model for the
//          automatic classification
//
//      input_vec : pointer to an array of floats, containing the features extracted from
//          the sample to classify
//
//      best_pred_id : pointer to an unsigned int where the id (from 0 to <category_nb - 1>) of the predicted
//          class/category will be returned. May use NULL if this value is not necessary
//
//      best_pred_score : pointer to a float where the prediction score for the predicted class/category
//          will be returned. May use NULL if this value is not necessary
//
//      pred_scores_per_cat : pointer to an array of floats (minimum size <category_nb>) where the prediction
//          scores for each class/category will be returned. May use NULL if these values are not necessary
//
void predict(const predictor *const pred, const float *const input_vec, uint32_t *const best_pred_id, float *const best_pred_score, float *const pred_scores_per_cat);

#ifdef __cplusplus
}
#endif



