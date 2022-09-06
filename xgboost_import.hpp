#pragma once

#include <stdint.h>
#include <string>
#include "boost_prediction.hpp"

//
// load_xgboost_predictor function :
//      Loads a prediction model from a text file in XGBoost text export format (XGBoost dump_model function)
//
// PARAMETERS :
//      pred : pointer to an empty/new predictor structure, from the "boost_prediction" library, which will be filled
//
//      filename : "path to/name of" the text file containing the XGBoost prediction model to import
//
//      nb_categories : number of classes in the prediction model to import. This information is needed
//          as it's not included in the XGBoost text export format
//
// RETURN VALUE : return 0 if success, negative error_code if fails
//
int load_xgboost_predictor(predictor *const pred, const std::string &filename, const uint8_t nb_categories);


//
// dump_predictor function :
//      Creates a binary file containing the prediction model, exportable to UVP6
//
// PARAMETERS :
//      pred : pointer to a populated predictor structure from the "boost_prediction" library,
//          which will be dumped to binary file
//
//      filename : "path to/name of" the binary file to export the prediction model
//
// RETURN VALUE : prediction model binary file size (in bytes) if success, negative error_code if fails
//
int dump_predictor(const predictor *const pred, const std::string &filename);

