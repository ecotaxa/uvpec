#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

//
// Structure Thumbnail describes a ROI in the input image by its origin (x_min, y_min), its dimensions (width, height),
//		and gives a hint about the localization of an object inside this ROI (x_obj, y_obj).
//		All coordinates are given in reference to the input image
//
typedef struct {
	uint16_t x_min;
	uint16_t y_min;
	uint16_t width;
	uint16_t height;
	uint16_t x_obj;
	uint16_t y_obj;
} Thumbnail;

//
// get_features function parameters :
//
// input : pointer to the first pixel (top-left) in the data buffer containing the image/vignette to analyze
//
// T  : Thumbnail structure filled with information from an ROI containing the object to extract features.
//		The ROI can cover the whole image/vignette (x_min = 0, y_min = 0, width = input_width, height = input_height),
//		but x_obj and y_obj MUST point to a pixel belonging to the object from which the features will be extracted
//
// input_row_stride : bytes per row in the data buffer. Usually equal to input_width if no padding
//
// threshold : values <= threshold are considered as background pixels, only values above the threshold are taken into account for features calculation
//
// features : pointer to an array of floats (minimum size 55) where the extracted features will be stored as the output of the function
//
void get_features(const uint8_t *const input, const Thumbnail *const T, const uint16_t input_row_stride, const uint8_t threshold, float *const features);

#ifdef __cplusplus
}
#endif
