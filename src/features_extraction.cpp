#include "features_extraction.hpp"

#include <cmath>
#include <cstring>
#include <vector>
#include <cstdlib>

using namespace std;

typedef struct {
    uint16_t x;
    uint16_t y;
} Coord;

void get_features(const uint8_t *const input, const Thumbnail *const T, const uint16_t input_row_stride, const uint8_t threshold, float *const features){

	// Get ROI dimensions and create a 0 initialized buffer to hold the mask
	const uint16_t rows = T->height;
	const uint16_t cols = T->width;
	uint8_t *const mask_buffer = (uint8_t*)calloc((size_t)rows*cols, sizeof(uint8_t));	

	// *** BEGIN OF : Getting mask using watershed algorithm ***
	{
		const uint8_t *const input_base = input + (T->y_min*input_row_stride) + T->x_min; // First ROI pixel pointer
		const uint16_t bottom_limit = rows - 1;
		const uint16_t right_limit = cols-1;

		uint8_t *mask_ptr;
		uint8_t pix_val;
		Coord S, N; // S coords are for Seed pixel, and N coords for its Neighbors
		vector< Coord > clist;

		clist.clear();

		// Get first seed position from segmentation result (x_obj, y_obj -> topmost left active pixel from the object), and convert into ROI referenced coordinates
		S.x = T->x_obj - T->x_min;
		S.y = T->y_obj - T->y_min;

		// Test seed, update mask, and put seed into the checklist (clist) to verify its neighbors
		pix_val = *(input_base + (S.y * input_row_stride) + S.x);
		if (pix_val > threshold){
			*(mask_buffer + (S.y * cols) + S.x) = pix_val;
			clist.push_back(S);
		}

		// Watershed propagation loop (unrolled for all neighbors)
		while (!clist.empty()){

			// get last pixel in the list and remove it from the list
			S = clist.back();
			clist.pop_back();

			// Verify each pixel neighbor (inside the ROI borders), and if it hasn't been tested yet (mask = 0), update mask and checklist
			// Active (foreground) pixels belonging to the object keep their graylevel value in the mask,
			// tested background pixels get the threshold value to differentiate from untested pixels (0)

			if (S.y > 0){ // We're not in the uppermost row, so there are upper neighbors to verify
				N.y = S.y - 1;

				if (S.x > 0){ // We're not in the leftmost column, so there is an upper left neighbor to verify
					N.x = S.x - 1;
					mask_ptr = mask_buffer + (N.y * cols) + N.x;
					if (*mask_ptr == 0){
						pix_val = *(input_base + (N.y * input_row_stride) + N.x);
						if (pix_val > threshold){
							*mask_ptr = pix_val;
							clist.push_back(N);
						} else *mask_ptr = threshold;
					}
				}

				// Upper neighbor verification
				N.x = S.x;
				mask_ptr = mask_buffer + (N.y * cols) + N.x;
				if (*mask_ptr == 0){
					pix_val = *(input_base + (N.y * input_row_stride) + N.x);
					if (pix_val > threshold){
						*mask_ptr = pix_val;
						clist.push_back(N);
					} else *mask_ptr = threshold;
				}

				if (S.x < right_limit){ // We're not in the rightmost column, so there is an upper right neighbor to verify
					N.x = S.x + 1;
					mask_ptr = mask_buffer + (N.y * cols) + N.x;
					if (*mask_ptr == 0){
						pix_val = *(input_base + (N.y * input_row_stride) + N.x);
						if (pix_val > threshold){
							*mask_ptr = pix_val;
							clist.push_back(N);
						} else *mask_ptr = threshold;
					}
				}
			}

			N.y = S.y;

			if (S.x > 0){ // We're not in the leftmost column, so there is a left neighbor to verify
				N.x = S.x - 1;
				mask_ptr = mask_buffer + (N.y * cols) + N.x;
				if (*mask_ptr == 0){
					pix_val = *(input_base + (N.y * input_row_stride) + N.x);
					if (pix_val > threshold){
						*mask_ptr = pix_val;
						clist.push_back(N);
					} else *mask_ptr = threshold;
				}
			}

			if (S.x < right_limit){ // We're not in the rightmost column, so there is a right neighbor to verify
				N.x = S.x + 1;
				mask_ptr = mask_buffer + (N.y * cols) + N.x;
				if (*mask_ptr == 0){
					pix_val = *(input_base + (N.y * input_row_stride) + N.x);
					if (pix_val > threshold){
						*mask_ptr = pix_val;
						clist.push_back(N);
					} else *mask_ptr = threshold;
				}
			}

			if (S.y < bottom_limit){ // We're not in the bottom most row, so there are bottom neighbors to verify
				N.y = S.y + 1;

				if (S.x > 0){ // We're not in the leftmost column, so there is a bottom left neighbor to verify
					N.x = S.x - 1;
					mask_ptr = mask_buffer + (N.y * cols) + N.x;
					if (*mask_ptr == 0){
						pix_val = *(input_base + (N.y * input_row_stride) + N.x);
						if (pix_val > threshold){
							*mask_ptr = pix_val;
							clist.push_back(N);
						} else *mask_ptr = threshold;
					}
				}

				// Bottom neighbor verification
				N.x = S.x;
				mask_ptr = mask_buffer + (N.y * cols) + N.x;
				if (*mask_ptr == 0){
					pix_val = *(input_base + (N.y * input_row_stride) + N.x);
					if (pix_val > threshold){
						*mask_ptr = pix_val;
						clist.push_back(N);
					} else *mask_ptr = threshold;
				}

				if (S.x < right_limit){ // We're not in the rightmost column, so there is a bottom right neighbor to verify
					N.x = S.x + 1;
					mask_ptr = mask_buffer + (N.y * cols) + N.x;
					if (*mask_ptr == 0){
						pix_val = *(input_base + (N.y * input_row_stride) + N.x);
						if (pix_val > threshold){
							*mask_ptr = pix_val;
							clist.push_back(N);
						} else *mask_ptr = threshold;
					}
				}
			}
		}
	}
	// *** END OF : Getting mask using watershed algorithm ***

	// Allocate and initialize arrays to hold the moments and histogram
	uint64_t moments[20] = {0};
    uint32_t hist[256] = {0};

	// *** BEGIN OF : Calculating moments and histogram on the previously calculated mask ***
    {
		uint8_t p;
		uint64_t x, y, t1, t2, t3, t4, mt1, mt2, mt3, mt4;
		bool calculate;
		const uint8_t *pix, *line;

		//Optimized using polynomial expansion up to the 3th order for loop unroll (4 pixels are calculated at once)
		for (y=0, line=mask_buffer; y < rows; y++, line+=cols){
			for (x=0, pix=line; x < cols; x+=4, pix+=4){

				p = *pix;
				if (p > threshold) {
					hist[p]++;
					t1 = p; mt1 = 1;
					calculate = true;
				} else {
					t1 = 0; mt1 = 0;
					calculate = false;
				}

				// These tests are necessary in order to manage for image widths which are not modulo 4
				if ((x+1) < cols) {
					p = *(pix+1);
					if (p > threshold){
						hist[p]++;
						t1 += p; t2 = p; t3 = p; t4 = p; mt1++; mt2 = 1; mt3 = 1; mt4 = 1;
						calculate = true;
					} else {
						t2 = 0; t3 = 0; t4 = 0; mt2 = 0; mt3 = 0; mt4 = 0;
					}
					if ((x+2) < cols) {
						p = *(pix+2);
						if (p > threshold){
							hist[p]++;
							t1 += p; t2 += 2*p; t3 += 4*p; t4 += 8*p; mt1++; mt2 += 2; mt3 += 4; mt4 += 8;
							calculate = true;
						}
						if ((x+3) < cols) {
							p = *(pix+3);
							if (p > threshold){
								hist[p]++;
								t1 += p; t2 += 3*p; t3 += 9*p; t4 += 27*p; mt1++; mt2 += 3; mt3 += 9; mt4 += 27;
								calculate = true;
							}
						}
					}
				} else {
					t2 = 0; t3 = 0; t4 = 0; mt2 = 0; mt3 = 0; mt4 = 0;
				}

				if (calculate){ // Update moments accumulators
					moments[0] += mt1;                                   //M_0_0
					moments[1] += (x*mt1)+mt2;                           //M_1_0
					moments[2] += y*mt1;                                 //M_0_1
					moments[3] += (x*x*mt1)+(2*x*mt2)+mt3;               //M_2_0
					moments[4] += y*((x*mt1)+mt2);                       //M_1_1
					moments[5] += y*y*mt1;                               //M_0_2
					moments[6] += (x*x*x*mt1)+(3*x*x*mt2)+(3*x*mt3)+mt4; //M_3_0
					moments[7] += y*((x*x*mt1)+(2*x*mt2)+mt3);           //M_2_1
					moments[8] += y*y*((x*mt1)+mt2);                     //M_1_2
					moments[9] += y*y*y*mt1;                             //M_0_3

					moments[10] += t1;                                   //M_weighted_0_0
					moments[11] += (x*t1)+t2;                            //M_weighted_1_0
					moments[12] += y*t1;                                 //M_weighted_0_1
					moments[13] += (x*x*t1)+(2*x*t2)+t3;                 //M_weighted_2_0
					moments[14] += y*((x*t1)+t2);                        //M_weighted_1_1
					moments[15] += y*y*t1;                               //M_weighted_0_2
					moments[16] += (x*x*x*t1)+(3*x*x*t2)+(3*x*t3)+t4;    //M_weighted_3_0
					moments[17] += y*((x*x*t1)+(2*x*t2)+t3);             //M_weighted_2_1
					moments[18] += y*y*((x*t1)+t2);                      //M_weighted_1_2
					moments[19] += y*y*y*t1;                             //M_weighted_0_3
				}
			}
		}
    }
    // *** END OF : Calculating moments and histogram on the previously calculated mask ***

    // Release allocated mask buffer
    free(mask_buffer);

    // *** BEGIN OF : Calculating features from moments, histogram and bounding box (T) properties ***
    {
    	// Calculations are performed in double precision (when necessary), and final result is casted as float
    	double nb_pixels = moments[0];
		features[0] = (float)nb_pixels;

		double width = (double)T->width;
		double height = (double)T->height;
		features[1] = (float)width;
		features[2] = (float)height;

		double intden = moments[10];
		double mean = intden/nb_pixels;
		features[3] = (float)mean;

		uint8_t vmin = 0, vmax = 0, mode = 0, median = 0, histcum1 = 0, histcum3 = 0;
		uint32_t mode_max = 0, pix_acc = 0, cnt;
		uint64_t square_gray_acc = 0;
		float first_quartile = 0.25 * nb_pixels;
		float second_quartile = 0.5 * nb_pixels;
		float third_quartile = 0.75 * nb_pixels;
		for (uint32_t i = 0; i < 256; i++){
			cnt = hist[i];
			if (cnt > 0){
				if (vmin == 0) vmin = (uint8_t)i;
				vmax = (uint8_t)i;
				if (cnt > mode_max){mode_max = cnt; mode = (uint8_t)i;}
				pix_acc += cnt;
				if ((histcum1 == 0) && (pix_acc > first_quartile)) histcum1 = (uint8_t)i;
				if ((median == 0) && (pix_acc > second_quartile)) median = (uint8_t)i;
				if ((histcum3 == 0) && (pix_acc > third_quartile)) histcum3 = (uint8_t)i;
				square_gray_acc += (uint64_t)cnt*(uint64_t)i*(uint64_t)i;
			}
		}

		double stddev = sqrt(((double)square_gray_acc/nb_pixels) - (mean * mean));
		features[4] = (float)stddev;
		features[5] = (float)mode;
		features[6] = (float)vmin;
		features[7] = (float)vmax;

		double u1 = 1.0/nb_pixels;
		double xc = u1*moments[1];
		double yc = u1*moments[2];
		features[8] = (float)xc;
		features[9] = (float)yc;

		double gray_u1 = 1.0/intden;
		double gray_xc = gray_u1*moments[11];
		double gray_yc = gray_u1*moments[12];
		features[10] = (float)gray_xc;
		features[11] = (float)gray_yc;

		double u20 = (double)moments[3] - (xc*moments[1]);      //µ_2_0
		double u11 = (double)moments[4] - (xc*moments[2]);      //µ_1_1
		double u02 = (double)moments[5] - (yc*moments[2]);      //µ_0_2
		double sub = u20 - u02;
		double add = u20 + u02;
		double delta = sqrt((4*u11*u11) + (sub*sub));
		double divider = u1/2;
		double major_eigval = (add+delta)*divider;
		double minor_eigval = (add-delta)*divider;
		features[12] = (float)(4 * sqrt(major_eigval));
		features[13] = (float)(4 * sqrt(minor_eigval));
		features[14] = (float)(0.5 * atan2(2*u11, sub));
		features[15] = (float)(sqrt(1-(minor_eigval/major_eigval)));

		features[16] = (float)intden;
		features[17] = (float)median;
		features[18] = (float)histcum1;
		features[19] = (float)histcum3;

		features[20] = (float)(sqrt(4*nb_pixels/M_PI));

		double vrange = vmax - vmin;
		features[21] = (float)vrange;
		features[22] = (float)((mean - vmin)/vrange);
		features[23] = (float)(100*(stddev/mean));
		features[24] = (float)(100*(stddev/vrange));

		double boxarea = width*height;
		features[25] = (float)boxarea;
		features[26] = (float)(nb_pixels/boxarea);

		double u30 = (double)moments[6] - (3*xc*moments[3]) + (2*xc*xc*moments[1]);                    //µ_3_0
		double u21 = (double)moments[7] - (2*xc*moments[4]) - (yc*moments[3]) + (2*xc*xc*moments[2]);  //µ_2_1
		double u12 = (double)moments[8] - (2*yc*moments[4]) - (xc*moments[5]) + (2*yc*yc*moments[1]);  //µ_1_2
		double u03 = (double)moments[9] - (3*yc*moments[5]) + (2*yc*yc*moments[2]);                    //µ_0_3

		features[27] = (float)u20;
		features[28] = (float)u11;
		features[29] = (float)u02;
		features[30] = (float)u30;
		features[31] = (float)u21;
		features[32] = (float)u12;
		features[33] = (float)u03;

		double u2 = u1*u1;
		double u3 = u2*sqrt(u1);

		double n20 = u20*u2;   //n_2_0
		double n11 = u11*u2;   //n_1_1
		double n02 = u02*u2;   //n_0_2
		double n30 = u30*u3;   //n_3_0
		double n21 = u21*u3;   //n_2_1
		double n12 = u12*u3;   //n_1_2
		double n03 = u03*u3;   //n_0_3

		double poly11 = 4*n11;                // 4n11
		double poly20m = n20 - n02;           // n20 - n02
		double poly30m = n30 - (3*n12);       // n30 - 3n12
		double poly21m = (3*n21) - n03;       // 3n21 - n03
		double poly30p = n30 + n12;           // n30 + n12
		double poly21p = n21 + n03;           // n21 + n03
		double poly30p_2 = poly30p*poly30p;   //(n30 + n12)^2
		double poly21p_2 = poly21p*poly21p;   //(n21 + n03)^2

		features[34] = (float)(n20 + n02);                              // Hu0, I1 - n20+n02
		features[35] = (float)((poly20m*poly20m) + (n11*poly11));       // Hu1, I2 - (n20 - n02)^2 + 4n11^2
		features[36] = (float)((poly30m*poly30m) + (poly21m*poly21m));  // Hu2, I3 - (n30 - 3n12)^2 + (3n21 - n03)^2
		features[37] = (float)(poly30p_2 + poly21p_2);                  // Hu3, I4 - (n30 + n12)^2 + (n21 +n03)^2
		features[38] = (float)((poly30m*poly30p*(poly30p_2 - (3*poly21p_2))) + (poly21m*poly21p*((3*poly30p_2) - poly21p_2)));  // Hu4, I5
		features[39] = (float)((poly20m*(poly30p_2 - poly21p_2)) + (poly11*poly30p*poly21p));                                   // Hu5, I6
		features[40] = (float)((poly21m*poly30p*(poly30p_2 - (3*poly21p_2))) - (poly30m*poly21p*((3*poly30p_2) - poly21p_2)));  // Hu6, I7

		u20 = (double)moments[13] - (gray_xc*moments[11]);      //µ_2_0
		u11 = (double)moments[14] - (gray_xc*moments[12]);      //µ_1_1
		u02 = (double)moments[15] - (gray_yc*moments[12]);      //µ_0_2
		u30 = (double)moments[16] - (3*gray_xc*moments[13]) + (2*gray_xc*gray_xc*moments[11]);                          //µ_3_0
		u21 = (double)moments[17] - (2*gray_xc*moments[14]) - (gray_yc*moments[13]) + (2*gray_xc*gray_xc*moments[12]);  //µ_2_1
		u12 = (double)moments[18] - (2*gray_yc*moments[14]) - (gray_xc*moments[15]) + (2*gray_yc*gray_yc*moments[11]);  //µ_1_2
		u03 = (double)moments[19] - (3*gray_yc*moments[15]) + (2*gray_yc*gray_yc*moments[12]);                          //µ_0_3

		features[41] = (float)u20;
		features[42] = (float)u11;
		features[43] = (float)u02;
		features[44] = (float)u30;
		features[45] = (float)u21;
		features[46] = (float)u12;
		features[47] = (float)u03;

		u2 = gray_u1*gray_u1;
		u3 = u2*sqrt(gray_u1);
		n20 = u20*u2;   //n_2_0
		n11 = u11*u2;   //n_1_1
		n02 = u02*u2;   //n_0_2
		n30 = u30*u3;   //n_3_0
		n21 = u21*u3;   //n_2_1
		n12 = u12*u3;   //n_1_2
		n03 = u03*u3;   //n_0_3
		poly11 = 4*n11;                // 4n11
		poly20m = n20 - n02;           // n20 - n02
		poly30m = n30 - (3*n12);       // n30 - 3n12
		poly21m = (3*n21) - n03;       // 3n21 - n03
		poly30p = n30 + n12;           // n30 + n12
		poly21p = n21 + n03;           // n21 + n03
		poly30p_2 = poly30p*poly30p;   //(n30 + n12)^2
		poly21p_2 = poly21p*poly21p;   //(n21 + n03)^2

		features[48] = (float)(n20 + n02);                              // Hu0, I1 - n20+n02
		features[49] = (float)((poly20m*poly20m) + (n11*poly11));       // Hu1, I2 - (n20 - n02)^2 + 4n11^2
		features[50] = (float)((poly30m*poly30m) + (poly21m*poly21m));  // Hu2, I3 - (n30 - 3n12)^2 + (3n21 - n03)^2
		features[51] = (float)(poly30p_2 + poly21p_2);                  // Hu3, I4 - (n30 + n12)^2 + (n21 + n03)^2
		features[52] = (float)((poly30m*poly30p*(poly30p_2 - (3*poly21p_2))) + (poly21m*poly21p*((3*poly30p_2) - poly21p_2)));  // Hu4, I5
		features[53] = (float)((poly20m*(poly30p_2 - poly21p_2)) + (poly11*poly30p*poly21p));                                   // Hu5, I6
		features[54] = (float)((poly21m*poly30p*(poly30p_2 - (3*poly21p_2))) - (poly30m*poly21p*((3*poly30p_2) - poly21p_2)));  // Hu6, I7
    }
    // *** END OF : Calculating features from moments, histogram and bounding box (T) properties ***
}
