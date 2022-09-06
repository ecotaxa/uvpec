#include "boost_prediction.hpp"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cfloat>

// *** BEGIN OF : Functions exported by the "boost_prediction" library ***

// Loads a predictor from a binary file containing a prediction model in UVP6 format
// (previously created using the "dump_predictor" function from "xgboost_import" library
int load_predictor(predictor *const pred, const char *const filename){

    FILE *f;
    f = fopen(filename, "rb");
    if (f == NULL) return -1;

    // Reading predictor data from file
	size_t c;
    int byte_counter;

	uint32_t pred_dimensions[3]; // 0-> nb_categories, 1-> trees_per_category, 2 -> total_nb_nodes
	int nb_trees;
	uint8_t *predictor_buffer = NULL;
	size_t data_offset;
	const int node_size = sizeof(float)+(2*sizeof(uint8_t));

	c = fread(pred_dimensions, sizeof(uint32_t), 3, f);
	if (c == 3){
        byte_counter = 3*sizeof(uint32_t);

        // allocating buffer for tree pointer structures, number of nodes per tree buffer, and all cumulated split and leaf nodes
        nb_trees = pred_dimensions[0]*pred_dimensions[1];
        data_offset = nb_trees*sizeof(decision_tree); // space for storing tree pointer structures (struct decision_tree)
        size_t data_size = nb_trees*sizeof(uint32_t); // space for storing number of nodes per tree counters
        data_size += pred_dimensions[2]*node_size; //space for storing all split and leaf nodes, plus a dummy root node 0 per tree
        size_t pred_buffer_size = data_offset + data_size;

        predictor_buffer = (uint8_t*)malloc(pred_buffer_size);

        if (predictor_buffer != NULL){
            c = fread((predictor_buffer + data_offset), 1, data_size, f);
            if (c == data_size) {
                byte_counter += data_size;
                // test if there's any data remaining in the file (corrupted file case)
                uint8_t dummy_read;
                c = fread(&dummy_read, 1, 1, f);
                if ((c!=0) || (feof(f)==0)) byte_counter = -5;
            } else byte_counter = -4;
        } else byte_counter = -3;
	} else byte_counter = -2;
	fclose (f);

	if (byte_counter > 0) { // successful read, fill predictor structure info and set tree structures pointers

        decision_tree *tree;
        int tree_idx;
        int nb_nodes;

        pred->category_nb = pred_dimensions[0];
        pred->trees_per_cat = pred_dimensions[1];
        pred->total_nb_nodes = pred_dimensions[2]; // we need to add one dummy root node 0 per tree
        pred->trees = (decision_tree*)predictor_buffer;
        pred->nodes_in_tree = (uint32_t*)(predictor_buffer + data_offset);
        data_offset += nb_trees*sizeof(uint32_t);

        for (uint32_t i = 0; i < pred->category_nb; i++) {
            for (uint32_t j = 0; j < pred->trees_per_cat; j++) {
                tree_idx = (i*pred->trees_per_cat) + j;
                tree = &(pred->trees[tree_idx]);
                nb_nodes = pred->nodes_in_tree[tree_idx];
                // Setting pointers to predictor's data segment for this tree
                tree->value = (float*)(predictor_buffer+data_offset);
                tree->idx = (uint8_t*)(tree->value + nb_nodes);
                tree->next = (uint8_t*)(tree->idx + nb_nodes);
                data_offset += nb_nodes*node_size;
            }
        }
	} else { // read error, release predictor_buffer
        free(predictor_buffer);
	}

    return byte_counter;
}


// Releases data buffers and clear the predictor structure, which can be discarded or
// re-used to load a new model, avoiding memory leak
void release_predictor(predictor *const pred){
    pred->category_nb = 0;
    pred->trees_per_cat = 0;
    pred->total_nb_nodes = 0;
    free(pred->trees);
    pred->trees = NULL;
    pred->nodes_in_tree = NULL;
}


// Performs automatic classification for a sample, using a prediction model previously loaded in
// the predictor structure, and an input features vector
void predict(const predictor *const pred, const float *const input_vec, uint32_t *const best_pred_id, float *const best_pred_score, float *const pred_scores_per_cat){

    float acc_val, exp_acc_val, acc_exp, best_score;
    uint32_t node, node_max, best_pred;
	uint8_t feat_idx;
	float val;
	int tree_idx;
	decision_tree *tree;

	// if we need to keep all prediction scores for debugging or test purposes
	bool keep_all_scores = (pred_scores_per_cat != NULL);

	acc_exp = 0;
	best_pred = 0;
	best_score = 0;
    for (uint32_t i = 0; i < pred->category_nb; i++) {
    	acc_val = 0;
		for (uint32_t j = 0; j < pred->trees_per_cat; j++) {
            tree_idx = (i*pred->trees_per_cat) + j;
			tree = &(pred->trees[tree_idx]);
			node_max = pred->nodes_in_tree[tree_idx];
			node = 1;
			while(node < node_max){
				feat_idx = tree->idx[node];
				val = tree->value[node];
				if (feat_idx < 255) {
					node = (uint32_t)tree->next[node];
					if (input_vec[feat_idx] >= val) node++;
				} else break;
			}
			if (node >= node_max) val = NAN;
			acc_val += val;
		}
		exp_acc_val = exp(acc_val);
        acc_exp += exp_acc_val;
        if (exp_acc_val > best_score) {
            best_pred = i;
            best_score = exp_acc_val;
        }
        if (keep_all_scores) pred_scores_per_cat[i] = exp_acc_val;
    }

    if (best_pred_id != NULL) *best_pred_id = best_pred;
    if (best_pred_score != NULL) *best_pred_score = best_score/acc_exp;
    if (keep_all_scores){
    	float R = 1.0/acc_exp;
        for (uint32_t i = 0; i < pred->category_nb; i++) pred_scores_per_cat[i] *= R;
    }
}
// *** END OF : Functions exported by the "boost_prediction" library ***
