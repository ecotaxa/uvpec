#include "xgboost_import.hpp"

#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace std;

// *** BEGIN OF : Internal use functions and data structures ***
enum node_type{
    BOOSTER = 0,
    SPLIT,
    LEAF,
    UNKNOWN
};

typedef struct {
    enum node_type type;
    int idx;
    float value;
    int feature;
    int yes_node;
} node;

typedef vector< node > tree_vec;
typedef vector< tree_vec > cat_vec;
typedef vector< cat_vec > pred_vec;

static bool node_comp(const node i, const node j) { return (i.idx < j.idx); }

static void get_node_info(const string line, node *N){

    int scan_counter, idx, feature_nb, yes_node;
    float value;

    scan_counter = sscanf(line.c_str(), " booster[%d]: ", &idx);
    if (scan_counter == 1) { // it's a "booster" node
        N->type = BOOSTER;
        N->idx = idx;
        N->value = NAN;
        N->feature = -1;
        N->yes_node = -1;
    } else {
        scan_counter = sscanf(line.c_str(), " %d:[f%d<%f] yes=%d ", &idx, &feature_nb, &value, &yes_node);
        if (scan_counter == 4) { // it's a "split" node
            N->type = SPLIT;
            N->idx = idx;
            N->value = value;
            N->feature = feature_nb;
            N->yes_node = yes_node;
        } else {
            scan_counter = sscanf(line.c_str(), " %d:leaf=%f ", &idx, &value);
            if (scan_counter == 2) { // it's a "leaf" node
                N->type = LEAF;
                N->idx = idx;
                N->value = value;
                N->feature = -1;
                N->yes_node = -1;
            } else { //unknown type of node
                N->type = UNKNOWN;
                N->idx = -1;
                N->value = NAN;
                N->feature = -1;
                N->yes_node = -1;
            }
        }
    }
}
// *** END OF : Internal use functions and data structures ***


// *** BEGIN OF : Functions exported by the "xgboost_import" library ***

// Loads a prediction model from a text file in XGBoost text export format (XGBoost dump_model function)
int load_xgboost_predictor(predictor *const pred, const std::string &filename, const uint8_t nb_categories){

    ifstream input_file;
    input_file.open(filename.c_str());
    if ( !input_file ) return -1;

    pred_vec V;
    pred_vec::iterator cat_it;
    cat_vec::reverse_iterator tree_it;
    V.clear();
    for (int k = 0; k < nb_categories; k++) {
        V.push_back(cat_vec());
        V[k].clear();
    }

    string line;
    node N;
    bool exit = false;
    bool error = false;
    cat_it = V.begin();

    do{
        getline(input_file, line);                      //get a line into text file
        if (!input_file.fail()){                        //verify if a line was correctly read
            if (line.size()!=0){
                get_node_info(line, &N);
                switch(N.type){
                    case BOOSTER :
                        cat_it->push_back(tree_vec());
                        tree_it = cat_it->rbegin();
                        tree_it->clear();
                        if (++cat_it == V.end()) cat_it = V.begin();
                    break;

                    case SPLIT :
                        tree_it->push_back(N);
                    break;

                    case LEAF :
                        tree_it->push_back(N);
                    break;

                    default :
                        exit = true;
                }
            }
        } else exit = true;
    } while (!input_file.eof() && !exit);

    if (exit && !input_file.eof()) error = true;
    input_file.close();
    if (error) return -2;

    int total_trees = 0;
    int total_nodes = 0;
    for (cat_it = V.begin(); cat_it != V.end(); cat_it++) {
        total_trees += cat_it->size();
        for (cat_vec::iterator t_it = cat_it->begin(); t_it != cat_it->end(); t_it++) {
            total_nodes += t_it->size();
            sort(t_it->begin(), t_it->end(), node_comp); // sorting each tree
        }
    }
    if ((total_trees % nb_categories) != 0) return -3;

    // allocating for tree pointer structures, number of nodes per tree buffer, and all cumulated split and leaf nodes
    const int node_size = sizeof(float)+(2*sizeof(uint8_t));
    int pred_buffer_size = total_trees*sizeof(decision_tree); // space for storing tree pointer structures (struct decision_tree)
    pred_buffer_size += total_trees*sizeof(uint32_t); // space for storing number of nodes per tree counters
    pred_buffer_size += (total_nodes+total_trees)*node_size; //space for storing all split and leaf nodes, plus a dummy root node 0 per tree

    uint8_t *predictor_buffer = (uint8_t*)malloc(pred_buffer_size);

    if (predictor_buffer != NULL){ // fill predictor structure info, fill data buffer with predictor's data, and set tree structures pointers

        decision_tree *tree;
        int tree_idx;
        int data_offset = 0;
        int nb_nodes = 0;

        pred->category_nb = (uint32_t)nb_categories;
        pred->trees_per_cat = (uint32_t)(total_trees/nb_categories);
        pred->total_nb_nodes = (uint32_t)(total_nodes + total_trees); // we need to add one dummy root node 0 per tree

        pred->trees = (decision_tree*)predictor_buffer;
        data_offset += total_trees*sizeof(decision_tree);

        pred->nodes_in_tree = (uint32_t*)(predictor_buffer + data_offset);
        data_offset += total_trees*sizeof(uint32_t);

        for (uint32_t i = 0; i < pred->category_nb; i++) {
            for (uint32_t j = 0; j < pred->trees_per_cat; j++) {
                tree_idx = (i*pred->trees_per_cat) + j;
                tree = &(pred->trees[tree_idx]);
                nb_nodes = (V[i][j].size() + 1); // each tree has a dummy root node 0, so nb_nodes = real_nb_nodes+1

                pred->nodes_in_tree[tree_idx] = nb_nodes;
                // Setting pointers to predictor's data segment for this tree
                tree->value = (float*)(predictor_buffer+data_offset);
                tree->idx = (uint8_t*)(tree->value + nb_nodes);
                tree->next = (uint8_t*)(tree->idx + nb_nodes);
                data_offset += nb_nodes*node_size;

                // Filling tree nodes
                // Root node 0 is a dummy leaf node
                tree->value[0] = NAN;
                tree->idx[0] = 255;
                tree->next[0] = 0;

                node *N_p;
                for (int k = 1; k < nb_nodes; k++) {
                    N_p = &(V[i][j][k-1]);      // indexes in the vector are indexes in our predictor minus 1 (dummy node consequence)
                    switch(N_p->type){
                        case SPLIT :
                            tree->value[k] = N_p->value;
                            tree->idx[k] = N_p->feature;
                            tree->next[k] = (N_p->yes_node + 1);
                        break;

                        case LEAF :
                            tree->value[k] = N_p->value;
                            tree->idx[k] = 255;
                            tree->next[k] = 0;
                        break;

                        default :
                            error = true;
                    }
                    if (error) break;
                }
                if (error) break;
            }
            if (error) break;
        }
    } else return -4;

    if (error){ // if failed, release memory and clean predictor structure
        pred->category_nb = 0;
        pred->trees_per_cat = 0;
        pred->total_nb_nodes = 0;
        free(pred->trees);
        pred->trees = NULL;
        pred->nodes_in_tree = NULL;
        return -5;
    } else return 0;
}

// Creates a binary file containing the prediction model, exportable to UVP6
int dump_predictor(const predictor *const pred, const std::string &filename){

    FILE *f;
    int byte_counter;
    size_t c;

	f = fopen(filename.c_str(), "wb");
    if (f == NULL) return -1;

	c = fwrite(pred, sizeof(uint32_t), 3, f);
	if (c == 3){
        byte_counter = 3*sizeof(uint32_t);

        const int node_size = sizeof(float)+(2*sizeof(uint8_t));
        size_t pred_buffer_size = pred->category_nb*pred->trees_per_cat*sizeof(uint32_t); // space for storing number of nodes per tree counters
        pred_buffer_size += pred->total_nb_nodes*node_size; //space for storing all split and leaf nodes, plus a dummy root node 0 per tree

        c = fwrite(pred->nodes_in_tree, 1, pred_buffer_size, f);
        if (c == pred_buffer_size) byte_counter+=pred_buffer_size;
        else byte_counter = -3;
    } else byte_counter = -2;

    fclose (f);
    return byte_counter;
}
// *** END OF : Functions exported by the "xgboost_import" library ***
