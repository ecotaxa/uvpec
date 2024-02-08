import numpy as np
import pandas as pd
import os
import glob
from os import environ
import xgboost as xgb
from numpy import asarray, argmax
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, pairwise_distances
from uvpec.custom import label_to_int, int_to_label
from cython_uvp6 import py_load_model_and_predict
import array

def evaluate_model(n_jobs, df_test, xgb_model, inflexion_filename, use_inflexion, output_dir, use_C):
    
    # extract key (model identifier)
    key = xgb_model.split('_')[-1].split('.')[0]
    #print(key)

    # create a dict() to convert labels to int (for xgboost)
    dico_label = {}
    for i, arg in enumerate(np.unique(df_test['labels'])):
        dico_label[arg] = i
    
    if use_C is True:
        # technically it is not needed if we are only interested in the predicted classes. 
        nb_classes = len(dico_label)

        # function to send features to C module
        def send_features(i, data):
            ft = data.iloc[i,0:55].to_numpy()
            ft = array.array('f', ft) # need to convert numpy type to float array (see cython code .pyx)
            return(ft)
        
        # name of binary model
        xgb_model_binary_bytes = bytes(os.path.join(output_dir,'Muvpec_'+key), encoding ='utf-8') # need to convert string to bytes
        # init predicted labels
        pred_class_idx = list()
        # load model and predict
        for i in range(df_test.shape[0]):
            predicted_label, _, _ = py_load_model_and_predict(xgb_model_binary_bytes, send_features(i, df_test), nb_classes)
            pred_class_idx.append(predicted_label)
        
        # predicted and true classes
        predicted_classes = int_to_label(dico_label, np.array(pred_class_idx))
        true_classes = np.array(df_test['labels'])

    else:
        # create the DMatrix for the test set
        y_test = label_to_int(dico_label, df_test['labels'])
        df_test = df_test.iloc[:,0:len(df_test.columns)-1]
        dtest = xgb.DMatrix(df_test, label=y_test)

        # Load best xgboost model for the settings provided by the user
        bst = xgb.Booster({'nthread': n_jobs})  # init model
        bst.load_model(xgb_model)
    
        # make prediction
        preds = bst.predict(dtest)

        # get the best prediction
        pred_class_idx = np.argmax(preds, axis=1)
        true_classes = int_to_label(dico_label, y_test)
        predicted_classes = int_to_label(dico_label, pred_class_idx)

    # get index of biological classes to compute come biological scores
    non_biological_classes = ['detritus','artefact','crystal','fiber','filament','reflection']
    living_classes = np.unique([s for i, s in enumerate(true_classes) if s not in non_biological_classes])

    # print some metrics
    accuracy = accuracy_score(true_classes, predicted_classes)
    balanced_accuracy = balanced_accuracy_score(true_classes, predicted_classes)

    # macro avg
    living_precision = precision_score(true_classes, predicted_classes, labels=living_classes, average='macro', zero_division=0)
    living_recall = recall_score(true_classes, predicted_classes, labels=living_classes, average='macro', zero_division=0)
    living_f1 = f1_score(true_classes, predicted_classes, labels=living_classes, average='macro', zero_division=0)

    # weighted avg
    living_precision_w = precision_score(true_classes, predicted_classes, labels=living_classes, average='weighted', zero_division=0)
    living_recall_w = recall_score(true_classes, predicted_classes, labels=living_classes, average='weighted', zero_division=0)
    living_f1_w = f1_score(true_classes, predicted_classes, labels=living_classes, average='weighted', zero_division=0)
    
    print('Here are some classification scores')
    print(f'Accuracy score is {accuracy}')
    print(f'Balanced accuracy score is {balanced_accuracy}')
    print(f'Macro living precision score is {living_precision}')
    print(f'Macro living recall is {living_recall}')
    print(f'Macro living f1 score is {living_f1}')
    print(f'Weighted living precision score is {living_precision_w}')
    print(f'Weighted living recall is {living_recall_w}')
    print(f'Weighted living f1 score is {living_f1_w}')

    if use_inflexion:
        # logloss plot (obtained after cross-validation)
        dataCV = pd.read_feather(inflexion_filename)
        dataCV = dataCV.reset_index()
        ax = dataCV.plot(x = 'index', y = 'test-mlogloss-mean')
        ax.set_xlabel('number of boosting rounds')
        fig = ax.get_figure()
        #fig.suptitle(inflexion_filename)
        fig.savefig(os.path.join(output_dir,'logloss_'+str(key)+'.jpg'))
        #fig.close()

    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes, normalize='true')
    classes = np.unique(true_classes)
    # Plot it
    plt.figure(figsize=(20,20))
    plt.imshow(cm, cmap='Greys')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    #plt.show()
    plt.savefig(os.path.join(output_dir,'Muvpec_'+str(key)+'_confusion_matrix.jpg'))
    plt.close()

    # Classification report
    classif_report = classification_report(true_classes, predicted_classes, output_dict=True)
    classif_report = pd.DataFrame(classif_report).transpose()

    # add living classes stats
    living_stats_df = pd.DataFrame(np.array([[living_precision, living_precision_w],[living_recall, living_recall_w],[living_f1, living_f1_w]]), index = ['precision', 'recall', 'f1-score'], columns = ['living macro avg', 'living weighted avg'])
    living_stats_df = living_stats_df.transpose()

    classif_report = pd.concat([classif_report, living_stats_df])

    # remove unused data
    classif_report.loc['weighted avg','support'] = np.nan
    classif_report.loc['macro avg','support'] = np.nan
    classif_report.loc['accuracy','support'] = np.nan

    classes = np.unique(true_classes)
    # List annotations for figure
    annot = [str(x) for x in classes]
    annot.extend(("accuracy", "macro avg", "weighted avg", "living macro avg","living weighted avg"))

    # make figure
    plt.figure(figsize = (20,20))
    sn.heatmap(classif_report, annot=True, vmin=0, vmax=1.0, yticklabels = annot, cmap="viridis")
    plt.savefig(os.path.join(output_dir,'Muvpec_'+str(key)+'_classif_report.jpg'))
    plt.close()
