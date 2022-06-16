import os
import pandas as pd
  
def create_taxoconf(output_dir, dict_id, MODEL_REF, key):
    """
    Function that creates a taxonomic configuration file for the UVP6.
    """
    
    # check if dict exists
    if type(dict_id) == dict:
        print('OK.')
    else:
        raise TypeError("Dictionnary with EcoTaxa's IDs does not exist.") 
    
    # prepation of the taxo configuration file
    print('Create TAXOCONF file')
    
    # number of classes in the current model
    n_classes = len(dict_id)
    classes = list(dict_id.keys()) # list of all classes in the CURRENT model
    
    # name of the taxo configuration file
    TAXOCONFNAME = 'TAXO_NKE_0'

    # creation of the header
    f = open(os.path.join(output_dir, "TAXO_NKE_0_"+key+".txt"), "w")
    f.write("// Taxonomic classification configuration parameters : \n\n// Parameter name: Configuration_name, String type, length range: 1 to 15 characters\n// Description: Name used to set this configuration into the field Taxo_conf, from the Acquisition configuration file\n// Can be modified for compatibility with different vector drivers\nConfiguration_name = "+TAXOCONFNAME+"\n\n// Parameter name: Model_reference, String type, length range: 1 to 15 characters\n// Description: Taxonomic classification model to be used. Automatically filled during model creation/export, do not edit\nModel_reference = "+MODEL_REF+"\n\n// Parameter name: Max_size_for_classification, Integer type, range: 0 to 65535\n// Description: Maximum vignette size to perform the embedded classification - Unit: pixels\nMax_size_for_classification = 65535\n\n// Parameter name: Model_nb_classes, Integer type, range: 2 to 40\n// Description: Number of classes used by the classification model. Automatically filled during model creation/export, do not edit\nModel_nb_classes = "+str(n_classes)+"\n\n")
    f.close()

    # append data for classes
    for i in range(n_classes):
        f = open(os.path.join(output_dir, "TAXO_NKE_0_"+key+".txt"), "a")
        f.write("// Parameter name: Taxo_ID_for_class_"+f"{i:02d}"+", Integer type, range: 0 to 9999999\n// Description: Taxonomic unique identifier for model's class "+f"{i:02d}"+". Automatically filled during model creation/export, do not edit\n// Only ID's for classes up to (Model_nb_classes - 1) must be declared, others will be automatically set to zero\nTaxo_ID_for_class_"+f"{i:02d}"+" = "+str(dict_id[classes[i]])+"\n\n")
        f.close()

    print('Done.')
