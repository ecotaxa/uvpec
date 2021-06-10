import os
import pandas as pd
  
def create_taxoconf(output_dir, dict_classes, MODEL_REF, key):
    """
    Function that creates a taxonomic configuration file for the UVP6.
    """
    
    # read IDs from EcoTaxa (csv file) and create a dictionnary from it
    print('Read EcoTaxa IDs')
    classes_id = pd.read_csv('../id_classes_ecotaxa.csv')
    classes_id.set_index(keys = 'display_name', inplace=True) # rename indices with label names
    classes_id = classes_id.to_dict() # convert to a dict()
    # check if dict exists
    if type(classes_id) == dict:
        print('OK.')
    else:
        raise TypeError("Dictionnary with EcoTaxa's IDs does not exist.") 
    
    # prepation of the taxo configuration file
    print('Create TAXOCONF file')
    
    # number of classes in the current model
    n_classes = len(dict_classes)
    classes = list(dict_classes.keys()) # list of all classes in the CURRENT model
    
    # name of the taxo configuration file
    TAXOCONFNAME = 'TAXO_NKE_0'

    # creation of the header
    f = open(os.path.join(output_dir, "TAXO_CONF_NKE_0_"+key+".txt"), "w")
    f.write("// Taxonomic classification configuration parameters : \n\n// Parameter name: Configuration_name, String type, length range: 1 to 15 characters\n// Description: Name used to set this configuration into the field Taxo_conf, from the Acquisition configuration file\nConfiguration_name = "+TAXOCONFNAME+"\n\n// Parameter name: Model_reference, String type, length range: 1 to 15 characters\n// Description: Taxonomic classification model to be used. Must match the model binary file name into UVP6 SD card\nModel_reference = "+MODEL_REF+"\n\n// Parameter name: Max_size_for_classification, Integer type, range: 0 to 65535\n// Description: Maximum vignette size to perform the embedded classification - Unit: pixels\nMax_size_for_classification = 65535\n\n")
    f.close()

    # append data for classes
    for i in range(n_classes):
        f = open(os.path.join(output_dir, "TAXO_CONF_NKE_0_"+key+".txt"), "a")
        f.write("// Parameter name: Taxo_ID_for_class_"+f"{i:02d}"+", Integer type, range: 0 to 9999999\n// Description: Taxonomic unique identifier for model's class "+f"{i:02d}"+"\nTaxo_ID_for_class_"+f"{i:02d}"+" = "+str(classes_id['id'][classes[i]])+"\n\n")
        
    print('Done.')
