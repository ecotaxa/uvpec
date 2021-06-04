import os
  
def create_taxoconf(output_dir, dict_classes, MODEL_REF):
    """
    Function that creates a taxonomic configuration file for the UVP6.
    """
    
    print('Create TAXOCONF file')
    
    # number of classes
    n_classes = len(dict_classes)
    
    # name of the taxo configuration file
    TAXOCONFNAME = 'TAXO_CONF_NKE_0'

    # creation of the header
    f = open(os.path.join(output_dir, "TAXOCONF.txt"), "w")
    f.write("// Taxonomic classification configuration parameters : \n\n// Parameter name: Configuration_name, String type, length range: 1 to 15 characters\n// Description: Name used to set this configuration into the field Taxo_conf, from the Acquisition configuration file\nConfiguration_name = "+TAXOCONFNAME+"\n\n// Parameter name: Model_reference, String type, length range: 1 to 15 characters\n// Description: Taxonomic classification model to be used. Must match the model binary file name into UVP6 SD card\nModel_reference = "+MODEL_REF+"\n\n// Parameter name: Max_size_for_classification, Integer type, range: 0 to 65535\n// Description: Maximum vignette size to perform the embedded classification - Unit: pixels\nMax_size_for_classification = 65535\n\n")
    f.close()

    # append data for classes
    for i in range(n_classes):
        f = open(os.path.join(output_dir, "TAXOCONF.txt"), "a")
        f.write("// Parameter name: Taxo_ID_for_class_"+f"{i:02d}"+", Integer type, range: 0 to 9999999\n// Description: Taxonomic unique identifier for model's class "+f"{i:02d}"+"\nTaxo_ID_for_class_"+f"{i:02d}"+" = "+str(i)+"\n\n")
        
    print('Done.')
