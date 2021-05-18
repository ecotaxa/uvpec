import pandas as pd

def sample_detritus(dataset, subsampling_percentage, random_state):
    """
    Function that randomly selects detritus based on a user-specified percentage
    """
    
    detritus = dataset[dataset['labels'] == 'detritus']
    other_classes = dataset[dataset['labels'] != 'detritus']
    detritus = detritus.sample(frac = subsampling_percentage/100, replace = False, random_state = random_state)
    new_dataset = pd.concat([detritus, other_classes])
    new_dataset = new_dataset.reset_index(inplace=False, drop = True)
    
    return(new_dataset)