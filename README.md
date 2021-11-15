# Deep-AL-for-DDI-IR

## Code

### main.py 
The main funciton program, it sets the main flow, including setting device and numbers of each parameters, predicting and generating training and validation samples for each iterations, evaluates the models, and save the results.

The following three py files are the modules the main.py needed.

### Utils.py  
Defined the assignment of dataset for active learning iterations, including loading data, generate initial dataset and validation datasets for two models, split and assign the uncertainty samples and save the dataset for manual reviewing.

### model.py 
Defined the FastTest algorithm

### run_model.py 
Defined how to train the model use the parameters including iteration, epoche, batch size, sampling methods and so on. 

*Note*

1. When running the main.py once, it means one round for active learning according to the strategy. 
2. The input and output for the main.py :
    - Input: the text of abstracts
    - Output: the evaluation results such as precision, recall values, etc. 
