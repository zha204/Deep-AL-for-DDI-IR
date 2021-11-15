<h1 align="center">Deep-AL-for-DDI-IR</h1>

# About The Project
    
Drug-drug interaction (DDI) information retrieval (IR) is an important natural language process (NLP) task for  DDI text mining from the PubMed literature. In this paper, for the first time, active learning (AL) is studied in DDI IR analysis. DDI IR analysis from PubMed abstracts faces the challenges of relatively small positive DDI samples and overwhelmingly large negative samples. New sampling schemes, including random sampling and positive sampling, are purposely designed to address these challenges. They reduce annotation labor, and improve the efficiency of AL analysis. The theoretical consistency of random sampling and positive sampling is also shown in the paper. Practically, PubMed abstracts are divided into two pools. Screened pool contains all abstracts that pass the DDI keywords query in PubMed, while unscreened pool includes all the other abstracts. At a prespecified recall rate of 0.95, DDI IR analysis performance is evaluated and compared in precision. In screened pool IR analysis using supporting vector machine (SVM), similarity sampling plus uncertainty sampling improve the precision of AL over uncertainty sampling, from 0.89 to 0.92 respectively. In the unscreened pool IR analysis, the integrated random sampling, positive sampling, and similarity sampling improve the IR analysis performance over uncertainty sampling along, from 0.72 to 0.81 respectively. When we change the SVM to a deep learning method, all sampling schemes consistently benefit DDI AL analysis in both screened pool and unscreened pool. Deep learning also has significant improvement of precision over SVM, 0.96 vs 0.91 in screened pool, and 0.90 vs 0.81 in the unscreened pool, respectively.


# Table of Content
## Data

### Screened sample pool 
1. **Negative Sample ID**: 933 Labeled Positive Abstracts
2. **Positive Sample ID**: 799 Labeled Negative Abstracts
3. **Screened Sample ID**: 3,169 Unlabeled Abstracts

### Unscreened sample pool 
**Unscreened Sample ID**: 9,999 Unlabeled Abstracts


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

**Note**

1. When running the main.py once, it means one round for active learning according to the strategy. 
2. The input and output for the main.py :
    - Input: the text of abstracts
    - Output: the evaluation results such as precision, recall values, etc. 
3. All the codes are written in python



