# Unconditional-Model

This code performs following tasks

- Finds best report with respect to baseline 1 method


- Generates report using the baseline2 method 


- 'baslines.py' is the main file containg all the functions to extract baseline report. 'extract_baseline_reports.ipynb' is the file using 'train.csv' and 'test.csv' (both these files should be stored in data folder) to extract baseline report using functions in 'baselines.py'. 



# chexpert_labeler_operations
All the operations we performed using the labels (annotated using chexpert labeler).

- Here you have to use 'Baseline3_extraction_IU.ipynb' to extract a baseline 3 report. This will also output the BLEU scores with respect to new baseline3. 
- Source files (which is chexpert based labels, computed using chexpert labels, against a training data) should be stored in the data folder. Moreover, a file with cleaned test report should also be stored in data folder to compute bleu scores.

- 'chexpert_labeler_metrics' can be used to compute all the clinical validation measures between candidate reports and reference reports with respect to chexpert based labels.


