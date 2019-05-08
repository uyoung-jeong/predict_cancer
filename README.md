This repository is created for 2019 BME203 Intro to Bioinfo course project 4. The XGBoost classifier model conucts classification of mutation information of the protein sequences. 

This model resulted 0.9413 validation result from the first run without any additional tuning. Additional hyperparameter tuning and feature engineering would increase the performance.

## Dataset
This model uses the dataset specified in [fathmm](http://fathmm.biocompute.org.uk/downloads.html).

All of the data files shall be saved in ./data directory

* Cancer-associated data: retrieved from [fathmm](http://fathmm.biocompute.org.uk/datasets/training/cancer/CanProVar.fa)

* Neutral data: retrieved from [fathmm](http://fathmm.biocompute.org.uk/datasets/training/inherited/humsavar.txt) and processed with [uniprot reviewed data](ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz)

* Note: header contents of 'humsavar.txt' is removed manually so that the first line becomes 
```
A1BG      P04217     VAR_018369  p.His52Arg     Polymorphism  rs893184    -
```

## Dependencies
To install all packages that are used:
```
pip install xgboost, scikit-learn, tqdm, numpy
```

## How to Run
Simply run main.py script:
```
python main.py

## Reference
* Shihab HA, Gough J, Cooper DN, Day INM, Gaunt, TR. (2013). Predicting the Functional Consequences of Cancer-Associated Amino Acid Substitutions. Bioinformatics 29:1504-1510
