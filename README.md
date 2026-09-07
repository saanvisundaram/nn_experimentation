# nn_experimentation
Repo of all mini neural network projects

Notes: most datasets unavailable on repository due to file size. All datasets used were open access & available online.

Project 1: Basic Classifier + Torch
  --> Built neural network with source code from [blog]. Used small, political affiliation dataset provided to classify affiliation based on homestate, education level, etc. Learnt effectiveness of hyperparameters, and the nitty-gritty methods & math behind classifiers.
  --> Used same dataset + code to implement PyTorch methods for cleaner neural network. Experimented with different activation functions. Familiarized myself with Torch methods & level of abstraction
  --> Changed dataset to OASIS-2 (Open Access Series of Imaging Studies 2) to classify dementia severity in patients based on brain and cognitive measurements (normalized whole brain volume, intracranial volume, socio-economic status, gender, age, education level, etc). Explored association between certain measurements and clinical dementia rating.

Project 2: tfidf & word2vec
  --> Downloaded & cleaned IMDb movie review + sentiment dataset available online. 
  --> Converted cleaned dataset to tfidf vectors through scikit-learn & trained model on sentiment prediction
  --> Downloaded Google word2vec & processed movie review dataset to gain word2vec vectors. Used to classify
  --> Compared tfidf & word2vec classifier results, learning tradeoffs and limitations for each
  --> Combined both vectors and trained final classifier with all information
  --> Heavily improved data cleaning abilities with large dataset & high variability.

Project 3: BERT
  --> Processed IMDb Dataset through BERT model & performed same classification
  --> Downloaded new dataset: survey responses with depression rating (find link). Cleaned dataset to have numerical outputs (depression rating) and fine tuned separate BERT model on it. 
