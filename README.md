This is a private repository containing my implementation of the DGMR architecture and data pipelines developed during my internship at KNMI.

The data preprocessing directory contains all the necessary scripts to:
  1. Download Data from KNMI dataplatform (access key need to be acquired)
  2. The data preprocessing pipeline to (i) iterate over the downloaded data (ii) extract multiple crops (iii) and Filter based on Precipitation intensity 
  (iv) data are saved in a tf record format
  (It can further handle missing data and also cases where the different features do not overlap in terms of time (ex. sampled 1.5 minutes apart))
  
The DGMR directory contains files:
  1. A custom implementantion of the DGMR architecture (Together with some SONNET components)
  2. To train the DGMR model.
  3. Multiple GPU setup
  4. Evaluation Functions saved for a tensorboard format.

The dataset is not provided.
  
Weights of trained models are not provided.

  
