# Understanding Return to the Emergency Department in Orthopaedic Patients
 This project explores 90 Day returns to the emergency department in orthopaedic patients at Duke University. Note their is no data in this Data folder as it is patient information. While it is de-identified it must continue to be held on Duke servers protected by a firewall. Please contact the authors of this project if you are interested in continuing to work with this dataset at bruno.valan@duke.edu. 
***

### requirements.txt 
contains requirements for this project. Note that all dependencies are the most up to date version with the exception of PyTorch (dependency conflict with PyTorch Tabular)
***

### setup.py 
contains script for data loading, merging, feature generation and pre-processing and model trainng and prediction. Run this file throught the command line to output the classification report of the optimized model on the test set. 
*** 

### models 
directory for saved model objects
***

### data 
directory to house data. Only the geographic data frame is saved here as the other data contains patient information and cannot be housed on GitHub.
***

### notebooks 
directory to store exploratory notebooks including generation of interactive maps
