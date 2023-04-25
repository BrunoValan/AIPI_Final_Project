# AIPI FInal Project
This is the repository for my final project for AIPI 540 - Deep Learning. This project explores 90 Day returns to the emergency department in orthopaedic patients at Duke University. Note their is no data in this Data folder as it is patient information. While it is de-identified it must continue to be held on Duke servers protected by a firewall. Please contact the authors of this project if you are interested in continuing to work with this dataset at bruno.valan@duke.edu. 
***
###requirements.txt 
contains requirements for this project. Note that all dependencies are the most up to date version with the exception of PyTorch (dependency conflict with PyTorch Tabular)
***
### setup.py 
contains script for data loading, merging, feature generation and pre-processing and 
### models 



├── README.md               <- description of project and how to set up and run it
├── requirements.txt        <- requirements file to document dependencies
├── Makefile [OPTIONAL]     <- setup and run project from command line
├── setup.py                <- script to set up project (get data, build features, train model)
├── app.py                  <- app to run project / user interface
├── scripts                 <- directory for pipeline scripts or utility scripts
    ├── make_dataset.py     <- script to get data [OPTIONAL]
    ├── build_features.py   <- script to run pipeline to generate features [OPTIONAL]
    ├── model.py            <- script to train model and predict [OPTIONAL]
├── models                  <- directory for trained models
├── data                    <- directory for project data
    ├── raw                 <- directory for raw data or script to download
    ├── processed           <- directory to store processed data
    ├── outputs             <- directory to store any output data
├── notebooks               <- directory to store any exploration notebooks used
├── .gitignore              <- git ignore file
