Preprocessing step:
    According to the dataset of interest choose the preprocessing file.
    1. LARa MoCap - LARa_dataset_preprocessing.py
    2. MobiAct - preprocessing_mobiact.py
    3. MotionSense - preprocessing_ms.py
    4. Sisfall - preprocessing_sisfall.py

    for all the datasets, appropriate creation of experiment folders for saving the preprocessed 
    files must be created prior to execution of python code. For example:
    -   preproces/expxx/sequences_train
    -   preproces/expxx/sequences_val
    -   preproces/expxx/sequences_val

    For all the preprocessing files, csv_reader and sliding_window python files are necessary.

Training Neural Network:
    
    For each experiment along with the preprocessing file, a folder to store the results and in-training model of the 
    neural network is necessary. For example:

    -   results/expxx/plots

    createfolder.py can be used for creating the folders in this format.

    Use the main.py for starting the experiment. The choice of neural network, dataset, batchsize and other such hyperparameters
    can be set here. 





