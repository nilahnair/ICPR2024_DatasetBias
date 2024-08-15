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

Hyperparameters used for each network for the respective dataset:

LARa MoCap:
            CNN-IMU network:
                learning rate: 0.001
                Batch Size: 100
                Epoch: 10

            LSMT network:
                learning rate: 0.001
                batch size: 100
                Epoch: 15

            Transformer (cnn-transformer): 
                learning rate: 0.001    
                batch size: 100 
                Epoch: 30

Mobiact:
            CNN network:
                learning rate: 0.001
                batch size: 100
                Epoch: 30

            LSTM network: 
                learning rate: 0.001
                batch size: 50
                Epoch: 15

            Transformer (cnn-transformer):
                learning rate: 0.001
                Batch Size: 50
                Epoch: 15

MotionSense: 
            CNN network:
                learning rate: 0.001
                batch size: 50
                Epoch: 30

            LSTM network: 
                learning rate: 0.001
                Batch size: 100
                learning rate: 30

            Transformer (cnn-transforemr):
                learning rate: 0.001
                bathc size: 100
                learning rate: 15

Sisfall:
            CNN network: 
                learning rate: 0.001
                batch size: 50
                Epoch: 50

            LSTM network:
                learning rate: 0.001
                batch size: 50
                Epoch: 15

            Transformer (cnn-transformer):
                learning rate: 0.001
                batch size: 100
                Epoch: 30

                





