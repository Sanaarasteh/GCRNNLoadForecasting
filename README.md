# Short-Term Load Forecasting (STLF) Using Graph Convolutional Recurrent Neural Networks (GCRNN)

This is the GitHub repository for the paper "Short-Term Load Forecasting Using Graph
Convolutional Recurrent Neural Networks" EAAI 2022. 

### Brief Model Description
In this paper, we present a **Graph Convolutional Recurrent Neural Network (GCRNN)**, based on the integration of 
Graph Convolutional Networks (GCN) and Long Short-Term Memory (LSTM) networks, to simultaneously incorporate the spatial and 
temporal information of similar users for more accurate short-term predictions.
In this paper, we integrate GNN and LSTM networks into a unified network to extract the relational and temporal
information simultaneously. In the proposed model, the LSTM’s conventional convolution gates are replaced with
graph convolution networks. Unfolding the resulting network can capture the temporal behavior of a node and the
effects that its neighbors impose through time. Tailored to the short-term load forecasting problem, the graph’s nodes
represent various consumers, and the links between the nodes indicate a similarity in their corresponding consumption
patterns. Such a model mitigates the issue of former studies in neglecting the relational information between the users.
Moreover, it produces powerful and informative representations suitable for load forecasting without unnecessarily
increasing the dimensionality of the data.

![GCRNN Framework](https://github.com/Sanaarasteh/GCRNNLoadForecasting/blob/master/images/framework.jpg)

### Structure of the Code and How to Use the Repository
The repository contains several important files and directories.
The python files in the main directory are general functionalities
common to both **Low Carbon London (LCL)** and **Customer Behavior Trials (CBTs)**
datasets.
* **layers.py -** contains the implementation for a Graph Convolutional
LSTM (GCLSTM) cell which is an LSTM cell which its gates has been replaced
by graph convolutional layers.
* **models.py -** contains the unified GCRNN network followed by an MLP to be
used for short-term load forecasting.
* **utils.py -** contains the helper functions used throughout the project.
* **transforms.py -** contains simple callables to transform numpy instances
to Torch tensor instances.
* **dataloader.py -** contains the standard Torch DataLoader overrides for properly
reading the **LCL** and **CBT** datasets.

Both **LowCarbonLondon** and **CustomerBehaviorTrials** directories contain
similar python files for manipulation and experimentation on the corresponding
datasets. Each directory contains the following files:
* **data_inspection.py -** contains necessary files to obtain preliminary information
about the data.
* **preprocessing.py -** contains necessary files to transform and convert the data into
the proper format.
* **results_reader.py -** contains a function for preparing and displaying the summary of
the performance of a specified model.
* **train_gcrnn.py -** contains the necessary procedure for training the GCRNN model.
* **train_lstm.py -** contains the necessary procedure for training the LSTM model.
* **train_svr.py -** contains the necessary procedure for training the SVR model.
* **train_ffnn.py -** contains the necessary procedure for training the Feed Forward MLP model.
* **train_arima.py -** contains the necessary procedure for training the ARIMA model.
* **test_gcrnn.py -** contains the necessary procedure for testing and evaluating the 
GCRNN model and drawing informative and qualitative plots and drawings.

The **dataset** directory contains the datasets. Here we have not included the 
Low Carbon London dataset due to its volume. However, one can
download the dataset from [Kaggle](https://www.kaggle.com/jeanmidev/smart-meters-in-london). Once
you have downloaded the dataset, extract it in a subdirectory named **London** in the **dataset** directory.
Also, **to respect the ownership rights we do not include the data files for the Customer 
Behavior Dataset.** One can request the data from 
[ISSDA](https://www.ucd.ie/issda/data/commissionforenergyregulationcer/) and extract the contents
in a subdirectory named **CER** in the **dataset** directory.
These namings for subdirectories are essential for the right execution of the code.

To run experiments on the LCL dataset:
1. Run the **generate_weather_dataset()** function in **preprocessing.py**
2. Run the **make_block_dataset()** function in **preprocessing.py**
3. Run **train_gcrn.py**, **train_lstm.py**, **train_ffnn.py**, **train_svr.py**, or **train_arima.py**
to run the experiment using one of the models.

To run experiments on the CBT dataset:
1. Run the **group_maker()** function in **data_inspection.py**
2. Run the **prepare_data()** function in **preprocessing.py**
3. Run **train_gcrn.py**, **train_lstm.py**, **train_ffnn.py**, **train_svr.py**, or **train_arima.py**
to run the experiment using one of the models.

