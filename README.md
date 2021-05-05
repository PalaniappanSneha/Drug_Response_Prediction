"# Drug_Response_Prediction" 
"# Drug_Response_Prediction" 

Flow of codes and project:
1) preprocessing_updated_data.ipynb
   - Preprocessing of omics data into matrices
2) pca_6_datasets.ipynb
   - Conduct PCA on omics data matrices
3) svd_6_datasets.ipynb
   - Conduct SVD on omics data matrices
4)  ica_6_datasets.ipynb
    - Conduct ICA on omics data matrices
5) dnn_preprocessing.ipynb
   - Preparing input data for input DNN (Append drug response to data etc...)
6) combined_data.ipynb
   - Preparing input data for combined DNN
7) datasets.py
   - Read data 
8) model.py 
   - Specifications of model
9) main.py
   - Individual DNN
10) main_combined.py 
   - Combined DNN
To run the individual model:
Ex. python3 main.py --data Exp --expr_dir experiments/Exp/ --esthres 50 --dropout 0.2 --out_lay1 256 --cv
To run the combined model:
Ex. python3 main_combined.py --expr_dir experiments/combined/ --esthres 50 --dropout 0.50 --out_lay1 256 --cv
 
