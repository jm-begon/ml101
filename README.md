# ml101
An introduction to Machine Learning (through notebook with the scikit ecocystem).

## Setup (using conda)

Create an environment for this tutorial:

> conda create --name ml101 python=3.8

Switch to the branch (`conda activate ml101`) and install the required packages:

> conda install numpy scipy scikit-learn matplotlib ipykernel jupyter
> conda install -c conda-forge py-xgboost

Make the environment available for the notebooks

> python -m ipykernel install --user --name=ml101

Launch Jupyter to run the notebooks:

> jupyter notebook

## Content

- Machine Learning paradigms (01-ml-paradigms)
- Regression (02-regression)
- Classification (03-classification)
- Complexity 
  	- Overfitting (04-overfitting)
  	- Error decompositions (05-errors)
  	- Regularization (06-regularization)
	- Model selection (07-model-selection)
- Advanced methods
  	- Ensemble (08-ensemble)
	- Boosting (09-boosting)
	- Deep learning (10-neural-nets)
- Appendix
	- Useful tools (a1-tools)
