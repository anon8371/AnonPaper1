# Sparse Distributed Memory is a Continual Learner

This is the codebase behind the paper *Sparse Distributed Memory is a Continual Learner* 

We provide code sufficient to reproduce all of our experiments. 

Follow these steps: 

1. Set up a new Conda environment using: 

`conda create --name SDMContLearn python=3.8.3`
`conda activate SDMContLearn`
`conda install pip`
`pip install -r requirements.txt`

You may be able to use other versions of the libraries found in `requirements.txt` but no promises. 

2. `cd py_scripts` and then run `python py_scripts/setup_cont_learning.py` to get all the datasets and split them. We provide the ConvMixer embeddings of CIFAR10 that were used. 

The ImageNet32 embeddings are available for download at. 

Need to git clone this. then cd all the way and run 

`cd nta/`
`git clone https://github.com/numenta/htmpapers`
`cd htmpapers/biorxiv/going_beyond_the_point_neuron`
`pip install -r requirements.txt `

3. Run `cd ..` to get back to the main directory and then `python test_runner.py` that will by default run an SDM model on Split MNIST. 

See `exp_commands` for experiments that can be run. 

# Code Base

* Folders: 
    * data/ - folder containing MNIST digits
    * py_scripts/ - all supporting scripts for training the models. 
    * models/ - contains all of our model architecture code



## SDM Biological Plausibility
* Replicating ["A functional model of adult dentate gyrus neurogenesis"](https://www.semanticscholar.org/paper/A-functional-model-of-adult-dentate-gyrus-Gozel-Gerstner/cb5c8122b52062b4d69504f919c6ffdaf0abc965)
    * MATLAB_Neurogenesis_Prep.ipynb - Prepare MNIST images by reducing them from 28x28 to 12x12 as in the paper.  
    * Neurogenesis.ipynb - run the actual experiments              
    * utils_Neurogenesis.py - where function calls should go

          
* ReLUStillExponentialDecay.ipynb - Testing that SDM with different ReLU and exponential activation functions still gives the exponential approximation to Transformer Attention.
