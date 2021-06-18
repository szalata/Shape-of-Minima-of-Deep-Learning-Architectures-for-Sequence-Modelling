# Shape of Minima of Deep Learning Architectures for Sequence Modelling

The description of our work is in report.pdf.

# Structure of the repo
We have added two submodules to the repository that encompass two modified python packages, one for visualization and one for computing hessian metrics. The modifications were necessary to make them work with models that take masks as input and to produce the desired ESD plots.

# Reproducing results
Cloning the repository:
```
git clone https://github.com/szalata/Shape-of-Minima-of-Deep-Learning-Architectures-for-Sequence-Modelling.git
cd Shape-of-Minima-of-Deep-Learning-Architectures-for-Sequence-Modelling
git submodule init
git submodule update
```


Create the conda environment from the environment.yml file:
```
conda env create -f environment.yml
```
Activate the conda environment:
```
conda activate optim_proj
```

Install submodules:

```
cd PyHessian; python setup.py install
cd ..
cd loss-landscapes; python setup.py install
cd ..
```

Execute all the experiments and generate plots:
```
./run.sh
```

The output will be saved in the directory `output` with subdirectories for each set of parameters.

