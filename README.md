# Shape of Minima of Deep Learning Architectures for Sequence Modelling

The description of our work is in report.pdf.

# Structure of the repo
We have added two submodules to the repository that encompass two modified python packages, one for visualization and one for computing hessian metrics. The modifications were necessary to make them work with models that take masks as input and to produce the desired ESD plots.

# Reproducing results
Cloning the repository:
```
git clone https://github.com/szalata/optim_proj.git
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

Execute all the experiments:
```
./run.sh
```


