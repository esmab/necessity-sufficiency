
This repository contains all the code and the experimental results for the paper ["Necessity and Sufficiency for Explaining Text Classifiers: A Case Study in Hate Speech Detection"](https://arxiv.org/abs/2205.03302) by Esma Balkir, Isar Nejadgholi, Kathleen C. Fraser, and Svetlana Kiritchenko.

All the datasets that are used to train the infilling model and the classifiers are included in the repository, except that of [Founta et al. 2018](https://arxiv.org/pdf/1802.00393.pdf) which needs to be obtained from the authors of the paper. The jupyter notebooks, when run sequentially, will train all the models and reproduce the results presented in the paper.  Functions for perturbing the inputs and calculating necessity and sufficiency can be found in **perturbation_functions.py**
