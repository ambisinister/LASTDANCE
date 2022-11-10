# LASTDANCE: Layerwise Activation Similarity to Training Data for Assessing Non-Conforming Events

This is the repo for LASTDANCE, a novel method for out-of-distribution detection which leverages feature trajectory through the network, rather than using a single "optimal" layer, or an "early exit" network/ensemble, which sacrifices performance on the task. Using this trajectory paradigm, we set a new state-of-the-art result on Competency Estimation.

For more info on this project, see out [pre-preprint](https://planetbanatt.net/articles/lastdance.pdf), or LASTDANCE_README.ipynb in this repository.

## Experiments

To get a file containing trajectories, you can run the following command:

```
python experiment.py
```

you can change the model to use model.eval(), or change the experiment to use svhn instead, as follows

```
python experiment.py --test --in_set=svhn
```

To get mahalanobis distances for OOD points, you can run the following after you have run experiment.py

```
python mahalanobis.py
```

## TODO
project onto unit ball -> cosine similarity
johnson lindenstrauss bound maybe?
openOOD / other datasets
classification performance
gmms instead of class conditional gaussians

## Cite

If you find our repository useful for your research, please consider citing our paper:

```
@article{banatt2022lastdance,
    author = {Banatt, Eryk},
    title = {LASTDANCE: Layerwise Activation Similarity to Training Data for Assessing Non-Conforming Events},
    year = {2022}
}
```