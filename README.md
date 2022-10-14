# On-Demand Sampling: Learning Optimally from Multiple Distributions

This is an early-release of the code used in the experiments of the Neurips 2022 paper [On-Demand Sampling:
Learning Optimally from Multiple Distributions (HJZ 22)](https://eric-zhao.com/files/On-Demand%20Sampling%20%5bNeurips%202022%5d.pdf).

### Instructions
First, download the Waterbirds, MultiNLI, and CelebA datasets to the root of this project.
Then, run `ready.sh`, which will call `run.sh`.
The latter script will run the experiments.
When complete, the paper's figures can be reproduced by running the `generate_paper_results.py` script.

### Acknowledgements
This codebase is based in large part on the codebase of [the Group DRO implementation of the original authors of S. Sagawa, et al. 2019](https://github.com/kohpangwei/group_DRO).
