# Overview
This repository provides R Source codes to reproduce numerical experiments in the following arXiv preprint:

```
@article{okuno2023HOTV,
    year      = {2023},
    publisher = {CoRR},
    volume    = {},
    number    = {},
    pages     = {},
    author    = {Akifumi Okuno},
    title     = {A stochastic optimization approach to train non-linear neural networks with a regularization of higher order total variation},
    journal   = {arXiv preprint}
}
```

## Main scripts
### <a href="https://github.com/oknakfm/HOTV/blob/main/0_demonstration.R">0_demonstration.R</a>
You can train a single neural network with the proposed stochastic algorithm. You can replace the training data <a href="https://github.com/oknakfm/HOTV/blob/main/0_demonstration.R#L34">(x,y)</a> and the optimization settings and the number of hidden units (stored in the <a href="https://github.com/oknakfm/HOTV/blob/main/0_demonstration.R#L42">``constants''</a> variable) to explore our regularization! 

### <a href="https://github.com/oknakfm/HOTV/blob/main/1_illustration.R">1_illustration.R</a>
This script provides illustration figures of the neural networks trained by several regularizations. 
Results are stored in the automatically generated ``A2_computed'' folder.

### <a href="https://github.com/oknakfm/HOTV/blob/main/2_experiments.R">2_experiments.R</a>
This script provides experimental results (predictive correlation) with several random seeds.
Results are stored in the automatically generated ``A2_computed'' folder.

## Verbose
### <a href="https://github.com/oknakfm/HOTV/blob/main/A0_scripts/gen_data.R">A0_scripts/gen_data.R</a>
1000 pairs of (x,y) following (i) linear model, (ii) quadratic model, and (iii) cubic functions, are generated. The generated instances are saved to the automatically generated ``A1_data'' folder.

### <a href="https://github.com/oknakfm/HOTV/blob/main/A0_scripts/functions.R">A0_scripts/functions.R</a>
This script provides functions describing neural networks and stochastic algorithms.

# Contact info.
Akifumi Okuno (okuno@ism.ac.jp)
