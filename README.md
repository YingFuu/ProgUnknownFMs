# ProgUnknownFMs

This repository contains the Python code of our paper titled "[Degradation Modeling and Prognostic Analysis Under Unknown Failure Modes](https://arxiv.org/abs/2402.19294)".

## Files

```dataset/					collected datasets
dataset/						[Collected dataset]
src/							[Python source code]
	data/						[Code for data preprocessing]
	dataloader/					[Data loading utilities]
	models/						[Model architecture definitions]
	config.py					[Configuration settings]
	UMAP_dr.py					[Script for performing UMAP dimension reduction]
	FM_identification.py		[Script for failure mode identification]
	train_evaluate.py			[Functions for training, evaluation, and prediction]
	exp.py						[Script for conducting experiments]
	result_analysis.py			[Script for analyzing the final results]
	
result/							[Computational results]
    figures-0722-supervised-False-align-False-linear/    [UMAP dimension reduction results]
    result-FD003-0101-BranchLSTM-loss_fm-0.12/           [Benchmark results of branch LSTM]
    result-FD003-1210									 [Results of proposed models]

```

## Requirements

```{}
python=3.9.12
pandas=1.5.2
numpy=1.21.5
matplotlib=3.5.1
torch=1.12.0
```


## Usages

1. Execute `src/UMAP_dr.py` to generate the dimension reduction results.

2. Execute `src/FM_identification.py` to determine the failure mode labels for the training units.

3. Execute `src/exp.py` to obtain the results of the proposed models, benchmark models, and baseline models.
4. Execute `src/result_analysis.py` to produce the figures presented in the paper.

## Reference

If you find the code useful, please cite our paper:

```{}
@article{fu2024degradation,
  title={Degradation Modeling and Prognostic Analysis Under Unknown Failure Modes},
  author={Fu, Ying and Huh, Ye Kwon and Liu, Kaibo},
  journal={arXiv preprint arXiv:2402.19294},
  year={2024},
  url={https://arxiv.org/abs/2402.19294}
}
```

