# ProgUnknownFMs

This repository contains the Python code of our paper titled "[Degradation Modeling and Prognostic Analysis Under Unknown Failure Modes](https://arxiv.org/abs/2402.19294)".

## Files

```dataset/					collected datasets
dataset/						[Collected dataset]
	Aircraft Engine/CMaps		[CMAPSS Jet Engine Simulated Data]
src/							[Python source code]
	data/						[Code for data preprocessing]
		JetEngine.py			[CMAPSS data preparation] 
		util.py					[Utility functions for data preprocessing]
	dataloader/					[Data loading utilities]
		sequence_dataloader.py				
		seq_grc_dataloader.py
		seq_branch_dataloader.py
		seq_grc_branch_dataloader.py
	models/						[Model architecture definitions]
		cnn.py
		cnn_branch.py
		fc.py
		fc_branch.py
		lstm.py
		lstm_branch.py
		lstm_grc.py
		lstm_joint.py
		lstm_joint_with_prob.py
		lstm_joint_with_prob_grc.py
	config.py					[Configuration settings]
	UMAP_dr.py					[Script for performing UMAP dimension reduction]
	FM_identification.py		[Script for failure mode identification]
	train_evaluate.py			[Functions for model training and evaluation]
	exp.py						[Script for conducting experiments]
	result_analysis.py			[Script for analyzing the final results]
	


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

