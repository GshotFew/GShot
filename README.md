# GShot: Few Shot Labeled Graph Generative Modeling


## Installation
The code has been tested over [PyTorch 1.8.0](https://pytorch.org/) version with Python 3.7.0.

Change cuda version as per your GPU hardware support.

```bash
pip install -r requirements.txt
```

[Boost](https://www.boost.org/) (Version >= 1.70.0) and [OpenMP](https://www.openmp.org/) are required for compling C++ binaries. Run `build.sh` script in the project's root directory.

```bash
./build.sh
```

# Scripts to Run GShot:
* `python meta_main_n.py leukemia`   :  **Training GShot Meta-Model** 
* `python tune_main_n.py leukemia`   :  **Fine-Tuning GShot for Leukemia** 
* `python evaluate_n.py leukemia`   :  **Evaluation on Leukemia** 


Here leukemia is the name of the config file present in `configs/` folder.
For AIDS-ca:`aids_ca`, Enzyme: `enzyme` , Spring: `spring`.

### Code description

1. `meta_main_n.py` is the main script file of GShot, and specific arguments are set in configs/`<Config_Filename>`
2. `tune_main_n.py` is the fine-tuning script file of GShot and specific arguments are set in configs/`<Config_Filename>`
3. `train.py` model specific training files.
4. `datasets/preprocess.py` and `util.py` contain dataset processing functions.
5. `datasets/process_dataset.py` reads graphs from different data formats.
6. `graphgen/model.py` model files


### DFS code generation:

- `dfscode/dfs_code.cpp` calculates the minimum DFS code required by GraphGen. It is adapted from [kaviniitm](https://github.com/kaviniitm/DFSCode).

- `dfscode/dfs_wrapper.py` is a python wrapper for the cpp file.



### Parameter setting:
- Configuration: Dataset specific parameters are present in configs/`<Config_Filename>`
- All the input arguments and hyper parameters setting are included in `base_args.py`.
- Dataset specific arguments which override the default arguments of `base_args.py` are present in configs folder.
- See the documentation in `base_args.py` for more detailed descriptions.

### Outputs
`base_args.py` contains parameters to specify where the default directory etc. should be

- `tensorboard/` contains tensorboard event objects which can be used to view training and validation graphs in real time.
- `model_save/` stores the model checkpoints
- `tmp/` stores all the temporary files generated during training and evaluation.

### Evaluation

- `evaluate_n.py`:  File to compute metrics between generated graphs and graphs from original dataset.

- We use [GraphRNN](https://github.com/snap-stanford/GraphRNN) implementation for structural metrics.
- [NSPDK](https://dtai.cs.kuleuven.be/software/nspdk) is evaluated using [EDeN](https://github.com/fabriziocosta/EDeN) python package.



## Baseline models:

### GraphGen 
- `python main_n.py aids_ca_single`  **Training GraphGen on AIDS-CA dataset** 
- `python evaluate_n.py aids_ca_single`  **Evaluating GraphGen on AIDS-CA dataset** 


### GraphRNN
- `python main_n.py aids_ca_single-grnn`  **Training GraphRNN on AIDS-CA dataset** 
- `python evaluate_n.py aids_ca_single-grnn`  **Evaluating GraphRNN on AIDS-CA dataset** 


### PreTrain GraphGen
- `python multi_main_n.py  aids_ca_multi`  **Training PreTrain GraphGen** 
- `python tune_main_n.py  aids_ca_multi`  **Fine-Tuning PreTrained GraphGen on AIDS-CA dataset** 
- `python evaluate_n.py aids_ca_multi`  **Evaluating  PreTrain GraphGen on AIDS-CA dataset** 





