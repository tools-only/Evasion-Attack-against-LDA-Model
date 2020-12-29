Code for the paper :
## "EvaLDA: Efficient Evasion Attacks Towards Latent Dirichlet Allocation"

### Authors : Qi Zhou*, Haipeng Chen*, Yitao Zheng, Zhen Wang. *Equal contribution

### To appear in: AAAI, 2021.

Link : https://arxiv.org/abs/2012.04864

### Instructions:

#### Installation

/Library/requirements.txt : the library for EvaLDA.

/Library/gensim.rar : the gensim which we modified.

Test data and pre-trained LDA model is in dataset/.
EvaLDA.ipynb is the code.

Bert is needed, see https://github.com/hanxiao/bert-as-service/blob/master/README.md for more detail.

#### Run the code
1. Configure the environment according to the Library/requirements.txt.
2. Download Library/gensim.rar, unzip to the local python third-party library path, replace the original Gensim.
3. Download /dataset/ , you may need to manually modify the data and model loading in the code (EvaLDA.ipynb) according to the download path.
4. Before run EvaLDA.ipynb, you should first open Bert server(see the bert link above).
5. Run the code.

### Cite

```
@article{zhou2020evalda,
  title={EvaLDA: Efficient Evasion Attacks Towards Latent Dirichlet Allocation},
  author={Zhou, Qi and Chen, Haipeng and Zheng, Yitao and Wang, Zhen},
  journal={arXiv preprint arXiv:2012.04864},
  year={2020}
}
```
