# dgm-online-eval
Official repository of the paper "A Multi-Armed Bandit Approach to Online Evaluation and Selection of Generative Models" (AISTATS 2025)

[Xiaoyan Hu](https://yannxiaoyanhu.github.io), [Ho-fung Leung](http://www.cse.cuhk.edu.hk/~lhf/), [Farzan Farnia](https://www.cse.cuhk.edu.hk/~farnia/Home.html) [[Paper](https://arxiv.org/abs/2406.07451)]

![Figure](https://github.com/yannxiaoyanhu/dgm-online-eval/blob/main/Fig1.png)

## Usage Examples

FD-based evaluation and selection: ```python fd_online_eval.py --total_nsamples=5000 --batch_size=5 --evaluator_name=fd-ucb --num_model=3```.

IS-based evaluation and selection: ```python is_online_eval.py --total_nsamples=5000 --batch_size=5 --evaluator_name=is-ucb --num_model=3```.

Implementations of FD-UCB, IS-UCB and methods can be found in algorithms.

## Acknowledgements

The authors would like to acknowledge the following repositories:

1. DINOv2: [https://github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2).
2. OpenCLIP: [https://github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip).
3. dgm-eval: [https://github.com/layer6ai-labs/dgm-eval].


## Citation
```
@misc{hu2024optimismbased,
      title={An Optimism-based Approach to Online Evaluation of Generative Models}, 
      author={Xiaoyan Hu and Ho-fung Leung and Farzan Farnia},
      year={2024},
      eprint={2406.07451},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
