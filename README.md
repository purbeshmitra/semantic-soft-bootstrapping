# Semantic Soft Bootstrapping (SSB)

🔗 Paper link: [Arxiv preprint](https://arxiv.org/abs/2507.02851)

🔗 Link to the trained model: [Hugging Face collection](https://huggingface.co/purbeshmitra/semantic-soft-bootsrapping)

Semantic Soft Bootstrapping (SSB), an RL-free self-distillation framework that improves long-context reasoning in LLMs by training the model on its own hinted reasoning as a teacher. Rather than relying on a separate larger teacher or on-policy gradient with sparse rewards, SSB uses the same base model in two semantic roles: a hinted teacher that sees both correct and incorrect solutions and synthesizes a robust explanation, and a hint-free student that learns to reproduce this behavior from the bare question alone. Starting from a raw problem–answer dataset, we construct paired teacher–student conversations and then precompute teacher logits over the answer tokens, enabling efficient offline distillation without any human annotation or online RL loop. This is depicted as following:
<p align="center">
  <img src="assets/ssb.png" alt="Alt Text" width="750">
</p>

Our results are shown below:
<p align="center">
  <img src="assets/ssb_results.png" alt="Alt Text" width="750">
</p>

## Citation
If you find our work useful, consider citing it as:
```bibtex
@article{mitra2025semantic,
  title={Semantic Soft Bootstrapping: Long Context Reasoning in LLMs without Reinforcement Learning},
  author={Mitra, Purbesh and Ulukus, Sennur},
  journal={arXiv preprint arXiv:},
  year={2025}
}
```
