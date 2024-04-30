This is my reproduction of [Anthropic's research](https://transformer-circuits.pub/2023/monosemantic-features) on monosemanticity in language models. I wrote a [blog post](https://jakeward.substack.com/p/monosemanticity-at-home-my-attempt) about it where I share details and results.

This repo is WIP, so the clearest code if you want to try this yourself is in [`neuronresampling.ipynb`](https://github.com/jnward/monosemanticity-repro/blob/main/neuron_resampling.ipynb).

Future work:
- [ ] Self-contained and minimal jupyter notebook/colab notebook
- [ ] Implement Anthropic's [recent updates](https://transformer-circuits.pub/2024/april-update/index.html) to their research (unconstrained encoder norm, L2 regularization, etc.)
- [ ] Experiments with multi-layer transformers
- [ ] Better docs...
