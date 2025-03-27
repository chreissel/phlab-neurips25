# Semi-Supervised Contrastive Pipelines for Science
This repository contains the codebase for our neurips '25 paper. I'm assuming you're running on the cannon cluster. If you're not, some things will be broken (e.g. datasets might point to data-containing directories that don't exist).

## Basic setup
To make things "easier", we're using pytorch lightning to organize/run experiments. It has a lot of opaque and annoying features, but in a landscape of bad tools it's probably the most widely used and easy to learn. 

To make things as flexible as possible we're using the lightning command line interface (CLI), wherein your experiment is fully defined by a `.yaml` config file that specifies:
- The model you want to train (a `LightningModule` where you define the network and implement functions that tell lightning what to do on a train/val/test step, i.e. evaluate the network on a batch and compute a loss). The top-level definitions of these modules should go in `models/litmodels.py`, with helper functions and pytorch network definitions going in other files in that directory.
- The dataset you want to train on (a `LightningDataModule` with functions that return a train, val, test, or predict dataloader). Defined in `data/datasets.py`.
- The optimizer / scheduler you want to use. Can either be defined in the `yaml` config passed to the CLI, or in the `configure_optimizers` method of the `LightningModule`. Depends how much control you need/want.
- Various administrative things, like what kind of logger to use, where/under what conditions to save model checkpoints, etc.

Have a look at [this config](configs/imagenette_fineTuneSimCLR.yaml) for an example - it's set up to fine-tune a pre-trained model from the original [SimCLR paper](https://arxiv.org/abs/2002.05709) on the ten-class [Imagenette](https://github.com/fastai/imagenette) dataset (using the original SimCLR image augmentations), resulting in an 8-dimensional contrastive space. 

## Running a training
Once you've written your models, datasets, loaders, and so on, just define a new config and run:
```bash
python cli.py fit --config configs/my_config.yaml
```
Voilà, you've trained a model.

## Evaluating a training
There's a way to make pytorch lightning do this, but I haven't implemented it yet. You probably want something that computes the learned embeddings on some test dataset and saves them somewhere. For now just do this however you want (a jupyter notebook or something).

## Doing experiments & getting downstream results
Any dataset-specific downstream work you do can go in a subfolder of `experiments` (e.g. clustering or training a classifier on partially labeled data to label the full dataset). I'm not going to impose any structure there, so this can be a garbage dump wasteland of jupyter notebooks. Use lightning for any larger scale trainings you want to do (e.g. a second supervised SimCLR step using the labels derived from the learned space from the first augmentation-based round of simCLR). 

## A note on networks/loaders
It's good practice to abstract away details from any *pytorch lightning* code you write. For example, the `LightningModule` called `SimCLRModel` in `models/litmodels.py` should suffice for essentially *any* augmentation-based contrastive training you want to do. Your loader should return a pair of augmentations, rather than explicitly applying augmentations with custom functions in the `training_step` function. The details of how to do the augmentations are offloaded to whatever custom loader you write. This is cleaner than writing 25 different `LightningModule`s with different data augmentations, and minimizes the amount of pytorch lightning code you have to write (it sucks, avoid it). 

## Some tips & tricks
The more you have to learn about how lightning works, the more you'll want to die. The documentation exists, but is generally low quality and the answers to some pretty fundamental questions are not easy to find. There are also obvious shortcomings in the software structure. Want to specify the scheduler step interval (step vs epoch) in the yaml config? Forget about it. ChatGPT is useful for most questions, since they'll be hard to find answers to on the lightning website.

## Conda environment
Of course I actually mean *Mamba* environment, or something even better (I hope you're not using vanilla conda). You're obviously going to need to install reasonably up-to-date versions of pytorch, pytorch lightning, probably numpy, some other things. I hope you're going to make at least a plot or two so install matplotlib as well. I won't provide a `yaml` for installing a mamba environment because we're not using any weird/bespoke packages that would justify a new environment taking up space. You probably already have an environment with everything you need.