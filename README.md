# layered-ac
This repository contains the code required to reproduce the results in the
paper "Coordinating Planning and Tracking in Layered Control Policies via
Actor-Critic Learning" by F. Yang and N. Matni.

The `notebooks/` folder constrains a notebook that illustrates how to use this
codebase. The experiment scripts are contained inside the `experiments` folder,
with the exception of the unicycle system, which is in the notebook. After
running the notebook or an experiment script, a `runs/` folder will be created
inside the script directory, which contains the training logs. These training
info can be visualized and read off by running `tensorboard --logdir
{directory_name}/run/`.
