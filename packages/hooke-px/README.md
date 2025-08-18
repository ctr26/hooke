# Big-Img

This repo contains code to train Diffusion Transformers on phenomics data with the flow matching (linear interpolant) formulation. It is intended to be as simple as possible to aid integration with the rest of Hooke.

Note that this repo uses a library called `ornamentalist` to automatically generate a CLI and configure hyperparameters. The main place where you might see this show up is in the signature of functions. If you are integrating with a project that doesn't use `ornamentalist`, it's perfectly safe to rip these parts out. This should only require minor modifications to the function signatures.