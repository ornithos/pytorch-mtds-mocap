
## Human Motion Prediction using Multi-Task Dynamical System

This repository provides the code for the mocap experiments using Multi-Task Dynamical Systems (MTDS). The main text for this is not yet published, but  see my [thesis](https://drive.google.com/file/d/1G6sZ7YTuqJ3VWza-d_WsnFoEPnrg4Smw/view?usp=sharing) for more details.

### Historical note
This repo started life via the [PyTorch clone](https://github.com/enriccorona/human-motion-prediction-pytorch) of the Martinez et al. (2018) model by [Enric Corona](https://github.com/enriccorona).<sup>[[1]](#JuliaRepoFootnote)</sup> This appeared to be the strongest competitor model I could find at the start of the project, but the original codebase is in Tensorflow. Note that the PyTorch clone keeps a lot of the command line options of the original codebase but silently removes the features -- I haven't attempted to improve the model beyond the features I needed, so please be aware of this when running the competitor model from the command line. A major problem with this codebase as it stands is that I'm unable to obtain the final scripts used to execute the code as these were run locally from my development machine. Due to the Covid pandemic, I was irrevocably separated from the resulting scripts for 18 months, and -- forgetting about the problems with this repo, effectively assented to my machine at the institute being wiped once I finished my PhD. The original experiment files have therefore been lost, for which I apologise, but I hope you understand the circumstances. I have put in a number of additional days' work to rectify this in the repo as best I can.

### Data

**The long way**: Download Ian Mason's mocap data linked from his [Github repository](https://github.com/ianxmason/Fewshot_Learning_of_Homogeneous_Human_Locomotion_Styles#training-the-models) for the paper "Few-shot Learning of Homogeneous Human Locomotion Styles", unzip it, and place it in a convenient directory. A substantial amount of pre-processing was performed on this data (Ian Mason used a bunch of scripts written within Unity -- this is not a route I wanted to go down) -- see my [mocap-mtds](https://github.com/ornithos/mocap-mtds) Julia library for the pre-processing scripts. In particular, see the [`2_preprocess.ipynb`](https://github.com/ornithos/mocap-mtds/blob/master/2_preprocess.ipynb) notebook. This is a bespoke and nontrivial preprocessing pipeline, and will require the user to become familiar with the various functionsin that project.

**The short way**: I've uploaded my processed files here: [https://bit.ly/38x2sra](https://bit.ly/38x2sra). This includes three files for inputs (`Us`), outputs (`Ys`) and a style lookup dict which maps the style label (in {1,2,...,8}) to the indices of the inputs/outputs.

### Training

I describe here how to train the model described in my thesis (linked above) using a 2-layer architecture with a 1024-hidden-unit GRU as the first layer, and a multi-task 128-hidden-unit RNN (MTDS) as the second. We fix the dynamic bias _b_ of the MTDS in order to permit smooth interpolation between styles. This can be achieved using the `src/learn_mtfixbmodel.py` file: have a look at the options using `python learn_mtfixbmodel.py -h`.

The default values used by `argparse` can be found in the `src/default.ini` file. I recommend having a look at this to see reasonable values of the parameters which were used in the experiments, including the architecture hyperparameters. A value of `-1` in this file indicates that a default is derived from other args in the code if not specified. Note that the *h\_phi* generator network here is confusingly referred to as `psi` - a metonym; I ended up changing the greek letter from `phi` -> `psi` for this parameter set post-hoc for consistency with other work.


To train a model over all styles, using the described architecture, run the following command from the command line:
```bash
python src/learn_mtfixbmodel.py --style_ix 99 --latent_k 7 --input_size 35 --bottleneck 24 --iterations 20000 --hard_em_iters 10000 \
  --learning_rate 3e-5 --learning_rate_mt 1e-3 --learning_rate_z 1e-3 --psi_affine --data_dir <data_path>
```

#### Flags
* `style_ix`: specifies the style index of the test set; i.e. to be held-out for the LOO experiments. If no data are to be held out (e.g. for style transfer purposes), then specify a number greater than 8 (yes - it's a hack, but :shrugs:; I used `style_ix 99` in my experiments for this purpose).
* `latent_k`: the dimension of the latent space for `z`.
* `input_size`: the dimension `n_u` of the input data.
* `bottleneck`: the dimension of the inter-layer bottleneck (i.e. the co-domain of the linear operator `H` in the paper/thesis).
* `iterations`: the total number of iterations for optimisation
* `hard_em_iters`: the number of iterations during optimisation for which the latent variable `z` is optimised as a "delta function" - i.e. a MAP approximation.<sup>[[2]](#InitDeltaFnFootnote)</sup> This is a form of KL-annealing to ensure the latent variable is well utilised.
* `learning_rate*`: the learning rate of each parameter set. Here `mt` is the parameter set of the multi-task network (i.e. `phi` in the paper), `z` is the learning rate of the variational posterior distributions, and the remaining learning rate is for the other parameters (largely the first-layer GRU - parameters `psi_1`, `H` in the paper).
* `psi_affine`: (boolean flag) use an affine `h_phi` instead of an MLP. Using an MLP did not appear to make significant improvements in our experiments, so use of this flag is recommended.
* `use_cpu`: (boolean flag) self-explanatory.
* `data_dir`: self-explanatory - point this to where you've downloaded the data (Us, Ys, and style_lkp).

There are plenty of other flags besides, see `parseopts.py` or `default.ini`.

### Optimisation

This is research-level code, not high performance code. Therefore training is likely to be slow (relative to the difficulty of the task) regardless of your hardware. At the time of writing I was not aware of any fast way to back-prop *through* RNN parameters: usually RNNs were constructed such that the parameters had to be leaves in the AD graph. So therefore I ended up writing all RNN operations for MTDS layers as native PyTorch code; I don't think we hit any optimised CuDNN kernels.

* **Timing**: On my current machine (Ryzen9 3900, RTX 3090) 100 iterations at the default settings take just over a minute, so 20_000 iterations take about 3 hours 20m.

* **Optimisers**: I found Adam to generally perform much better than SGD or SGD+Nesterov during initial benchmarking, but if you want to play around with this, you can specify one of these three optimisers in the cli. See `-h` for more details. Some other important default optimisation parameters include (from `default.ini`):

```
learning_rate_decay_factor: 0.95
learning_rate_step: 10000
batch_size: 16
iterations: 10000
test_every: 1000
first3_prec: 1.0
```

* **Optimisation params**: The batch size can likely be increased if you have a fairly decent GPU. By default there is exponential decay on the learning rate by a factor of 0.95 every 1000 iterations. We report test performance every 1000 iterations. `first3_prec` exists in order to apply greater precision to the root node of the skeleton - increase this from 1 if following the path exactly is of greater importance to you. I found results to be satisfactory by leaving it at one (especially since some stylistic variation necessitates deviations from the desired path), but feel free to experiment.

* **Optimisation output**: Training should generate some logging to stdout which looks similar to the following.

```
experiments/style_99/out_64/iterations_20000/decoder_size_1024/zdim_7/ar_coef_0/psi_lowrank_30/optim_Adam/lr_3e-05/std/edin_Us_30fps_final/edin_Ys_30fps_final/residual_vel/full_mtds
Reading training data (test index 99).
Using files edin_Us_30fps_final.npz; edin_Ys_30fps_final.npz
done reading data.
==== MTGRU ====
Input size is 32
hidden state size = 1024
layer 1 output size = 24
~~~~ MT Module ~~~~
hidden state size = 128
hierarchical latent size = 7
layer 2 output size = 64
Linear(in_features=7, out_features=68544, bias=True)
step 0000; step_loss: 2803.7881 (2267.8516)
step 0010; step_loss: 2563.5249 (2027.5898)
step 0020; step_loss: 2352.5552 (1816.6194)
step 0030; step_loss: 2054.4102 (1518.4752)
step 0040; step_loss: 1912.0212 (1376.0854)
step 0050; step_loss: 2304.8735 (1768.9365)
step 0060; step_loss: 2162.8218 (1626.8854)
step 0070; step_loss: 2213.5605 (1677.6216)
step 0080; step_loss: 2398.9727 (1863.0342)
step 0090; step_loss: 1885.9126 (1349.9763)
...
```

The first line tells you where the model checkpoints will be saved, we then get some information about the training data, and then some architectural details about the model being trained. After the two dynamic layers (here a GRU and MT-module), we are also shown the `h_phi` network, which in this case is `Linear(in_features=7, out_features=68544, bias=True)`, i.e. an affine network. The step losses provided show two numbers, the first number is the ELBO, the parenthetical number is the reconstruction loss (i.e. the ELBO without the KL term, see also [footnote 2](#InitDeltaFnFootnote) for the early stages of optimisation).

### Visualization

Ian Mason uses Unity for output visualisations of the training data. In order to avoid the pain of getting to know a huge and unfamiliar C# codebase, I've hacked a workflow together in Julia to visualize the result in the browser. These make use of [`three.js`](https://threejs.org/) and [`MeshCat.jl`](https://github.com/rdeits/MeshCat.jl). Due to various updates within MeshCat, I've had to patch the code up from the original implementation. Unfortunately it no longer works quite as well as it used to; specifically the angles of the bones between joints are slightly messed up, but I don't have sufficient time to fix what appears to be a cosmetic problem. In sum, this is a nontrivial piece of work, and it could be cleaner, and probably more concise, but again, time precludes fixing this. I would welcome anyone with more experience with 3D graphics/animation trying to streamline this, or point to a more mature implementation. See `experimentation/animation.ipynb` for an example of getting this working. An example of the output can be seen below.

#### Notes to self
Visualization is proving a right pain. Getting a Julia environment that works, and updating the code for breaking changes in the packages since I wrote it has taken quite some time. Now I'm at the stage of trying to construct the skeleton from the output, and it's proving hard, so I've dumped everything from my laptop and I'm starting to go through it. Current position.

* I've trained a MTDS model per the paper for Mocap. However this is for n_u = 32 (I've omitted the root joint input, which is essentially passed through - this is in line with the paper but at variance to the actual experiments from what I can tell.)
    * Update paper
    * Re-train model for n_u = 35.... also n_y = 67? See animations.ipynb

* Try to work out what's going on in mocap-mtds-macbook in animations.ipynb. This seems to be different to what I'm expecting, and uses a whole bunch of additional tooling that is breaking. I'm currently trying to avoid this.

* Next steps:
    1. Get something working in animations.ipynb
    2. Port the "something working" to the mocap-mtds project
    3. Retrain models as required
    4. Work out competitor model training
    5. Visualise the output of each model, and ideally get a picture of the latent space.
    6. IDEAL: calculate MSE.

#### Footnotes

<a name="#JuliaRepoFootnote">[1]</a>: The work began originally in [mocap-mtds](https://github.com/ornithos/mocap-mtds), a Julia project. The thinking was that in using PyTorch I could take advantage of hitting CuDNN kernels that were not yet plumbed in in Julia. From talking with Mike Innes, I was actually incorrect in thinking this; and moreover, the MTDS cannot take advantage of these anyway (to the best of my knowledge), as the parameters must be back-propped too. So the existence of this Python project is more of a historical curiosity rather than a necessity.

<a name="#InitDeltaFnFootnote">[2]</a>: We use "low variance" Gaussian distributions instead of delta functions, since the KL term of the ELBO would otherwise blow-up, and it avoids requiring bespoke code for the early stages of optimisation. The posterior distributions are initialised with approx 0.005 stdev, and the stdev is not optimised until after the initial `hard_em_iters` have elapsed.