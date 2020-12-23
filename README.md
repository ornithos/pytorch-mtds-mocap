
## Human Motion Prediction using Multi-Task Dynamical System

This repo started life via the PyTorch clone of the Martinez et al. (2018) model.<sup>[[1]](#JuliaFootnote)</sup> Note that the PyTorch clone keeps a lot of the command line options of the original codebase but silently removes the features -- I think the codebase here therefore suffers from some of the same problems --  caveat emptor. A major problem with the library as it stands is that I'm unable to obtain the final scripts used to execute the code as these were run locally from my development machine which I haven't had access to in 9 months due to the Covid pandemic. There has been strictly no access granted since March 2020 and I'm writing as of December 2020. I'm therefore only able to guide in the right direction.

### Data

**The long way**: Download Ian Mason's mocap data linked from his [Github repository](https://github.com/ianxmason/Fewshot_Learning_of_Homogeneous_Human_Locomotion_Styles#training-the-models) for the paper "Few-shot Learning of Homogeneous Human Locomotion Styles", unzip it, and place it in a convenient directory. A substantial amount of pre-processing was performed on this data (Ian Mason used a bunch of scripts written within Unity -- this is not a route I wanted to go down) -- see my [mocap-mtds](https://github.com/ornithos/mocap-mtds) Julia library for the pre-processing scripts. In particular, see the [`2_preprocess.ipynb`](https://github.com/ornithos/mocap-mtds/blob/master/2_preprocess.ipynb) notebook.

**The short way**: I've uploaded my processed files here: [https://bit.ly/38x2sra](https://bit.ly/38x2sra). This includes three files for inputs (`Us`), outputs (`Ys`) and a style lookup dict which maps the style label (in {1,2,...,8}) to the indices of the inputs/outputs.

### Training

To train the model, I recommend using the two-layer architecture we describe in our upcoming paper. This uses a 1024-hidden-unit GRU as the first layer, and a multi-task 128-hidden-unit RNN (MTDS) in the second. We fix the dynamic bias _b_ of the MTDS in order to permit smooth interpolation between styles. This can be achieved using the `src/learn_mtfixbmodel.py` file: have a look at the options using `python learn_mtfixbmodel.py -h`. The default values have been extracted from the `argparse` code and stashed in the `src/default.ini` file. I recommend having a look at this to see reasonable values of the parameters which were used in the experiments, including the architecture hyperparameters. A value of `-1` in this file indicates that a default is derived from other args in the code if not specified. Note that the *h\_phi* generator network here is confusingly referred to as `psi` - a metonym, but using a different letter for the parameter than the new paper (which uses `phi`).


To train a model over all 
```bash
python src/learn_mtfixbmodel.py --style_ix 99 --latent_k 7 --input_size 64 --bottleneck 24 --iterations 20000 --hard_em_iters 10000 \
  --learning_rate 3e-5 --learning_rate_mt 1e-3 --learning_rate_z 1e-3 --psi_affine
```

The `style_ix` flag specifies the style index of the test set; i.e. to be held-out for the LOO experiments. If no data are to be held out (e.g. for style transfer purposes), then specify a number greater than 8 (a sensible default I used was 99). Other useful flags include:

```
--use_cpu      # use CPU rather than GPU
--data_dir     # specify the directory in which to find the Us, Ys, and style_lkp
```

## Visualization

Again, Ian Mason uses Unity - I've hacked a bunch of utilities together in Julia to visualize the result in the browser. These make use of [`three.js`](https://threejs.org/) and [`MeshCat.jl`](https://github.com/rdeits/MeshCat.jl).

