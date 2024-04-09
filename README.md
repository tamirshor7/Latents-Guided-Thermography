# Latents-Guided-Thermography
#### Tamir Shor (tamir.shor@campus.technion.ac.il), Chaim Baskin, Alex Bronstein
This repo contains official Pytorch implementation for the algorithm proposed in "Leveraging Latents for Efficient Thermography Classification and Segmentation".
Our algorithm focuses on using an expressive latent space in order to perform efficient, accurate and fully automatic tumor benign/malignant classification and novel 7-region semantic segmentation. This approach replaces previous methods focused on complex feature selection pipelines or hevay, difficult to train neural architectures. Using our potent latent space, we show it suffices to train a relatively small, simple decoder network to achieve accurate results over the two downstream tasks. <\br>
Our classification produces SOTA results, while we are the first to solve the 7-region semantic segmentation problem (segmenting left/right breasts, left/right nipples, left/right armpits and neck).

## Set-up
### Data
As a first step you must mount your dataset under the "data" folder. In the paper we used the open [DMR dataset](https://visual.ic.uff.br/dmi/). To recreate our experiments place the DMR folder under the data folder. 
If you wish to use your own dataset, you should also place it under the data folder, however you must also add your custom dataset classes to "src/datasets", similar to "src/datasets/dmr_dataset.py".

### Environment
You can install the conda environment we used with:
<pre>
conda env create  -f environment.yaml
conda activate llft
</pre>

## Training
Our algorithm consists of two decoupled parts - training of the encoder to learn the expressive latent space, and training the decoder to achieve the downstream task.

### Encoder Training
The encoder is trained in an unsupervised fashion, we follow the CUTS contrastive learning-based architecutre proposed in [this paper](https://arxiv.org/abs/2209.11359), implemented in [this repo]([https://arxiv.org/abs/2209.11359](https://github.com/ChenLiu-1996/CUTS)).
In our experiments described in the paper we compare between training the encoder over grayscale thermographies and RGB-heatmap thermographies. The choice between data types is controlled by the --gray flag.
To train the encoder you must run the main_latent.py script as follows:
<pre>
python main_latent.py --gray <if set, train over grayscale data> --cfg-path <path to folder containing config yaml files. ./config by default>
</pre>
If you are using your own custom dataset, make sure to add your config file pointing to this dataset to the config folder. See example in "config/dmr_cfg.yaml".

### Decoder Training
For our decoder we use a basic UNet model implemented under the "model" folder. You can add other architectures there if needed.
In our experiments in the paper we consider two downstream tasks - fully supervised tumor classification and few-shot region semantic segmentation.
For each task we compare between 4 cases using an encoder trained on heatmap/grayscale data, and for each of those using a decoder trained on heatmap/grayscale data.
To train for the classification task:
<pre>
  python main_binary.py --gray_encoder <If set, use encoder trained on grayscale data, else heatmap data> --gray_decoder <if set, use decoder trained on grayscale data, else heatmap data> --cfg-path <path to config folder, "./config" by default> --encoder-ckpt-root <path to folder containing the folders dmr_runs and dmr_runs_gray, containing the respective trained encoder checkpoints>
</pre>

To train the segmentation task:
<pre>
  python main_seg.py --gray_encoder <If set, use encoder trained on grayscale data, else heatmap data> --gray_decoder <if set, use decoder trained on grayscale data, else heatmap data> --cfg-path <path to config folder, "./config" by default> --encoder-ckpt-root <path to folder containing the folders dmr_runs and dmr_runs_gray, containing the respective trained encoder checkpoints>
</pre>
Note that to train the segmentation task, you'll need some segmentations labels under the path "data/DMR/". Our own labels are under this path in this repo ("52_instances_default.json").

## Acknowledgements
 - Our encoder model, as well as the encoder training script and several utility scripts used in our repo, are based on the [official CUTS implementation](https://github.com/ChenLiu-1996/CUTS).
 - We extend our gratitude to [ThermoMind](https://www.thermomind.io/) for supplying us with the segmentation annotations. 
