# ChromBPNet: Deep learning models of base-resolution chromatin profiles
                This repo is under construction. Please check back in a week

- This repo contains code for the chrombpnet paper Anusri Pampari*, Anna Shcherbina*, Anshul Kundaje. (*authors contributed equally to the work)  
- General queries/thoughts have been addressed in the discussion section below.
- Please contact [Anusri Pampari] (\<first-name\>@stanford.edu)  for suggestions and comments. More instructions about reporting bugs detailed below.
- Authors would like to thank Avanti Shrikumar and Surag Nair for their help with the project.

## Quick Links

- [About](#chrombpnet)
- [Installation](#installation)
- [Tutorial](#tutorial-on-how-to-train-chrombpnet-models)
    - [Preprocessing](#preprocessing)
    - [Train and Evaluate Bias Model](#train-and-evaluate-bias-model)
    - [Train and Evaluate ChromBPNet Model](#train-and-evaluate-chrombpnet-model)

##  ChromBPNet

Chromatin profiles (DNASE-seq and ATAC-seq) exhibit multi-resolution shapes and spans regulated by co-operative binding of transcription factors (TFs). This complexity is further difficult to mine because of confounding bias from enzymes (DNASE-I/Tn5) used in these assays. Existing methods do not account for this complexity at base-resolution and do not account for enzyme bias correctly, thus missing the high-resolution architecture of these profile. Here we introduce ChromBPNet to address both these aspects.

ChromBPNet (shown in the image as `chrombpnet model`) is a fully convolutional neural network that uses dilated convolutions with residual connections to enable large receptive fields with efficient parameterization. It also performs automatic assay bias correction in two steps, first by learning simple model on chromatin background that captures the enzyme effect (called `bias model` in the image). Then we use this model to regress out the effect of the enzyme from the ATAC-seq/Dnase-seq profiles. This two step process ensures that the sequence component of the ChromBPNet model (called `sequence model`) does not learn enzymatic bias. 

![Image](images/chrombpnet_arch.PNG)

If you are interested in learning more about the detailed architectures used, please refer to the following architecture files - 

- bias model: https://github.com/kundajelab/chrombpnet/blob/master/src/training/models/bpnet_model.py
- chrombpnet model: https://github.com/kundajelab/chrombpnet/blob/master/src/training/models/chrombpnetwith_bias_model.py.

## Installation

This section will discuss the packages needed to train a chrombpnet model. Firstly, it is recommended that you use a GPU and have the necessary NVIDIA drivers already installed and setup as chrombpnet model training is faster on a GPU. Secondly there are two ways to ensure you have the necessary packages to run train chrombpnet models which we detail below,

### 1. Installation setup through Docker

```
docker run -it --cpus=8 --memory=100g --gpus device=0  --rm --mount  type=bind,src="chrombpnet_paper",target=/chrombpnet_paper -t vivekramalingam/tf-atlas
```

### 2. Installation setup through Conda

```
pip install -r requirements.txt
```
in a new conda environment preferably. See this link to find the appropriate CUDA and cuDNN versions

TODO -
setup conda environment
setup docker
test setup in a new environment
	
##  Tutorial on how to train chrombpnet models

Here we provide a step-by-step guide to training and evaluating chrombpnet models using the GM12878 ATAC-seq data (ENCSR095QNB) [here][url1] whic is a is bulk ATAC-seq data.

###  Preprocessing

#### Step 1: Download experimental data

We will first start by creating a directory named `data/downloads` and downloading the corresponding files (bams and peak files) for ENCSR095QNB ENCODE dataset using the commands in the bash script below. 

```
mkdir data
mkdir data/downloads
bash step1_download_bams_and_peaks.sh data/downloads
```

Following are some things to keep in mind when using custom datasets/downloads -
- For bulk ATAC-seq/DNASE-seq dataset we use the latest ENCODE ATAC-seq protocol https://github.com/ENCODE-DCC/atac-seq-pipeline. The pipeline outputs both stringent (IDR peaks) and relaxed (Overlap peak) thresholding of peaks across replicates and we use the relaxed thresholding of peaks.
- If you are downloading the data from the ENCODE portal you can download the peaks flagged default for ATAC-seq datasets. For DNASE-seq datasets you might have to use the MACS2 protocol to call peaks on the filtered bams (TODO - provide scripts)
- For paired end data we download the filtered bams output from the pipeline and for single-end data we download the unfiltered bams from the pipeline. Please refer to the documentation below to understand the reason for this difference `src/helpers/preprocessing/`
- TODO - add notes on how will this be different for scATAC


#### Step 2: Make Bigwigs from bam files (IMPORTANT STEP! PLEASE READ CAREFULLY)

We will now create unstranded bigwigs (i.e. the + and - strand ends are combined into one bigwig) from the downloaded bam files (from step1) using the command below. This script uses the following two commands (1) `src/helpers/preprocessing/bam_to_bigwig.sh`: Considers that the given bam files are *unshifted* and does a shift of +4/-4 and (2) `src/helpers/preprocessing/analysis/build_pwm_from_bigwig.py` generates an image of the bias motif and does a sanity check if the shift is correct/incorrect. 

```
bash step2_make_bigwigs_from_bams.sh data/downloads/merged.bam data/downloads/ ATAC_PE data/downloads/hg38.fa data/downloads/hg38.chrom.sizes
```

After running this command open the `bias_pwm.png` image generated by the script in `data/downloads` folder. You will see the following Tn5 motif PWM for this dataset.

![Image](images/bias_pwm.png)

Following are some things to keep in mind when using custom datasets/downloads -

- **IMPORTANT NOTE 1:** If you are running these commands on custom experimental bam - *read the documentation* in the directory `$src/helpers/preprocessing/` and  `$src/helpers/analysis/` to make sure you are using the script correctly. Next use this script `$src/helpers/preprocessing/analysis/` and make sure that the script throws no warnings and that you see Tn5 or DNase-I bias pwm in `bias_pwm.png`. If you do not see this it is likely that the bam's that you provided have a shift of some kind. Please provide only unshifted bams to the script. **Do not proceed further if you do not see a Tn5 or DNase-I motif after this step.** 

- **IMPORTANT NOTE 2:** If you are running the pipeline on custom generated bigwigs (without using `$src/helpers/preprocessing/`) make sure the bigwigs are unstranded and make sure the shifts are done correctly. To check this run the scripts in this directory  `$src/helpers/preprocessing/analysis/` and make sure that the script throws no warnings and you see the Tn5 or DNase-I bias pwm in `bias_pwm.png` output generated. **Do not proceed further if you do not see a Tn5 or DNase-I motif after this step.** 

- TODO - add preprocessing scripts for scATAC datasets.

#### Step 3: Generate background regions gc-matched with the peaks

Here we will generate non-peak background regions that GC-match with the peak regions. We will use the non-peaks regions to train and evaluate a bias model. We will also use these regions in model training and as background regions to get marginal footprints. There are two key steps to this process - 

Firstly, we will start by dividing the entire genome into overlapping bins of `inputlen` regions. ChromBpnet models are trained on `inputlen` of 2114, so we will divide the entire genome into non-overlapping bins of length of 2114 input length and calculate their gc-fraction value. 

For convenience the genome wide buckets we created on human genome (hg38) reference can be downloaded as follows -  

```
wget http://mitra.stanford.edu/kundaje/anusri/chrombpnet_downloads/genomewide_gc_hg38_stride_50_inputlen_2114.bed -O data/downloads/genomewide_gc_hg38_stride_50_inputlen_2114.bed
```
To generate this file directly from the scripts run the command below - 

```
python src/helpers/make_gc_matched_negatives/get_genomewide_gc_buckets/get_genomewide_gc_bins.py -g data/downloads/hg38.fa -c data/downloads/hg38.chrom.sizes -o data/downloads/genomewide_gc_hg38_stride_50_inputlen_2114.bed -il 2114 --s 50
```
NOTE: The script above can take several hours to complete, but it is a one-time run for every reference genome. Please contribute if you know of ways to speed this step.

Secondly, we will filter the regions from the genome-wide buckets created from step 1 such that they do not fall in peak regions or blacklist regions but have similar GC-distribution as the peaks.

```
mkdir data/negatives_data
bash step3_get_background_regions.sh data/downloads/hg38.fa data/downloads/hg38.chrom.sizes data/downloads/blacklist.bed.gz data/negatives_data overlap.bed.gz 2114 genomewide_gc_hg38_stride_50_inputlen_2114.bed
```

Following are some things to keep in mind when using custom datasets -

- To understand all the outputs generated by these scripts refer to the documentation at `src/helpers/make_gc_matched_negatives`

###  Train and Evaluate Bias Model

We are now ready to train a bias model! We will first define how we want to split the dataset by mentioning our splits in `splits.py` and generating the corresponding `json` formatted files. If you want to mention custom splits please edit the `splits.py` file directly. This script is programmed to generate five json files each with a different fold information. You can choose any fold for model training. For the purpose of this tutorial we will use `fold_0.json`

```
mkdir data/splits
python src/helpers/make_chr_splits/splits.py -o data/splits
```

#### Step 4: Train Bias Model

We will train a bias model on the non-peak regions by running the `step4_train_bias_model.sh` command as shown below 

```
mkdir models
mkdir models/bias_model
bash step4_train_bias_model.sh data/hg38.fa data/shifted.sorted.bam.chrombpnet.unstranded.bw data/overlap.bed.gz data/negatives_with_summit.bed  0.9 2114 1000 data/splits/fold_0.json 0.5

```

The script `step4_train_bias_model.sh` runs the following three steps -  
 1. Generate hyperparmeters file for bias model: In this step we will filter non-peaks regions for training and also find some important training hyperparameters. Read the documentation in `src/helpers/hyperparameters` to understand both these steps in detail. As a part of the filtration step we will filter out non-peaks regions of length 2114 which have total counts greater than a threshold (given by `min(total counts in peaks)*bias_threshold_factor`). The `min(total counts in peaks)` is the minimum of the total counts in peak regions of length 2114 and the `bias_threshold_factor` is default set to `0.5` and can be adjusted by the user. This step ensures that the bias model training is done only in sufficiently low-count regions so that the bias model captures only the effect of the bias motif and not the effect of cell-type specific motifs.
 2. Train the bias model using the hyper-paramters and filtered regions from step 1 using the train/valid chromosomes in `data/splits/fold_0.json`. This step will output a bias model  `models/bias_model/bias.h5` in h5py format.
 3. Get predictions on the non-peak regions in test chromosomes in `data/splits/fold_0.json` and report the metrics in `models/bias_model/bias_metrics.json` and the predictions in `models/bias_model/bias_predictions.h5`

Following are some things to keep in mind when using custom datasets -

- **IMPORTANT NOTE 1:** Read the documentation in `src/helpers/hyperparameters` carefully to adapt the bias training hyper-parameters for your datasets. 
    - If `bias_threshold_factor` is set to very low value you might filter out a lot of non-peak regions and the bias model might be sub-optimal (this will be reflected in the final performance metrics output by the model - poor jsd performance might imply a sub-optimal bias model). 
    - If `bias_threshold_factor` is set to very high you might include non-peak regions with high counts and this might lead to the bias model capturing cell-type specific motifs too (which is not ideal as we want to regress out only the bias motifs effect and not cell-type specific motifs effect). We will do Step 5 of the tutorial to make sure we capture only bias motifs. If the output in Step 5 shows cell-type specific motifs in addition to the bias motifs you have to repeat bias model training with a lower `bias_threshold_factor`.

#### Step 5: Interpret bias model

In this step we will try to summarize the motifs captured by the bias model by first getting contribution scores (using DeepSHAP) of the bias model in peak regions and then summarizing the contribution scores as PWMs (using TF-MoDISCO algorithm) to identify motifs. For the bias model we expect to find PWMs of only the bias enzyme (Tn5/DNASE-I). If we see any other cell-type specific motifs apart form the bias enzymes me will need to re-do Step 4 with reduced `bias_threshold_factor`.

NOTE: It is computationally expensive to run this step on all the peaks, since we are looking for a quick sanity check for our bias models we will subsample 30K peaks from our original peak set for interpretation. Additionally we will also make sure that this peak set does not overlap with the blacklist regions.

```
inputlen=2114
blacklist_region=data/downloads/blacklist.bed.gz
chrom_sizes=data/downloads/hg38.chrom.sizes
overlap_peak=data/downloads/overlap.bed.gz

mkdir data/subsample_peaks

flank_size=$(( inputlen/2 ))
bedtools slop -i $blacklist_region -g $chrom_sizes -b $flank_size > data/subsample_peaks/temp.txt
bedtools intersect -v -a $overlap_peak -b data/subsample_peaks/temp.txt | shuf  > data/subsample_peaks/temp_n.txt
shuf -n 30000 data/subsample_peaks/temp_n.txt > data/subsample_peaks/30K.subsample.overlap.bed
rm  data/subsample_peaks/temp.txt
rm data/subsample_peaks/temp_n.txt
```

Once we have the subsampled peak set for interpretation we will run the script below to first get interpretation scores and then summarize the interpretation scores.

```
bash step5_interpret_bias_model.sh  data/hg38.fa data/subsample_overlap/30K.subsample.overlap.bed  output/bias_model/model.0.h5
```

The step will generate . browse through the images in both these folders and make sure you only see the bias motif (Tn5/DNase-I) and not any other cell-type specific motifs.

###  Train and Evaluate ChromBPNet Model

Now that we have a bias model we will use it to regress out the bias enzymes effect from the ATAC-seq and DNAE-seq signal so that the sequence component of the model can capture only the cell-type specific behavior. 

#### Step 6: Train ChromBPNet Model (This step will also generate the sequence model)

We will use the bias model trained in Step 4 to regress out the effect of the bias enzyme,

```
bash step6_train_chrombpnet_model.sh data/downloads/hg38.fa data/downloads/shifted.sorted.bam.chrombpnet.unstranded.bw data/downloads/overlap.bed.gz data/downloads/negatives_with_summit.bed data/splits/fold_0.json models/bias_model/bias.h5

```

The script `step4_train_bias_model.sh` runs the following three steps -  
 1. Generate hyperparmeters file for bias model: In this step we will filter non-peaks regions for training and also find some important training hyperparameters. Read the documentation in `src/helpers/hyperparameters` to understand both these steps in detail. As a part of the filtration step we will filter out non-peaks regions of length 2114 which have total counts greater than a threshold (given by `min(total counts in peaks)*bias_threshold_factor`). The `min(total counts in peaks)` is the minimum of the total counts in peak regions of length 2114 and the `bias_threshold_factor` is default set to `0.5` and can be adjusted by the user. This step ensures that the bias model training is done only in sufficiently low-count regions so that the bias model captures only the effect of the bias motif and not the effect of cell-type specific motifs.
 2. Train the bias model using the hyper-paramters and filtered regions from step 1 using the train/valid chromosomes in `data/splits/fold_0.json`. This step will output a bias model  `models/bias_model/bias.h5` in h5py format.
 3. Get predictions on the non-peak regions in test chromosomes in `data/splits/fold_0.json` and report the metrics in `models/bias_model/bias_metrics.json` and the predictions in `models/bias_model/bias_predictions.h5`

This will output 2 models one is chrombpnet model (this is the model with bias) and the other is the sequence model (this is the moel without the bias)

Following are some things to keep in mind when using custom datasets -
- 

#### Step 7: Interpret chrombpnet model and sequence model

We will now interpret on the same peak regions that we used for interpreting the bias model.

```
bash step5_interpret_bias_model.sh  data/hg38.fa data/subsample_overlap/30K.subsample.overlap.bed  output/chrombpnet_model/model_wo_bias.0.h5
```

In the images produced here you should not see the bias motif and should only see the cell-type specfic motifs in the profile modisco.
