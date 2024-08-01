import argparse
import pyfaidx
import pyBigWig
import pandas as pd
import numpy as np
import os
from chrombpnet.helpers.hyperparameters import param_utils as param_utils
from tensorflow import keras
import json
import importlib
from tqdm import tqdm

def parse_data_args():
    parser=argparse.ArgumentParser(description="find hyper-parameters for chrombpnet defined in src/training/models/chrombpnet_with_bias_model.py")
    parser.add_argument("-g", "--genome", type=str, required=True, help="Genome fasta")
    parser.add_argument("-i", "--bigwig", type=str, required=True, help="Bigwig of tn5 insertions. Ensure it is +4/-4 shifted")
    parser.add_argument("-p", "--peaks", type=str, required=True, help="10 column bed file of peaks. Sequences and labels will be extracted centered at start (2nd col) + summit (10th col).")
    parser.add_argument("-n", "--nonpeaks", type=str, required=True, help="10 column bed file of non-peak regions, centered at summit (10th column)")
    parser.add_argument("-sr", "--negative_sampling_ratio", type=float, default=0.1, help="Ratio of negatives to positive samples per epoch")
    parser.add_argument("-oth", "--outlier-threshold", type=float, default=0.9999, help="threshold to use to filter outlies")
    parser.add_argument("-j", "--max-jitter", type=int, required=True, default=500, help="Maximum jitter applied on either side of region (default 500 for chrombpnet model)")
    parser.add_argument("-fl", "--chr-fold-path", type=str, required=True, help="Fold information - dictionary with test,valid and train keys and values with corresponding chromosomes")
    return parser

def parse_model_args(parser):
    # arguments here defined the following model - src/training/models/chrombpnet_with_bias_model.py
    parser.add_argument("-il", "--inputlen", type=int, required=True, help="Sequence input length")
    parser.add_argument("-ol", "--outputlen", type=int, required=True, help="Prediction output length")
    parser.add_argument("-fil", "--filters", type=int, default=512, help="Number of filters to use in chrombpnet mode")
    parser.add_argument("-dil", "--n-dilation-layers", type=int, default=8, help="Number of dilation layers to use in chrombpnet model")
    parser.add_argument("-b", "--bias-model-path", type=str, required=True, help="path of bias model")
    parser.add_argument("-op", "--output-prefix", help="output prefix for storing hyper-param TSV for chrombpnet")
    args = parser.parse_args()
    return args

# def adjust_bias_model_logcounts(bias_model, seqs, cts):
#     """
#     Given a bias model, sequences and associated counts, the function adds a 
#     constant to the output of the bias_model's logcounts that minimises squared
#     error between predicted logcounts and observed logcounts (infered from 
#     cts). This simply reduces to adding the average difference between observed 
#     and predicted to the "bias" (constant additive term) of the Dense layer.
#     Typically the seqs and counts would correspond to training nonpeak regions.
#     ASSUMES model_bias's last layer is a dense layer that outputs logcounts. 
#     This would change if you change the model.
#     """

#     # safeguards to prevent misuse
#     #assert(bias_model.layers[-1].name == "logcount_predictions")
#     assert(bias_model.layers[-1].name == "logcounts" or bias_model.layers[-1].name == "logcount_predictions")
#     assert(bias_model.layers[-1].output_shape==(None,1))
#     assert(isinstance(bias_model.layers[-1], keras.layers.Dense))

#     print("Predicting within adjust counts")
#     _, pred_logcts = bias_model.predict(seqs, verbose=True)
#     delta = np.mean(np.log(1+cts) - pred_logcts.ravel())

#     dw, db = bias_model.layers[-1].get_weights()
#     bias_model.layers[-1].set_weights([dw, db+delta])
#     return bias_model

def adjust_bias_model_logcounts(bias_model, seqs, cts,num_tasks=1):
    """
    Given a bias model, sequences and associated counts, the function adds a 
    constant to the output of the bias_model's logcounts that minimises squared
    error between predicted logcounts and observed logcounts (infered from 
    cts). This simply reduces to adding the average difference between observed 
    and predicted to the "bias" (constant additive term) of the Dense layer.
    Typically the seqs and counts would correspond to training nonpeak regions.
    ASSUMES model_bias's last layer is a dense layer that outputs logcounts. 
    This would change if you change the model.
    """

    # safeguards to prevent misuse
    #assert(bias_model.layers[-1].name == "logcount_predictions")
    assert(bias_model.layers[-1].name == "logcounts" or bias_model.layers[-1].name == "logcount_predictions")
    assert(bias_model.layers[-1].output_shape==(None,num_tasks))
    assert(isinstance(bias_model.layers[-1], keras.layers.Dense))

    print("Predicting within adjust counts")
    if num_tasks > 1:
        # avoid out of memory issue
        bs = 256
        delta = []
        for i in range(0, len(seqs),bs):
            current_seq = seqs[i:i+bs]
            current_cts = cts[i:i+bs]
            _, pred_logcts = bias_model.predict(current_seq, verbose=True)
            # precompute the delta for each batch
            d = np.mean(np.log(1+current_cts) - pred_logcts,axis=0)
            delta += [d]
        delta = np.stack(delta,axis=-1)
        delta = np.mean(delta,axis=-1)
    else:
        _, pred_logcts = bias_model.predict(seqs, verbose=True)
        delta = np.mean(np.log(1+cts) - pred_logcts.ravel())

    dw, db = bias_model.layers[-1].get_weights()
    bias_model.layers[-1].set_weights([dw, db+delta])
    return bias_model

def load_single_bias_weight_to_multitask_bias(pretrained_bias_model,args):
    architecture_from_file = "/oak/stanford/groups/akundaje/ziwei75/personal_genomics/src/personal_genome_branch_population_pool/chrombpnet/training/models/bpnet_model.py"
    architecture_module=importlib.machinery.SourceFileLoader('',architecture_from_file).load_module()

    bias_n_dil_layers = 0
    for l in pretrained_bias_model.layers:
        if 'conv' in l.name:
            bias_n_dil_layers += 1
    bias_n_dil_layers -= 1

    bias_params = {}
    bias_params["num_tasks"]=int(args.num_tasks)
    bias_params["filters"]=pretrained_bias_model.layers[2].filters
    bias_params["n_dil_layers"]=bias_n_dil_layers
    assert pretrained_bias_model.layers[0].name == 'sequence'
    bias_params["inputlen"]=pretrained_bias_model.input_shape[1]

    # assume the first outut is the profile
    assert len(pretrained_bias_model.output_shape) == 2
    assert pretrained_bias_model.output_shape[1][1] == 1
    bias_params["outputlen"]=pretrained_bias_model.output_shape[0][1]
    bias_params["bias"]=True

    ### create a bias model with multitask
    bias_model=architecture_module.getModelGivenModelOptionsAndWeightInits(args, bias_params,load_pretrain=True)

    ### transfer weights for the backbond
    for i in range((len(pretrained_bias_model.layers) - 1)):
        weights = pretrained_bias_model.layers[i].get_weights()
        bias_model.layers[i].set_weights(weights)
        
    ### transfer the weights for the count head by duplciating the weights 
    assert (pretrained_bias_model.layers[-1].name == "logcount_predictions" or pretrained_bias_model.layers[-1].name == "logcounts")
    assert (bias_model.layers[-1].name == "logcount_predictions" or pretrained_bias_model.layers[-1].name == "logcounts")

    num_tasks = bias_model.layers[-1].get_weights()[1].shape[0]
    tiled_count_weight = np.tile(pretrained_bias_model.layers[-1].get_weights()[0],num_tasks)
    tiled_count_bias = np.tile(pretrained_bias_model.layers[-1].get_weights()[1],num_tasks) 
    bias_model.layers[-1].set_weights([tiled_count_weight,tiled_count_bias])
    
    print("creted a multitask bias from single task bias")
    print(bias_model.summary())
    return bias_model

def main(args): 

    # read the fold information - we will evaluate hyperparams on the train+valid set and do nothing on the test set 
    splits_dict=json.load(open(args.chr_fold_path))
    chroms_to_keep=splits_dict["train"]+splits_dict["valid"]
    test_chroms_to_keep=splits_dict["test"]
    print("evaluating hyperparameters on the following chromosomes",chroms_to_keep)

    # read from bigwigw and fasta file
    # bw = pyBigWig.open(args.bigwig) 
    genome = pyfaidx.Fasta(args.genome)

    # read peaks and non peaks    
    in_peaks =  pd.read_csv(args.peaks,
                           sep='\t',
                           header=None,
                           names=["chr", "start", "end", "1", "2", "3", "4", "5", "6", "summit"])
    in_nonpeaks =  pd.read_csv(args.nonpeaks,
                           sep='\t',
                           header=None,
                           names=["chr", "start", "end", "1", "2", "3", "4", "5", "6", "summit"])

    assert(in_peaks.shape[0] != 0) # peaks file is empty
    assert(in_nonpeaks.shape[0] !=0) # non peaks file is empty
    assert(args.inputlen >= args.outputlen) # inputlen should be greater than the outputlen 
                                            # inputlen and outlen are chosen based on the filters and dilations layers used

    # get train/valid peaks and test peaks seperately
    peaks = in_peaks[(in_peaks["chr"].isin(chroms_to_keep))]
    test_peaks = in_peaks[(in_peaks["chr"].isin(test_chroms_to_keep))]

    nonpeaks = in_nonpeaks[(in_nonpeaks["chr"].isin(chroms_to_keep))]
    test_nonpeaks = in_nonpeaks[(in_nonpeaks["chr"].isin(test_chroms_to_keep))]

    bigwig_list = args.bigwig_list
    args.num_tasks=1
    bw = pyBigWig.open(bigwig_list[0]) 

    # step 1 filtering: filter peaks that are in the edges - which prevents us from making the inputlen regions - do this for all splits
    peaks = param_utils.filter_edge_regions(peaks, bw, args.inputlen+2*args.max_jitter, peaks_bool=1)
    test_peaks = param_utils.filter_edge_regions(test_peaks, bw, args.inputlen, peaks_bool=1)
   
    nonpeaks = param_utils.filter_edge_regions(nonpeaks, bw, args.inputlen, peaks_bool=0)
    test_nonpeaks = param_utils.filter_edge_regions(test_nonpeaks, bw, args.inputlen, peaks_bool=0)

    # step 2 filtering: filter peaks that are outliers in train and valid set - no filtering on test set
    
    if int(args.num_tasks) > 1:
        peak_cnts, nonpeak_cnts = [], []
        for bw in bigwig_list:
            bw = pyBigWig.open(bw)
            p_c, peak_seqs = param_utils.get_seqs_cts(genome, bw, peaks, args.inputlen, args.outputlen)
            np_c, nonpeak_seqs = param_utils.get_seqs_cts(genome, bw, nonpeaks, args.inputlen, args.outputlen)    
            peak_cnts += [p_c]
            nonpeak_cnts += [np_c]
        peak_cnts = np.stack(peak_cnts,axis=1)
        nonpeak_cnts = np.stack(nonpeak_cnts,axis=1)
        assert(peak_cnts.shape[0] == peaks.shape[0])
        assert(nonpeak_cnts.shape[0] == nonpeaks.shape[0])
    else:
        if len(args.bigwig_list) > 1:
            peak_cnts, nonpeak_cnts = [], []
            for bw in tqdm(bigwig_list):
                bw = pyBigWig.open(bw)
                p_c, peak_seqs = param_utils.get_seqs_cts(genome, bw, peaks, args.inputlen, args.outputlen)
                np_c, nonpeak_seqs = param_utils.get_seqs_cts(genome, bw, nonpeaks, args.inputlen, args.outputlen)    
                peak_cnts += [p_c]
                nonpeak_cnts += [np_c]
            peak_cnts = np.stack(peak_cnts,axis=1)
            nonpeak_cnts = np.stack(nonpeak_cnts,axis=1)
            peak_cnts = np.average(peak_cnts,axis=1)
            nonpeak_cnts = np.average(nonpeak_cnts,axis=1)
            print(peak_cnts.shape,nonpeak_cnts.shape)
            # peak_cnts, peak_seqs = param_utils.get_seqs_cts(genome, bw, peaks, args.inputlen, args.outputlen)
            # nonpeak_cnts, nonpeak_seqs = param_utils.get_seqs_cts(genome, bw, nonpeaks, args.inputlen, args.outputlen)    
            assert(len(peak_cnts) == peaks.shape[0])
            assert(len(nonpeak_cnts) == nonpeaks.shape[0])
        else:
            peak_cnts, peak_seqs = param_utils.get_seqs_cts(genome, bw, peaks, args.inputlen, args.outputlen)
            nonpeak_cnts, nonpeak_seqs = param_utils.get_seqs_cts(genome, bw, nonpeaks, args.inputlen, args.outputlen)    
            assert(len(peak_cnts) == peaks.shape[0])
            assert(len(nonpeak_cnts) == nonpeaks.shape[0])

    if args.negative_sampling_ratio > 0:
        if int(args.num_tasks) > 1:
            nonpeak_index = np.random.randint(nonpeak_cnts.shape[0], size=(int(args.negative_sampling_ratio*len(peak_cnts))))
            final_cnts = np.concatenate((peak_cnts,nonpeak_cnts[nonpeak_index,:]))
        else:
            final_cnts = np.concatenate((peak_cnts,np.random.choice(nonpeak_cnts, replace=False, size=(int(args.negative_sampling_ratio*len(peak_cnts))))))

    else:
        final_cnts = peak_cnts

    upper_thresh = np.quantile(final_cnts, args.outlier_threshold,axis=0)
    lower_thresh = np.quantile(final_cnts, 1-args.outlier_threshold,axis=0)

    if int(args.num_tasks) > 1:
        assert len(upper_thresh) == args.num_tasks
        peaks = peaks[np.all(peak_cnts < upper_thresh,axis=1) & np.all(peak_cnts > lower_thresh,axis=1)]
        nonpeaks = nonpeaks[np.all(nonpeak_cnts < upper_thresh,axis=1) & np.all(nonpeak_cnts > lower_thresh,axis=1)]
    else:
        peaks = peaks[(peak_cnts< upper_thresh) & (peak_cnts>lower_thresh)]
        nonpeaks = nonpeaks[(nonpeak_cnts< upper_thresh) & (nonpeak_cnts>lower_thresh)]

    print("Number of peaks after removing outliers: ", peaks.shape[0])
    print("Number of nonpeaks after removing outliers: ", nonpeaks.shape[0])

    # combine train valid and test peak set and store them in a new file
    if nonpeaks.shape[0] > peaks.shape[0]:
        train_nonpeaks = nonpeaks.sample(n=peaks.shape[0], random_state=1)
    else:
        train_nonpeaks = nonpeaks
        
    frames = [peaks, test_peaks]
    all_peaks = pd.concat(frames)
    all_peaks.to_csv("{}filtered.peaks.bed".format(args.output_prefix), sep="\t",  header=False, index=False)
    frames = [train_nonpeaks, test_nonpeaks]
    all_nonpeaks = pd.concat(frames)
    all_nonpeaks.to_csv("{}filtered.nonpeaks.bed".format(args.output_prefix), sep="\t", header=False, index=False)

    # find counts loss weight for model training - using train and validation set
    # counts_loss_weight = np.median(final_cnts[(final_cnts <= upper_thresh) & (final_cnts>=lower_thresh)])/10
    # assert(counts_loss_weight != 0)
    # find counts loss weight for model training - using train and validation set
    if int(args.num_tasks) > 1:
        final_cnts = final_cnts[np.all(final_cnts < upper_thresh,axis=1) & np.all(final_cnts > lower_thresh,axis=1)]
        counts_loss_weight = np.median(final_cnts,axis=0)/10
        assert np.all(counts_loss_weight > 1.0)
        assert np.all(counts_loss_weight != 0)
    else:
        counts_loss_weight = np.median(final_cnts[(final_cnts <= upper_thresh) & (final_cnts>=lower_thresh)])/10
        assert(counts_loss_weight != 0)
        assert(counts_loss_weight > 1.0) # counts loss weight can go less than 1.0 if you have very low-read depth - make sure you have enough density in counts
                                     # check peak-calling
    
    #assert(counts_loss_weight > 1.0) # counts loss weight can go less than 1.0 if you have very low-read depth - make sure you have enough density in counts
                                     # check peak-calling

    # if counts_loss_weight < 1.0:
    #     counts_loss_weight = 1.0
    #     print("WARNING: you are training on low-read depth data")
    
    # adjust bias model for training  - using train and validation set
    # the bias model might be trained on a difference read depth compared to the given data - so this step scales the bias model to account for that
    # bias_model = param_utils.load_model_wrapper(args.bias_model_path)
    # bias_model_scaled = adjust_bias_model_logcounts(bias_model, nonpeak_seqs[(nonpeak_cnts< upper_thresh) & (nonpeak_cnts>lower_thresh)], nonpeak_cnts[(nonpeak_cnts< upper_thresh) & (nonpeak_cnts>lower_thresh)],\
    #     len(args.sample_list))
    # # save the new bias model
    # bias_model_scaled.save("{}bias_model_scaled.h5".format(args.output_prefix))

    
    ### bias_model_path = os.path.join(args.output_dir, "bias_model_scaled.h5")
    ### For multitask model, previous bias model can be a single task model or multitask model
    # adjust bias model for training  - using train and validation set
    if int(args.num_tasks) > 1:
        pretrained_bias_model = param_utils.load_model_wrapper(args.bias_model_path)
        bias_model_path = os.path.join(args.output_dir, "bias_model_scaled.h5")
        
        ### For multitask model, previous bias model can be a single task model or multitask model
        pretrained_bias_model = param_utils.load_model_wrapper(args.bias_model_path)
        assert ((pretrained_bias_model.layers[-1].name == "logcount_predictions") or (pretrained_bias_model.layers[-1].name == "logcounts"))
        previous_num_tasks = pretrained_bias_model.output_shape[1][1]
        if previous_num_tasks == 1:
            bias_model = load_single_bias_weight_to_multitask_bias(pretrained_bias_model)
        else:
            assert previous_num_tasks == int(args.num_tasks)
            bias_model = pretrained_bias_model
         
        adjust_seqs = nonpeak_seqs[np.all(nonpeak_cnts < upper_thresh,axis=1) & np.all(nonpeak_cnts > lower_thresh,axis=1)]
        adjust_cts = nonpeak_cnts[np.all(nonpeak_cnts < upper_thresh,axis=1) & np.all(nonpeak_cnts > lower_thresh,axis=1)]
        bias_model_scaled = adjust_bias_model_logcounts(bias_model, adjust_seqs,adjust_cts,args.num_tasks)
    else:
        # the bias model might be trained on a difference read depth compared to the given data - so this step scales the bias model to account for that
        bias_model = param_utils.load_model_wrapper(args.bias_model_path)
        bias_model_scaled = adjust_bias_model_logcounts(bias_model, nonpeak_seqs[(nonpeak_cnts< upper_thresh) & (nonpeak_cnts>lower_thresh)], nonpeak_cnts[(nonpeak_cnts< upper_thresh) & (nonpeak_cnts>lower_thresh)])
    # save the new bias model
    bias_model_scaled.save("{}bias_model_scaled.h5".format(args.output_prefix))

    # store the parameters being used  - in a TSV file
        # store the parameters being used  - in a TSV file
    
    file = open("{}chrombpnet_data_params.tsv".format(args.output_prefix),"w")
    if int(args.num_tasks) > 1:
        file.write("\t".join(["counts_sum_min_thresh", ",".join([str(round(l_th,2)) for l_th in list(lower_thresh)])]))
        file.write("\n")
        file.write("\t".join(["counts_sum_max_thresh", ",".join([str(round(u_th,2)) for u_th in list(upper_thresh)])]))
        file.write("\n")
        file.write("\t".join(["trainings_pts_post_thresh", str(sum((np.all(final_cnts<upper_thresh,axis=1) & np.all(final_cnts>lower_thresh, axis=1))))]))
        file.write("\n")
    else:
        file.write("\t".join(["counts_sum_min_thresh", str(round(lower_thresh,2))]))
        file.write("\n")
        file.write("\t".join(["counts_sum_max_thresh", str(round(upper_thresh,2))]))
        file.write("\n")
        file.write("\t".join(["trainings_pts_post_thresh", str(sum((final_cnts<upper_thresh) & (final_cnts>lower_thresh)))]))
        file.write("\n")
    file.close()

    file = open("{}chrombpnet_model_params.tsv".format(args.output_prefix),"w")
    if int(args.num_tasks) > 1:
        file.write("\t".join(["counts_loss_weight", ",".join([str(round(c_t,2)) for c_t in counts_loss_weight])]))
    else:
        file.write("\t".join(["counts_loss_weight", str(round(counts_loss_weight,2))]))
    file.write("\n")
    file.write("\t".join(["filters", str(args.filters)]))
    file.write("\n")
    file.write("\t".join(["n_dil_layers", str(args.n_dilation_layers)]))
    file.write("\n")
    file.write("\t".join(["bias_model_path", "{}bias_model_scaled.h5".format(args.output_prefix)]))
    file.write("\n")
    file.write("\t".join(["inputlen", str(args.inputlen)]))
    file.write("\n")
    file.write("\t".join(["outputlen", str(args.outputlen)]))
    file.write("\n")
    file.write("\t".join(["max_jitter", str(args.max_jitter)]))
    file.write("\n")
    file.write("\t".join(["chr_fold_path", str(args.chr_fold_path)]))
    file.write("\n")
    file.write("\t".join(["negative_sampling_ratio", str(args.negative_sampling_ratio)]))
    file.write("\n")
    file.write("\t".join(["num_tasks", str(args.num_tasks)]))
    file.close()

if __name__=="__main__":
    # read the arguments
    parser = parse_data_args()
    args = parse_model_args(parser)

    main(args)

    
