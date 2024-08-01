import numpy as np ;
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import Input, Cropping1D, add, Conv1D, GlobalAvgPool1D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from chrombpnet.training.utils.losses import *
import tensorflow as tf
import random as rn
import os 

os.environ['PYTHONHASHSEED'] = '0'

def getModelGivenModelOptionsAndWeightInits(args, model_params,load_pretrain=False):
    #default params (can be overwritten by providing model_params file as input to the training function)
    num_tasks=int(model_params["num_tasks"])
    filters=int(model_params['filters'])
    n_dil_layers=int(model_params['n_dil_layers'])
    sequence_len=int(model_params["inputlen"])
    out_pred_len=int(model_params["outputlen"])
    bias=bool(model_params["bias"])
    
    conv1_kernel_size=21
    profile_kernel_size=75
    
    if load_pretrain == False:
        if num_tasks > 1:
            counts_loss_weight = [float(c) for c in model_params['counts_loss_weight'].split(",")]
        else:
            counts_loss_weight=float(model_params['counts_loss_weight'])
        print("counts_loss_weight:"+str(counts_loss_weight))
        
    print("params:")
    print("filters:"+str(filters))
    print("n_dil_layers:"+str(n_dil_layers))
    print("conv1_kernel_size:"+str(conv1_kernel_size))
    print("profile_kernel_size:"+str(profile_kernel_size))
    
    #read in arguments
    if load_pretrain == False:
        seed=args.seed
        np.random.seed(seed)    
        tf.random.set_seed(seed)
        rn.seed(seed)

    #define inputs
    inp = Input(shape=(sequence_len, 4),name='sequence')    

    # first convolution without dilation
    x = Conv1D(filters,
                kernel_size=conv1_kernel_size,
                padding='valid', 
                activation='relu',
                name='bpnet_1st_conv')(inp)

    layer_names = [str(i) for i in range(1,n_dil_layers+1)]
    for i in range(1, n_dil_layers + 1):
        # dilated convolution
        conv_layer_name = 'bpnet_{}conv'.format(layer_names[i-1])
        conv_x = Conv1D(filters, 
                        kernel_size=3, 
                        padding='valid',
                        activation='relu', 
                        dilation_rate=2**i,
                        name=conv_layer_name)(x)

        x_len = int_shape(x)[1]
        conv_x_len = int_shape(conv_x)[1]
        assert((x_len - conv_x_len) % 2 == 0) # Necessary for symmetric cropping

        x = Cropping1D((x_len - conv_x_len) // 2, name="bpnet_{}crop".format(layer_names[i-1]))(x)
        x = add([conv_x, x])

    # Branch 1. Profile prediction
    # Step 1.1 - 1D convolution with a very large kernel
    if bias:
        prof_out_precrop = Conv1D(filters=1,
                            kernel_size=profile_kernel_size,
                            padding='valid',
                            name='prof_out_precrop')(x)
    else:
        prof_out_precrop = Conv1D(filters=num_tasks,
                                  kernel_size=profile_kernel_size,
                                  padding='valid',
                                  name='prof_out_precrop')(x)

    # Step 1.2 - Crop to match size of the required output size
    cropsize = int(int_shape(prof_out_precrop)[1]/2)-int(out_pred_len/2)
    assert cropsize>=0
    assert (cropsize % 2 == 0) # Necessary for symmetric cropping
    prof = Cropping1D(cropsize,
                name='logits_profile_predictions_preflatten')(prof_out_precrop)

    # Branch 2. Counts prediction
    # Step 2.1 - Global average pooling along the "length", the result
    #            size is same as "filters" parameter to the BPNet function

    if num_tasks > 1:
        # bias model share the same profile head in multitask mode
        if bias:
            profile_out = []
            for i in range(num_tasks):
                profile_out += [tf.squeeze(prof,axis=2)]
            profile_out = tf.stack(profile_out,axis=2,name="logits_profile_predictions")
            print("profile_out shape:",profile_out.shape)
        else:
            profile_out = prof
    else:
        profile_out = Flatten(name="logits_profile_predictions")(prof)
        
    gap_combined_conv = GlobalAvgPool1D(name='gap')(x) # acronym - gapcc

    # Step 2.3 Dense layer to predict final counts
    count_out = Dense(num_tasks, name="logcount_predictions")(gap_combined_conv)

    # instantiate keras Model with inputs and outputs
    print(profile_out.shape)
    model=Model(inputs=[inp],outputs=[profile_out, count_out])

    if load_pretrain == False:
        if num_tasks > 1:
            model.compile(optimizer=Adam(learning_rate=args.learning_rate),
                          loss=[multi_class_multinomial_nll,weighted_mse_wrapper(counts_loss_weight)])
        else:
            model.compile(optimizer=Adam(learning_rate=args.learning_rate),
                            loss=[multinomial_nll,'mse'],
                            loss_weights=[1,counts_loss_weight])

    return model 

def save_model_without_bias(model, output_prefix):
    # nothing to do 
    # all model architectures have this function
    # defining this tosafeguard if the users uses the arugument save_model_without_bias argument on bias model accidentally 
    return