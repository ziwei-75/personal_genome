import numpy as np ;
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import Input, Cropping1D, add, Conv1D, GlobalAvgPool1D, Dense, Add, Concatenate, Lambda, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from chrombpnet.training.utils.losses import *
import tensorflow as tf
import random as rn
import os 

os.environ['PYTHONHASHSEED'] = '0'


def load_pretrained_bias(model_hdf5):
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import get_custom_objects
    custom_objects={"tf": tf, 
                    "multinomial_nll":multinomial_nll,
                    "multi_class_multinomial_nll":multi_class_multinomial_nll}     
    get_custom_objects().update(custom_objects)
    pretrained_bias_model=load_model(model_hdf5,compile=False)
    #freeze the model
    num_layers=len(pretrained_bias_model.layers)
    for i in range(num_layers):
        pretrained_bias_model.layers[i].trainable=False
    return pretrained_bias_model

class CustomModel(tf.keras.Model):
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Calculate loss for each individual sample
        profile_label, count_label = y
        profile_pred, count_pred = y_pred
        num_tasks = count_pred.shape[1]
        metrics = {m.name: m.result() for m in self.metrics}
        for i in range(num_tasks):
            sample_profile_label = profile_label[:,:,i]
            sample_profile_pred = profile_pred[:,:,i]
            sample_count_label = count_label[:,i]
            sample_count_pred = count_pred[:,i]
            sample_mse = tf.keras.metrics.mean_squared_error(sample_count_label,sample_count_pred)
            sample_mll = multinomial_nll(sample_profile_label,sample_profile_pred)
            metrics["sample %s mse"%(i+1)] = sample_mse
            metrics["sample %s multinomial_nll"%(i+1)] = sample_mll
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return metrics



def bpnet_model(filters, n_dil_layers, sequence_len, out_pred_len, num_tasks):

    conv1_kernel_size=21
    profile_kernel_size=75

    #define inputs
    inp = Input(shape=(sequence_len, 4),name='sequence')    

    # first convolution without dilation
    x = Conv1D(filters,
                kernel_size=conv1_kernel_size,
                padding='valid', 
                activation='relu',
                name='wo_bias_bpnet_1st_conv')(inp)

    layer_names = [str(i) for i in range(1,n_dil_layers+1)]
    for i in range(1, n_dil_layers + 1):
        # dilated convolution
        conv_layer_name = 'wo_bias_bpnet_{}conv'.format(layer_names[i-1])
        conv_x = Conv1D(filters, 
                        kernel_size=3, 
                        padding='valid',
                        activation='relu', 
                        dilation_rate=2**i,
                        name=conv_layer_name)(x)

        x_len = int_shape(x)[1]
        conv_x_len = int_shape(conv_x)[1]
        assert((x_len - conv_x_len) % 2 == 0) # Necessary for symmetric cropping

        x = Cropping1D((x_len - conv_x_len) // 2, name="wo_bias_bpnet_{}crop".format(layer_names[i-1]))(x)
        x = add([conv_x, x])

    # Branch 1. Profile prediction
    # Step 1.1 - 1D convolution with a very large kernel
    prof_out_precrop = Conv1D(filters=num_tasks,
                        kernel_size=profile_kernel_size,
                        padding='valid',
                        name='wo_bias_bpnet_prof_out_precrop')(x)

    # Step 1.2 - Crop to match size of the required output size
    cropsize = int(int_shape(prof_out_precrop)[1]/2)-int(out_pred_len/2)
    assert cropsize>=0
    assert (cropsize % 2 == 0) # Necessary for symmetric cropping
    
    # not flatten in multitask mode
    if num_tasks > 1:
        profile_out = Cropping1D(cropsize,
            name='wo_bias_bpnet_logits_profile_predictions')(prof_out_precrop)
    else:
        prof = Cropping1D(cropsize,
            name='wo_bias_bpnet_logitt_before_flatten')(prof_out_precrop)
        profile_out = Flatten(name="wo_bias_bpnet_logits_profile_predictions")(prof)

    # Branch 2. Counts prediction
    # Step 2.1 - Global average pooling along the "length", the result
    #            size is same as "filters" parameter to the BPNet function
    gap_combined_conv = GlobalAvgPool1D(name='gap')(x) # acronym - gapcc

    # Step 2.3 Dense layer to predict final counts
    count_out = Dense(num_tasks, name="wo_bias_bpnet_logcount_predictions")(gap_combined_conv)

    # instantiate keras Model with inputs and outputs
    model=Model(inputs=[inp],outputs=[profile_out, count_out], name="model_wo_bias")

    return model


def getModelGivenModelOptionsAndWeightInits(args, model_params):
    # logdir = args.output_prefix + "tensorboard"
    # file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    # file_writer.set_as_default()
    assert("bias_model_path" in model_params.keys()) # bias model path not specfied for model
    filters=int(model_params['filters'])
    n_dil_layers=int(model_params['n_dil_layers'])
    bias_model_path=model_params['bias_model_path']
    sequence_len=int(model_params['inputlen'])
    out_pred_len=int(model_params['outputlen'])
    num_tasks=int(model_params['num_tasks'])

    if num_tasks > 1:
        counts_loss_weight = [float(c) for c in model_params['counts_loss_weight'].split(",")]
    else:
        counts_loss_weight=float(model_params['counts_loss_weight'])
    print("counts_loss_weight:"+str(counts_loss_weight))

    bias_model = load_pretrained_bias(bias_model_path)
    bpnet_model_wo_bias = bpnet_model(filters, n_dil_layers, sequence_len, out_pred_len,num_tasks)

    #read in arguments
    seed=args.seed
    np.random.seed(seed)    
    tf.random.set_seed(seed)
    rn.seed(seed)
    
    inp = Input(shape=(sequence_len, 4),name='sequence')    

    ## get bias output
    bias_output=bias_model(inp)
    ## get wo bias output
    output_wo_bias=bpnet_model_wo_bias(inp)
    if num_tasks > 1:
        # first one is profile output second one is count
        assert(len(bias_output[0].shape)==3) # bias model profile head is of incorrect shape (None,out_pred_len) expected
        assert(len(bias_output[1].shape)==2) # bias model counts head is of incorrect shape (None,1) expected
        assert(len(output_wo_bias[0].shape)==3)
        assert(len(output_wo_bias[1].shape)==2)
        assert(bias_output[0].shape[1]==out_pred_len) # bias model profile head is of incorrect shape (None,out_pred_len) expected
        assert(bias_output[0].shape[2]==num_tasks) # bias model profile head is of incorrect shape (None,out_pred_len) expected
        assert(bias_output[1].shape[1]==num_tasks) #  bias model counts head is of incorrect shape (None,1) expected
        assert(output_wo_bias[0].shape[1]==out_pred_len) 
        assert(output_wo_bias[0].shape[2]==num_tasks)
        assert(output_wo_bias[1].shape[1]==num_tasks)
    else:
        assert(len(bias_output[1].shape)==2) # bias model counts head is of incorrect shape (None,1) expected
        assert(len(bias_output[0].shape)==2) # bias model profile head is of incorrect shape (None,out_pred_len) expected
        assert(len(output_wo_bias[0].shape)==2)
        assert(len(output_wo_bias[1].shape)==2)
        assert(bias_output[1].shape[1]==1) #  bias model counts head is of incorrect shape (None,1) expected
        assert(bias_output[0].shape[1]==out_pred_len) # bias model profile head is of incorrect shape (None,out_pred_len) expected


    profile_out = Add(name="logits_profile_predictions")([output_wo_bias[0],bias_output[0]])
    print("profile_out",profile_out.shape)
    if num_tasks > 1:
        concat_counts = tf.stack([output_wo_bias[1], bias_output[1]],axis=-1)
        count_out = Lambda(lambda x: tf.math.reduce_logsumexp(x, axis=-1, keepdims=False),
                    name="logcount_predictions")(concat_counts)
    else:
        concat_counts = Concatenate(axis=-1)([output_wo_bias[1], bias_output[1]])
        print("concat_counts.shape",concat_counts.shape)
        count_out = Lambda(lambda x: tf.math.reduce_logsumexp(x, axis=-1, keepdims=True),
                            name="logcount_predictions")(concat_counts)
    print("count_out.shape", count_out.shape)
    # instantiate keras Model with inputs and outputs
    if num_tasks > 1:
        model=CustomModel(inputs=[inp],outputs=[profile_out, count_out])
    else:
        model=Model(inputs=[inp],outputs=[profile_out, count_out])

    if num_tasks > 1:
        model.compile(optimizer=Adam(learning_rate=args.learning_rate),
                        loss=[multi_class_multinomial_nll,weighted_mse_wrapper(counts_loss_weight)])
    else:
        model.compile(optimizer=Adam(learning_rate=args.learning_rate),
                        loss=[multinomial_nll,'mse'],
                        loss_weights=[1,counts_loss_weight])
    return model 


def save_model_without_bias(model, output_prefix):
    model_wo_bias = model.get_layer("model_wo_bias").output
    #counts_output_without_bias = model.get_layer("wo_bias_bpnet_logcount_predictions").output
    model_without_bias = Model(inputs=model.get_layer("model_wo_bias").inputs,outputs=[model_wo_bias[0], model_wo_bias[1]])
    print('save model without bias') 
    model_without_bias.save(output_prefix+"_wo_bias.h5")