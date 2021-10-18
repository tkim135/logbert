import os
import sys
import argparse

import pycuda.autoinit
import pycuda.driver as cuda
import multiprocessing
import tensorrt as trt
import numpy as np
import torch
import time
from tqdm import tqdm
from datasets import load_dataset

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


sys.path.insert(1, '/home/ubuntu/TensorRT/samples/python/introductory_parser_samples/..')
import common

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
A100_FLOPS = int(19.5 * 10**12) # https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf

def build_engine_onnx(model_file, batch_size, seq_len, candidate_num, hidden_size, no_head):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.EXPLICIT_BATCH)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # optimization profile
    profile = builder.create_optimization_profile()
    bert_input_shape = (batch_size, seq_len)
    time_input_shape = (batch_size, seq_len, 1)
    profile.set_shape('bert_input', bert_input_shape, bert_input_shape, bert_input_shape)
    profile.set_shape('time_input', time_input_shape, time_input_shape, time_input_shape)
    if no_head:
        bert_output_shape = (batch_size, seq_len, hidden_size)
        profile.set_shape('bert_output', bert_output_shape, bert_output_shape, bert_output_shape)
    else:
        logkey_output_shape = (batch_size, seq_len, candidate_num)
        cls_output_shape = (batch_size, hidden_size)
        profile.set_shape('logkey_output', logkey_output_shape, logkey_output_shape, logkey_output_shape)
        profile.set_shape('cls_output', cls_output_shape, cls_output_shape, cls_output_shape)
    config.add_optimization_profile(profile)

    # config.max_workspace_size = common.GiB(1)
    config.max_workspace_size = common.GiB(9)
    # Load the Onnx model and parse it in order to populate the TensorRT network.
#     import pdb
#     pdb.set_trace()
    with open(model_file, 'rb') as model:
        if not parser.parse(model.read()):
            print ('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print (parser.get_error(error))
            return None
    return builder.build_engine(network, config)

def measure_tensorrt_util(context,
                     bindings,
                     inputs,
                     outputs,
                     stream,
                     batch,
                     batch_size,
                     n_iterations,
                     peak_device_flops):
    print("Measuring inference performance.")
    start = time.time()
    pbar = tqdm(range(n_iterations), total=n_iterations, dynamic_ncols=True)
    for _ in pbar:
        for i in range(len(inputs)):
            np.copyto(inputs[i].host, batch[i].cpu().numpy().ravel())
            # np.copyto(inputs[i].host, batch[i][:, :128].cpu().numpy().ravel())
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    end = time.time()
    print('throughput: {:.4f} samples/s measured over {} iterations.'.format(
        n_iterations * batch_size / (end - start), n_iterations))
