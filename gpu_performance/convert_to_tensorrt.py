import os
import torch
from typing import Tuple


def torch2onnx(model: torch.nn.Module,
                            inputs: Tuple[torch.Tensor, ...],
                            batch_size: int) -> None:
    """
    convert torch bootleg model into onnx model.
    """
    model.cuda().half()
    cuda_inputs = []
    for inp in inputs:
        try:
            if inp.dtype == torch.float32:
                cuda_inputs.append(inp.cuda().half())
            else:
                cuda_inputs.append(inp.cuda())
        except:
            cuda_inputs.append(inp)


    model_dir = './onnx/'
    torch.onnx.export(
        model,
        tuple(cuda_inputs),
        os.path.join(model_dir, "b%d_model.onnx" % batch_size),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["bert_input", "time_input"],
        output_names=["logkey_output", "cls_output"],
    )

    print('model exported to onnx model!')
    print(os.path.join(model_dir, "b%d_model.onnx" % batch_size))

