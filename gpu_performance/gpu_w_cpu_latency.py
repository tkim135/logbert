import time
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Tuple
from bert_pytorch.model import BERTLog

def measure_gpu_utilization(model: torch.nn.Module,
                            inputs: Tuple[torch.Tensor, ...],
                            optim: torch.optim.SGD,
                            model_flops: int,
                            batch_size: int,
                            n_iterations: int,
                            peak_device_flops: int,
                            labels: Optional[torch.Tensor] = None,
                            is_logkey: Optional[bool] = True,
                            no_head: Optional[bool] = False):
    """
    Measures the overall utilization of a GPU by comparing the effecitive flops on a given model to it's peak flops.
    """
    
    model.cuda().half()
    if no_head:
        model = model.bert

    # define loss function after model has been moved to GPU
    criterion = nn.NLLLoss(ignore_index=0)
    
    cpu_inputs = []
    for inp in inputs:
        cpu_inputs.append(inp.cpu())
            
    if optim is None:
        print("Measuring inference performance.")
        print('Timing including the cpu->gpu data transfering time')
        model.eval()
        start = time.time()
        pbar = tqdm(range(n_iterations), total=n_iterations, dynamic_ncols=True)
        
        for _ in pbar:
            with torch.no_grad():
                cuda_inputs = []
                for inp in inputs:
                    if inp.dtype == torch.float32:
                        cuda_inputs.append(inp.cuda().half())
                    else:
                        cuda_inputs.append(inp.cuda())
                model(*cuda_inputs)
            
            
        torch.cuda.synchronize()
        end = time.time()
    else:
        print("Measuring training performance.")
        model.train()
        start = time.time()
        pbar = tqdm(range(n_iterations), total=n_iterations, dynamic_ncols=True)
        for _ in pbar:
            model.zero_grad()
            cuda_inputs = []
            for inp in inputs:
                if inp.dtype == torch.float32:
                    cuda_inputs.append(inp.cuda().half())
                else:
                    cuda_inputs.append(inp.cuda())
            result = model(*cuda_inputs)
            if no_head:
                result.backward(torch.ones_like(result))
            else:
                # move labels to GPU also
                if labels.dtype == torch.float32:
                    labels = labels.cuda().half()
                else:
                    labels = labels.cuda()
                loss = torch.tensor(0) if not is_logkey else criterion(result["logkey_output"].transpose(1, 2), labels)
                loss.backward()
            optim.step()
        torch.cuda.synchronize()
        end = time.time()
    print('throughput: {:.4f} samples/s, measured over {} iterations.'.format(
        n_iterations * batch_size / (end - start), n_iterations))
    return n_iterations * batch_size / (end - start)
