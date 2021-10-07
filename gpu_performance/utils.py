import time
import torch
from tqdm import tqdm

def measure_gpu_utilization(model: torch.nn.Module,
                            inputs,
                            optim: torch.optim.SGD,
                            model_flops: int,
                            batch_size: int,
                            n_iterations: int = 1000,
                            peak_device_flops: int = 120 * 10**12) -> None:
    """
    Measures the overall utilization of a GPU by comparing the effecitive flops on a given model to it's peak flops.
    """
    model.cuda().half()
    cuda_inputs = []
    for inp in inputs:
        if inp.dtype == torch.float32:
            cuda_inputs.append(inp.cuda().half())
        else:
            cuda_inputs.append(inp.cuda())

    if optim is None:
        print("Measuring inference performance.")
        model.eval()
        start = time.time()
        pbar = tqdm(range(n_iterations), total=n_iterations, dynamic_ncols=True)
        for _ in pbar:
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
            loss = model(*cuda_inputs)
            loss.backward(torch.ones_like(loss))
            optim.step()
        torch.cuda.synchronize()
        end = time.time()
    print('throughput: {:.4f} samples/s, measured over {} iterations.'.format(
        n_iterations * batch_size / (end - start), n_iterations))
