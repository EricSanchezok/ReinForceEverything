import torch
import time

device = torch.device('cuda') 

a = torch.randn(10000, 10000, device=device)
b = torch.randn(10000, 10000, device=device)

start_time = time.perf_counter()
c = torch.mm(a, b)
end_time = time.perf_counter()

duration = end_time - start_time
print(f'Duration: {duration} seconds')

flops = 2 * a.shape[0] * a.shape[1] * b.shape[1]
tflops = flops / duration / 1e12
print(f'TFLOPS: {tflops}')