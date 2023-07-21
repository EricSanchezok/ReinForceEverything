import torch

def test_cuda_performance():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA Device Name:", torch.cuda.get_device_name(0))

        # 获取CUDA设备的计算能力
        major = torch.cuda.get_device_capability(0)[0]
        cores_per_sm = {
            3.0: 192,
            3.5: 192,
            3.7: 192,
            5.0: 128,
            5.2: 128,
            6.0: 64,
            6.1: 128,
            6.2: 128,
            7.0: 64,
            7.2: 64,
            7.5: 64
        }
        cores = cores_per_sm.get(major, 0)
        clock_rate = torch.cuda.get_device_properties(0).clockRate * 1e-6  # GHz

        tflops = 2.0 * cores * clock_rate
        print(f"Theoretical TFLOPS: {tflops:.2f} TFLOPS")
    else:
        print("No CUDA device available.")

if __name__ == "__main__":
    test_cuda_performance()
