from cartridges.training.profiling_train import count_cuda_api_calls


def test_count_cuda_api_calls():
    names = [
        "cudaLaunchKernel", "cudaLaunchKernel", "cuLaunchKernel",
        "cudaStreamSynchronize", "cudaDeviceSynchronize",
        "cudaMemcpyAsync",
        "aten::mm", "forward",
    ]
    assert count_cuda_api_calls(names) == {"launches": 3, "syncs": 2, "memcpys": 1}


def test_count_cuda_api_calls_empty():
    assert count_cuda_api_calls([]) == {"launches": 0, "syncs": 0, "memcpys": 0}
