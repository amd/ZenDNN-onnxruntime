import onnxruntime

cfg = onnxruntime.OrtArenaCfg({'max_mem':10000, 'arena_extend_strategy':0, 'initial_chunk_size_bytes':100, 'max_dead_bytes_per_chunk':1000, 'initial_growth_chunk_size_bytes':20})
onnxruntime.create_and_register_cuda_allocator(0, 100000, cfg) # (device_id, gpu_mem_limit, OrtArenaCfg)
