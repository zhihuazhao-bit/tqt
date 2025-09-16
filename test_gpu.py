import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name}')
        print(f'Compute Capability: {props.major}.{props.minor}')
        print(f'SM: sm{props.major}{props.minor}')
        print('PyTorch CUDA architectures:', torch.cuda.get_arch_list())