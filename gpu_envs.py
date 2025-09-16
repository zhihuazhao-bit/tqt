import xformers
import torch
print(f'xFormers: {xformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    try:
        import xformers.ops
        device = torch.device('cuda:0')
        q = torch.randn(2, 8, 64, device=device, dtype=torch.float16)
        k = torch.randn(2, 8, 64, device=device, dtype=torch.float16)
        v = torch.randn(2, 8, 64, device=device, dtype=torch.float16)
        out = xformers.ops.memory_efficient_attention(q, k, v)
        print('✅ GPU版本xFormers安装成功！')
    except Exception as e:
        print(f'❌ 仍然是CPU版本: {e}')