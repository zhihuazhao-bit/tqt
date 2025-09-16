import sys

def check_mmcv_full():
    print("Python:", sys.version)
    try:
        import torch
        print("PyTorch:", torch.__version__)
        print("torch.version.cuda:", torch.version.cuda)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU count:", torch.cuda.device_count())
            print("Current device:", torch.cuda.current_device())
            print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    except Exception as e:
        print("PyTorch import/CUDA check failed:", repr(e))
        return

    try:
        import mmcv
        print("MMCV:", mmcv.__version__)
    except Exception as e:
        print("Import mmcv failed:", repr(e))
        return

    try:
        from mmcv import ops
        print("Imported mmcv.ops successfully:", True)
        # 关键算子存在性检查（不同版本可能略有差异）
        keys = [
            "DeformConv2d", "ModulatedDeformConv2d", "RoIAlign", "nms", "soft_nms"
        ]
        for k in keys:
            ok = hasattr(ops, k)
            print(f"ops has {k}: {ok}")
        # 运行一次 GPU 上的 NMS 以检验 CUDA 扩展
        try:
            import torch
            import numpy as np
            if torch.cuda.is_available():
                boxes = torch.tensor([
                    [10, 10, 20, 20],
                    [12, 12, 22, 22],
                    [50, 50, 70, 70]
                ], dtype=torch.float32, device="cuda")
                scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32, device="cuda")
                keep = ops.nms(boxes, scores, iou_threshold=0.5)
                print("NMS on CUDA ok, keep indices:", keep)
            else:
                print("CUDA not available, skipping CUDA NMS test.")
        except Exception as e:
            print("Running CUDA NMS failed:", repr(e))
    except Exception as e:
        print("Import mmcv.ops failed:", repr(e))

if __name__ == "__main__":
    check_mmcv_full()
