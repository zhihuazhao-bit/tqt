from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SetIterHook(Hook):
    """同步 runner.iter 到模型，以便模型内部使用当前迭代数。

    在 before_train_iter 设置 `model.iter = runner.iter`。若模型被 DataParallel/
    DistributedDataParallel 包装，会下探到实际模块。
    """

    def before_train_iter(self, runner):
        model = runner.model
        # 兼容 DP/DDP 包装
        if hasattr(model, 'module'):
            model = model.module
        setattr(model, 'iter', runner.iter)
