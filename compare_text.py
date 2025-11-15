import os
import mmcv
import torch
import argparse

from mmcv.utils import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

if not hasattr(MMDistributedDataParallel, '_use_replicated_tensor_module'):
    MMDistributedDataParallel._use_replicated_tensor_module = False

from mmseg.models import build_segmentor
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset

import models

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument(
        '--config', 
        default='configs/tqt/tqt_eva_vit-b_1e-5_10k-r2r-512-light-traversable-pixel-proj-cls-prefix.py',
        help='test config file path')
    parser.add_argument(
        '--checkpoint', 
        default='work_dirs/tqt_eva_vit-b_1e-5_10k-r2r-512-light-traversable-pixel-proj-cls-prefix/best_mIoU_iter_2000.pth',
        help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        default=False,
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument(
        '--show', 
        default=True,
        action='store_true', 
        help='show results')
    parser.add_argument(
        '--show-dir', 
        default='./work_dirs/test/tqt-eva-b-sufficient-traversable-cls',
        help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./work_dirs/test/')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--aug_ratio_start', type=float, default=-1)
    parser.add_argument('--exp_tag', default=None)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    config_name = os.path.basename(args.config).split('.')[0]
    if args.show_dir:
        args.show_dir = os.path.join(args.show_dir, config_name)
    if args.save_dir:
        args.save_dir = os.path.join(args.save_dir, config_name)

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # 导入 SimpleTokenizer
    from models.backbones.utils import SimpleTokenizer

    # 初始化 SimpleTokenizer
    tokenizer = SimpleTokenizer()
    import numpy as np
    tokens = np.zeros((len(cfg.class_names)), dtype=np.int64)
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    # 示例：对 class_names 中的每个类别名称进行分词
    for i, class_name in enumerate(cfg.class_names):
        token = [sot_token] + tokenizer.encode(class_name) + [eot_token]
        tokens[i] = len(token) + 12
    cfg.model.context_length = int(tokens.max())
    cfg.model.eva_clip.context_length = int(tokens.max())+8
    
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.aug_test:
        if args.aug_ratio_start > 0.:
            cfg.data.test.pipeline[1].img_ratios = [
                args.aug_ratio_start, 1.0, 1.25, 1.5, 1.75
            ]
        # hard code index
        else:
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75
            ]
        cfg.data.test.pipeline[1].flip = True

    if args.checkpoint != 'None':
        cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.int)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    cfg.model.class_names = list(dataset.CLASSES)

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    if args.checkpoint == 'None':
        model.CLASSES = dataset.CLASSES
        model.PALETTE = dataset.PALETTE
    elif "CLIP-ViT" in args.checkpoint:
        model.backbone.init_weights(args.checkpoint)
        model.text_encoder.init_weights(args.checkpoint)
        model.CLASSES = dataset.CLASSES
        model.PALETTE = dataset.PALETTE        
    else:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            print('"CLASSES" not found in meta, use dataset.CLASSES instead')
            model.CLASSES = dataset.CLASSES
        if 'PALETTE' in checkpoint.get('meta', {}):
            model.PALETTE = checkpoint['meta']['PALETTE']
        else:
            print('"PALETTE" not found in meta, use dataset.PALETTE instead')
            model.PALETTE = dataset.PALETTE

    model = model.cuda()
    from models.backbones.utils import tokenize
    texts = []
    labels = []
    annotations = []
    for w in ['sunny', 'rainy', 'snowy', 'foggy']:
        for l in ['daytime', 'dusk', 'nighttime']:
            for r in ['unpaved', 'paved']:
                scene = f"A {w} scene during {l} on a {r}, "
                texts.append(tokenize(scene + 'traversable area', context_length=cfg.model.context_length))
                labels.append(0)
                annotations.append(f"{w}, {r}, {l}, traversable")
                texts.append(tokenize(scene + 'non-traversable area', context_length=cfg.model.context_length))
                labels.append(1)
                annotations.append(f"{w}, {r}, {l}, not traversable")
    text_tensor = torch.cat(texts).cuda()
    with torch.no_grad():
        text_features = model.text_encoder(text_tensor, context=model.contexts)
    
    texts2 = []
    for w in ['sunny', 'rainy', 'snowy', 'foggy']:
        for l in ['daytime', 'dusk', 'nighttime']:
            for r in ['unpaved', 'paved']:
                scene = f"A {w} scene during {l} on a {r}, "
                texts2.append(tokenize("A region that is " + scene + 'traversable area', context_length=cfg.model.eva_clip.context_length))
                labels.append(0)
                annotations.append(f"Fix, {w}, {r}, {l}, traversable")
                texts2.append(tokenize("A region that is " + scene + 'non-traversable area', context_length=cfg.model.eva_clip.context_length))
                labels.append(1)
                annotations.append(f"Fix, {w}, {r}, {l}, not traversable")
    text_tensor2 = torch.cat(texts2).cuda()
    with torch.no_grad():
        text_features2 = model.text_encoder(text_tensor2, context=None)

    text_features = torch.cat([text_features, text_features2], dim=0)

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np

    # Move features to CPU and convert to numpy
    text_features_np = text_features.cpu().numpy()

    # PCA to 2 dimensions
    pca = PCA(n_components=2)
    text_features_pca = pca.fit_transform(text_features_np)

    # t-SNE to 2 dimensions
    # Set perplexity to be less than the number of samples
    perplexity = min(30, len(texts) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    text_features_tsne = tsne.fit_transform(text_features_np)

    # Visualization
    plt.figure(figsize=(16, 7))
    
    class_names = ['traversable', 'non-traversable']
    # PCA plot
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(text_features_pca[:, 0], text_features_pca[:, 1], c=labels, cmap='viridis')
    plt.title('PCA of Text Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(handles=scatter.legend_elements()[0], labels=class_names)
    for i, txt in enumerate(annotations):
        plt.annotate(i, (text_features_pca[i, 0], text_features_pca[i, 1]), fontsize=6)

    # t-SNE plot
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(text_features_tsne[:, 0], text_features_tsne[:, 1], c=labels, cmap='viridis')
    plt.title('t-SNE of Text Features')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.legend(handles=scatter.legend_elements()[0], labels=class_names)
    for i, txt in enumerate(annotations):
        plt.annotate(i, (text_features_tsne[i, 0], text_features_tsne[i, 1]), fontsize=6)

    plt.tight_layout()
    plt.savefig('text_features_visualization_traversable.png')

    print("Index to Annotation Mapping:")
    for i, annotation in enumerate(annotations):
        print(f"{i}: {annotation}")

    plt.show()


if __name__ == '__main__':
    main()
