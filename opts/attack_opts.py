import argparse
def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', default='videos', type=str, help='directory path of data')
    parser.add_argument('--result_path', default='result_video', type=str, help='Result directory path')
    parser.add_argument('--test_path', default='', type=str, help='test path | eval.py')
    # datasets
    parser.add_argument('--dataset', default='ucf101', type=str, help='Used dataset (kinetics | ucf101 | hmdb51)')
    parser.add_argument('--model', default='c3d', type=str, help='Used action recognition model (c3d | lrcn | i3d)')
    parser.add_argument('--n_classes', default=101, type=int,
                        help='Number of classes (kinetics: 400, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--pretrain_path', default='pre_model/resnext-101-kinetics-ucf101_split1.pth', type=str,
                        help='Pretrained model (.pth)')
    parser.add_argument('--model_type', default='resnext', type=str,
                        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--model_depth', default=101, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
    # dataloader
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration of inputs')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')
    parser.add_argument('--scale_in_test', default=1.0, type=float, help='Spatial scale in test')
    parser.add_argument('--no_softmax_in_test', action='store_true',
                        help='If true, output for each clip is not normalized using softmax.')
    parser.set_defaults(no_softmax_in_test=False)
    parser.add_argument('--norm_value', default=1, type=int,
                        help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')\

    # device
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--seed', default=0, type=int, help='Manually set random seed')
    # attack setting
    parser.add_argument('--pop_size', default=15, type=int, help='')
    parser.add_argument('--steps', default=3000, type=int, help='')
    parser.add_argument('--init_rate', default=0.4, type=float, help='')
    parser.add_argument('--mutation_rate', default=1, type=int, help='')
    parser.add_argument('--targeted', action='store_true')
    parser.add_argument('--cf_rate',default=0.7,type=float)
    parser.add_argument('--time_mua',default=2,type=int)
    parser.add_argument('--w',default=1.0,type=float)
    args = parser.parse_args()
    return args

