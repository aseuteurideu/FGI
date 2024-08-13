## code source - https://github.com/TengdaHan/DPC/tree/master/backbone
from resnet_2d3d import * 

def select_resnet_half(network, track_running_stats=True):
    param = {'feature_size': 512}
    if network == 'resnet18':
        model = resnet18_2d3d_half(track_running_stats=track_running_stats)
        param['feature_size'] = 128
    elif network == 'resnet34':
        model = resnet34_2d3d_half(track_running_stats=track_running_stats)
        param['feature_size'] = 128
    elif network == 'resnet50':
        model = resnet50_2d3d_half(track_running_stats=track_running_stats)
    elif network == 'resnet101':
        model = resnet101_2d3d_half(track_running_stats=track_running_stats)
    elif network == 'resnet152':
        model = resnet152_2d3d_half(track_running_stats=track_running_stats)
    elif network == 'resnet200':
        model = resnet200_2d3d_half(track_running_stats=track_running_stats)
    else: raise IOError('model type is wrong')

    return model, param
