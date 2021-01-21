# author : 'wangzhong';
# date: 21/01/2021 18:19

import argparse
import collections
import torch
import numpy as np
import loader.data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer.trainer import Trainer


# 保证随机状态一致
SEED = 42
torch.manual_seed(SEED)
# 下面这两个都是针对GPU的，提升速度和避免随机性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config: ConfigParser):
    # 获取一个logging.getLogger，默认日志级别为debug
    logger = config.get_logger('train')
    # 数据模块
    # 获取config中读取到的config.json里的loader的名字，并实例化，用json里的参数去填充
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # 模型模块
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # 损失与评估模块
    criterion = getattr(module_loss, config['loss'])
    # 这里面存的是function，也可能存的是类，通过__name__方法获得名字
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # 优化器模块
    # filter，过滤掉false值
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    # 学习率衰减策略
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # 训练模型
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


# 一样可以创建model，
# def create_model_test(parse):
#     args = parse.parse_args()
#     model_name = args.model_name
#     # 这里的相对目录是针对当前文件的相对目录，不需要改
#     model = getattr(module_data, model_name)(data_dir="data/", batch_size=218)
#     return model


if __name__ == '__main__':
    # 从cpu上训练的模型，拿到GPU上，会出问题，记得load的时候，加上map_location参数
    args = argparse.ArgumentParser(description='PyTorch Template')
    # 读取config文件，读取各个参数
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    # args.add_argument('-m', '--model_name', default=None, type=str)
    # 可以更改json文件中的参数直接用命令的方式
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    # 返回一个ConfigParse类，它会把这次跑的config文件放到这次跑的目录下
    # 它的属性主要有这次读取的config文件，resume，save_dir, log_dir
    # 其实，如果我不特殊指定lr和bs，上面这步根本不需要
    config = ConfigParser.from_args(args, options)
    main(config)
    # model = create_model_test(args)
