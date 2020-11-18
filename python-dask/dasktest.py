# author : 'wangzhong';
# date: 18/11/2020 12:44

"""
dask通过调用简单的方法就可以进行分布式计算、并支持部分模型的并行化处理
步骤：
1.使用conda或者pip安装dask，建议直接安装完整版： pip3 install 'dask[complete]'
2.在这台主机上命令行输入dask-scheduler开启dask注册中心，可以看到该机器的ip和端口，如192.168.0.103:8786
3.dask的分布式需要不同的机器作为worker去执行任务，现在有了注册中心，剩下就需要worker
4.命令行输入 dask-worker 192.168.0.103:8786(注册中心的ip和端口，该处自行更改)
5.注册好之后，运行代码即可进行分布式计算，可视化统计面板地址：http://127.0.0.1:8787/status
"""

from dask.distributed import Client
from time import time


def square(x):
    return x ** 2


if __name__ == '__main__':
    MAX = 1000
    st = time()
    client = Client('192.168.0.103:8786')   # 这里的地址记得根据我上面说的修改掉。
    A = client.map(square, range(MAX))
    total = client.submit(sum, A)
    print(total.result())
    et = time()
    print(et - st)