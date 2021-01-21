# author : 'wangzhong';
# date: 21/01/2021 14:46

"""
辅助模块，设置一些简单的辅助函数
"""

import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):

    # 无止尽地生成多个data loader，每次取其中一个loader，然后遍历这个loader
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        # 全部置为0
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        # 结果为 {'loss': 0, 'acc': 0, 'top5acc': 0}
        return dict(self._data.average)

if __name__ == '__main__':
    mt = MetricTracker(*["loss", "acc", "top5acc"])
    df = mt._data
    print(dict(df.average))