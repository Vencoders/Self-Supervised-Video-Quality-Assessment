import os
import re
import skvideo.io
from collections import OrderedDict
from scipy import io as sio
import numpy as np
import json

if __name__ == "__main__":

    with open('./live_subj_score_nr_ref.json') as f:
        data = json.load(f)
        typeLen = [1, 4, 3, 4, 4]
        idx = 1
        f = dict()
        for i, curLen in enumerate(typeLen):
            for x in range(curLen):
                f[idx] = i
                idx += 1

        for mode in ['train', 'test']:
            typeInfo = []
            for name in data[mode]['dis']:
                cur = name.split('_')[0]
                cur = cur[2:]
                ans = f[int(cur)]
                typeInfo.append(ans)
            data[mode]['type'] = typeInfo

        ret = data

        with open('./LIVE_subj_score_5.json', 'w') as f:
            json.dump(ret, f, indent=4)