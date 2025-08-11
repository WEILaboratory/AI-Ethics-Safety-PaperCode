import glob
import os
import random
import shutil

from module import *

config = {
    "{}".format(SIMG): {
        "path": "/data/train",
        "count": 1,
    },
    "{}/train".format(DIMG): {
        "path": "/data/train",
        "count": 1,
    },
    "{}/val".format(DIMG): {
        "path": "/data/val",
        "count": 1,
    },
}

if __name__ == '__main__':
    for i in [SIMG, FLOG, FIMG, FSIMG, DIMG]:
        shutil.rmtree(i, ignore_errors=True)

    for name, conf in config.items():
        os.makedirs(name, exist_ok=True)

        for path in random.sample(glob.glob(conf["path"] + "/*"), conf["count"]):
            os.symlink(path, "./{}/{}".format(name, os.path.basename(path)))


