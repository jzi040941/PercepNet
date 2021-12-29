import glob
import random
import sys
import os

def main(path):
    dev_ratio = 0.2
    filelist = glob.glob(os.path.join(path, "*.out"))
    random.shuffle(filelist)
    border_idx = int(len(filelist)*(1-dev_ratio))
    train_set_list = filelist[:border_idx]
    dev_set_list = filelist[border_idx:]
    with open(os.path.join(path, "train.txt"), "w") as outfile:
        outfile.write("\n".join(train_set_list))
    with open(os.path.join(path, "dev.txt"), "w") as outfile:
        outfile.write("\n".join(dev_set_list))

if __name__ == '__main__':
    main(sys.argv[1])