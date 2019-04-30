import numpy as np
import itertools

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def read_amc(file_name):
    with open(file_name, mode='r') as amc_file:
        lines = amc_file.readlines()[3:]
        motion = []
        for chunk in chunks(lines, 30):
            pos = []
            for line in chunk:
                clean_line = list(map(float, line.strip().split()[1:]))
                pos += clean_line
            motion += [pos]
    return motion

### sample for
# read_amc('/home/amh/Workbench/Motion Capture/dataset/subjects/01/01_01.amc')
