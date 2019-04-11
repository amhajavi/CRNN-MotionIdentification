import os
import numpy as np
from progressbar import progressbar
dims =[6, 3, 3, 3, 3, 3, 3, 2, 3, 1, 1, 2, 1, 2, 2, 3, 1, 1, 2, 1, 2, 3, 1, 2, 1, 3, 1, 2, 1]

def zeropad(record , length = 250):
    if len(record) < length:
        print(len(record), length)
        try:
            pad = np.zeros((length-len(record),))
            record = np.concatenate([record, pad])
        except Exception as e:
            raise
    return record

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def split(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

def split_strip(x):
    return [_.strip() for _ in x.split()]

def read_user_list():
    user_list = {}
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'subjects')
    files = []
    with open('motion_meta_matrix.txt','w+') as motion_meta:
        for subject in os.listdir(dir_path):
            user_path = os.path.join(dir_path,subject)
            amc_files = [os.path.join(user_path, file)
                                    for file in os.listdir(user_path) if file.endswith('.npy')]
            if len(amc_files)>2:
                for item in amc_files:
                    files += ['{} {} \n'.format(item,subject)]
        np.random.shuffle(files)
        for file in files:
            motion_meta.write(file)

def convert_amc_matrix_identities(file_name, index=0):
    data = []
    with open(file_name, 'r') as mocap_file:
        lines = mocap_file.readlines()[3:]
        for _, frame in enumerate(chunks(lines,30)):
            frame_point = []
            for idx, joint in enumerate(frame[1:]):
                frame_point += list(zeropad(list(map(float, joint.strip().split()[1:])), length=dims[idx]))
            if len(frame_point) != 62:
                print(frame[0].strip().split())
                print("hereeee isss the frameeeeeeeeeeee",_)
                print(file_name)
                print(frame_point)
                raise
            data.append(frame_point)
    for chunk in chunks(data, 250):
        try:
            index += 1
            np.savetxt(file_name.replace('.amc','_{}.npy').format(index), chunk)
        except TypeError as te:
            print(chunk)
            raise te
    return chunks(data, 250)


def create_matrix_files():
    with open('data/motion_meta.txt','r+') as motion_meta:
        motion_data_set = motion_meta.readlines()
    motion_data_set = list(map(split_strip, motion_data_set[:10000]))
    for file, id in progressbar(motion_data_set):
        convert_amc_matrix_identities(file)

read_user_list()
