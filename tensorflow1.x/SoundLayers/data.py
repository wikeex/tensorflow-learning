import os
import numpy as np
import librosa
import math
from random import shuffle
import shutil


def rename():
    folders = os.listdir('./data/sound')
    for folder in folders:
        files = os.listdir('./data/sound/{0}'.format(folder))
        for file in files:
            name_list = file.split('.')
            suffix = '.' + name_list[1] if name_list else None
            os.rename(
                './data/sound/'+folder+'/'+file,
                './data/sound/'+folder+'_'+str(files.index(file))+suffix
            )


def split_dataset(basedir):
    """
    按所给目录下文件的数量将文件划分为训练集、评估集和测试集三个集合。
    :return:
    """
    folders = os.listdir(basedir)
    name_dict = {}
    for file_name in folders:

        # 去除文件夹
        if os.path.isdir(os.path.join(basedir, file_name)):
            print(file_name)
            continue

        name = file_name.split('_')[0]
        if name_dict.get(name) is None:
            name_dict[name] = [file_name]
        else:
            name_dict[name].append(file_name)

    for k, v in name_dict.items():
        if v is None:
            continue

        train_count = len(v) * 7 // 10
        eval_count = len(v) * 1 // 10

        shuffle(v)

        for i in v[:train_count]:
            shutil.move(os.path.join(basedir, i), os.path.join(basedir, 'train', i))
        for i in v[train_count:train_count+eval_count]:
            shutil.move(os.path.join(basedir, i), os.path.join(basedir, 'eval', i))
        for i in v[train_count+eval_count:]:
            shutil.move(os.path.join(basedir, i), os.path.join(basedir, 'test', i))


def one_hot_from_files():
    """
    将分类名称转换成ont hot编码。
    :return: 键为分类名称，值为one hot编码列表的字典。
    """
    files = os.listdir('G:/sound')
    char_list = []
    one_hot = dict()

    for file in files:
        char = file.split('_')[0]
        char_list.append(char) if char not in char_list else ...
    for char in char_list:
        x = [0] * len(char_list)
        x[char_list.index(char)] = 1
        one_hot[char] = x
    return one_hot


def extract_classification(output, one_hot):
    """
    从模型推理结果中提取分类及概率。
    :param output:
    :param one_hot:
    :return:
    """
    classification = [''] * 59
    for k, v in one_hot.items():
        classification[v.index(1)] = k

    softmax = output.tolist()[0]

    result = list()

    for i in range(10):
        max_value = max(softmax)
        index = softmax.index(max_value)
        result.append((classification[index], max_value))
        softmax.pop(index)

    return tuple(result)


def mel_batch_generator(slice_size, dest_path, source_path='G:/sound/', fixed=True):
    files = os.listdir(source_path)

    for file in files:
        print("loaded batch {0}".format(file))
        if not file.endswith('.wav'):
            continue
        wave, sr = librosa.load(source_path+file, sr=22050)
        mel = librosa.feature.melspectrogram(wave, n_fft=2205, hop_length=1102, n_mels=512)

        if fixed:
            division = math.ceil(mel.shape[1] / slice_size)
            mel_split = np.array_split(mel, division, axis=1)
            for i, np_data in enumerate(mel_split):
                index = str(i)

                np_data_fixed = np.pad(
                    np_data,
                    ((0, 0), (0, slice_size - len(np_data[0]))),
                    mode='constant',
                    constant_values=0
                )

                preffix = file.split('.')[0]
                name = dest_path + preffix + '_' + index + '.npy'
                np.save(name, np_data_fixed)
        else:
            name = dest_path + file.split('.')[0] + '.npy'
            np.save(name, mel)


def np_load(batch_type, path='G:/sound_fixed/'):
    """
    加载npy文件
    :param batch_type:
    :param path:
    :return:
    """
    files = os.listdir(os.path.join(path, batch_type))
    labels_dict = one_hot_from_files()

    while True:
        shuffle(files)
        for file in files:
            label = labels_dict.get(file.split('_')[0])
            sound_data = np.load(os.path.join(path, batch_type, file))
            slice_number = math.ceil(sound_data.shape[1] / 10)
            data_list = np.array_split(sound_data, slice_number, axis=1)
            for index, data_slice in enumerate(data_list):
                np_data_padding = np.pad(
                    data_slice,
                    ((0, 0), (0, 10 - data_slice.shape[1])),
                    mode='constant',
                    constant_values=0
                )
                if index + 1 == len(data_list):
                    end = True
                else:
                    end = False
                yield np_data_padding, np.array(label, dtype=np.float32), end


if __name__ == '__main__':
    split_dataset(basedir='F:/sound_half_second/')
