import copy

import torch
from torch.utils import data
import os
import sys
import pandas as pd
from augmentation import *
from scipy.io import wavfile
from utils import min_max_normalize

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def my_collate_rawaudio(batch):
    batch = list(filter(lambda x: x is not None and x[1].size()[0] == 48000, batch))
    if len(batch) == 0:
        return [[],[],[],[],[],[],[]]
    return torch.utils.data.dataloader.default_collate(batch)


def my_collate(batch):
    batch = list(filter(lambda x: x is not None and x[1].size()[3] == 99, batch))
    if len(batch) == 0:
        return [[],[],[],[],[],[]]
    return torch.utils.data.dataloader.default_collate(batch)

class deepfake_3d_rawaudio(data.Dataset):
    def __init__(self, out_dir,
                 mode='train',
                 transform=None,
                 vis_min_fake_len=2, vis_max_fake_len=-1,
                 aud_min_fake_len=2, aud_max_fake_len=-1,
                 using_pseudo_fake=False,
                 dataset_name='dfdc'):
        assert dataset_name in ['dfdc','fakeavceleb']
        self.mode = mode
        self.transform = transform
        self.out_dir = out_dir
        self.dataset_name = dataset_name

        # splits
        if mode == 'train':
            assert dataset_name == 'dfdc'
            split = os.path.join(self.out_dir, 'train_split.csv')
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'test'):
            assert not using_pseudo_fake
            if dataset_name == 'dfdc':
                split = os.path.join(self.out_dir, 'test_imbalance_split.csv')
            else:  # elif dataset_name == 'fakeavceleb':
                split = os.path.join(self.out_dir, 'test_balance_fakeavceleb_split.csv')

            video_info = pd.read_csv(split, header=None)
        else:
            raise ValueError('wrong mode')

        video_info_fake = None
        video_info_real = None
        self.using_pseudo_fake = using_pseudo_fake if mode == 'train' else False

        # get label list
        self.label_dict_encode = {}
        self.label_dict_decode = {}
        self.label_dict_encode['fake'] = 0
        self.label_dict_decode['0'] = 'fake'
        self.label_dict_encode['real'] = 1
        self.label_dict_decode['1'] = 'real'

        self.video_info = video_info
        self.video_info_real = video_info_real
        self.video_info_fake = video_info_fake

        self.vis_min_fake_len = vis_min_fake_len
        self.vis_max_fake_len = vis_max_fake_len
        self.aud_min_fake_len = aud_min_fake_len
        self.aud_max_fake_len = aud_max_fake_len



    def _generate_pseudo_fake(self, t_seq, audio, other_t_seq, other_audio):
        # t_seq = time x channel x h x w
        # audio = 48000

        # decide to make realVid-fakeAud (0), fakeVid-realAud (1), or fakeVid-fakeAud (2)
        chosen_type = random.choice([0, 1, 2])
        if chosen_type == 0:
            audio = self._augment_pseudo_fake(audio, audio.shape[0],
                                              minimum_fake_length=self.aud_min_fake_len, maximum_fake_length=self.aud_max_fake_len,
                                              other_data=other_audio)
        elif chosen_type == 1:
            t_seq = self._augment_pseudo_fake(t_seq, t_seq.shape[0],
                                              minimum_fake_length=self.vis_min_fake_len, maximum_fake_length=self.vis_max_fake_len,
                                              other_data=other_t_seq)
        elif chosen_type == 2:
            audio = self._augment_pseudo_fake(audio, audio.shape[0],
                                              minimum_fake_length=self.aud_min_fake_len, maximum_fake_length=self.aud_max_fake_len,
                                              other_data=other_audio)
            t_seq = self._augment_pseudo_fake(t_seq, t_seq.shape[0],
                                              minimum_fake_length=self.vis_min_fake_len, maximum_fake_length=self.vis_max_fake_len,
                                              other_data=other_t_seq)
        else:
            sys.exit(1)

        return t_seq, audio, chosen_type

    def _select_pseudo_fake_window(self, data_length, minimum=2, maximum=-1):
        if 0 < minimum <= 1:
            minimum = max(2, int(minimum * data_length))
        if minimum == -1:
            minimum = data_length
        if maximum == -1:
            maximum = data_length
        elif 0 < maximum <= 1:
            maximum = min(minimum, int(maximum * data_length))
        assert minimum >= 2
        assert maximum >= minimum

        # randomly select pseudo fake time length from minimum to maximum
        fake_len = random.randint(minimum, maximum)

        # select starting position
        start_pos = random.randint(0, data_length-fake_len)

        # end position
        end_pos = min(data_length, start_pos + fake_len)  # excluded position

        return fake_len, start_pos, end_pos

    def _repeat_and_skip_pseudo_fake(self, data, fake_len, start_pos, end_pos, numrepeat_min=2, numrepeat_max=2):
        # For example, real data: I1, I2, I3, I4, I5, I6, I7, I8. Pseudo fake from I3 to I6.
        # repeat and skip.  --> I1, I2, I3, I3, I5, I5, I7, I8. (type_0_numrepeat = [2])
        if numrepeat_max == -1:
            numrepeat_max = fake_len

        numrepeat = min(fake_len, random.randint(numrepeat_min, numrepeat_max))
        num_segments = math.ceil(fake_len / numrepeat)
        original_data = copy.deepcopy(data)

        for ns in range(num_segments):
            start_i = start_pos + ns * numrepeat
            end_i = min(start_pos + fake_len, start_i + numrepeat)
            j = 0
            for i in range(start_i, end_i):
                data[i] = original_data[start_i]
                # print('replacing data ', i, ' with data ', start_i)
                j += 1
                
        return data

    def _backward_or_flip_pseudo_fake(self, data, fake_len, start_pos, end_pos, everynframes_min=2, everynframes_max=2):
        # For example, real data: I1, I2, I3, I4, I5, I6, I7, I8. Pseudo fake from I3 to I6.
        # backward when everynframes = fake_len. --> I1, I2, I6, I5, I4, I3, I7, I8.
        # backward when everynframes = 2. --> I1, I2, I4, I3, I6, I5, I7, I8.
        if everynframes_max == -1:
            everynframes_max = fake_len
        everynframes = min(fake_len, random.randint(everynframes_min, everynframes_max))
        original_data = copy.deepcopy(data)

        num_flips = math.ceil(fake_len / everynframes)

        for nf in range(num_flips):
            start_i = start_pos + nf * everynframes
            end_i = min(start_pos + fake_len, start_i + everynframes)

            j = 0
            for i in range(start_i, end_i):
                data[i] = original_data[end_i-j-1]
                # print('replacing data ', i, ' with data ', end_i-j-1)
                j += 1

        return data

    def _random_pseudo_fake(self, data, fake_len, start_pos, end_pos):
        # For example, real data: I1, I2, I3, I4, I5, I6, I7, I8. Pseudo fake from I3 to I6.
        # random order --> I1, I2, I4, I3, I5, I6, I7, I8. Make sure I4, I3, I5, I6 != I3, I4, I5, I6
        index_list = [i for i in range(start_pos, end_pos)]
        ori_list = copy.deepcopy(index_list)
        original_data = copy.deepcopy(data)

        while ori_list == index_list:
            random.shuffle(index_list)

        data[start_pos:end_pos] = original_data[index_list]
        return data

    def _replace_with_other(self, data, fake_len, start_pos, end_pos, other_data):
        # For example, real data: I1, I2, I3, I4, I5, I6, I7, I8. Pseudo fake from I3 to I6.
        # random order --> I1, I2, J3, J4, J5, J6, I7, I8. Where J is from other clip

        data[start_pos:end_pos] = other_data[start_pos:end_pos]
        return data

    def _translate_pseudo_fake(self, data, fake_len, start_pos, end_pos, direction='left', padding='repeat', translate_min=1, translate_max=1):
        # For example, real data: I1, I2, I3, I4, I5, I6, I7, I8. Pseudo fake from I3 to I6. With translate = 1 and left
        assert direction in ['left', 'right']
        assert padding in ['repeat', 'mirror'] # repeat:  I1, I2, I4, I5, I6, I6, I7, I8; mirror:  I1, I2, I4, I5, I6, I5, I7, I8;
        original_data = copy.deepcopy(data)
        if translate_max == -1:
            translate_max = fake_len
        translate = min(fake_len-1, random.randint(translate_min, translate_max))
        # print('start pos: ', start_pos, ' ; end_pos: ', end_pos, ' ; fake len: ', fake_len)

        if direction == 'left':
            for i in range(start_pos, end_pos-translate):
                # print('replacing data ', i, ' with data ', i+translate)
                data[i] = original_data[max(0, min(i+translate, len(original_data)-1))]
            j=0
            for i in range(end_pos-translate, end_pos):  # padding
                j+=1
                if padding == 'repeat':
                    # print('padding repeat: replacing data ', i, ' with data ', end_pos-1)
                    data[i] = original_data[max(0, min(end_pos-1, len(original_data)-1))]
                else: # elif padding == 'mirror':
                    # print('padding mirror: replacing data ', i, ' with data ', end_pos-translate-j+1)
                    data[i] = original_data[max(0, min(end_pos-translate-j+1, len(original_data)-1))]
        else:  # elif direction == 'right'
            for i in range(start_pos + translate, end_pos):
                data[i] = original_data[max(0, min(i-translate, len(original_data)-1))]
                # print('replacing data ', i, ' with data ', i-translate)
            j=0
            for i in range(start_pos, start_pos+translate):  # padding
                if padding == 'repeat':
                    data[i] = original_data[max(0, min(start_pos, len(original_data)-1))]
                    # print('padding repeat: replacing data ', i, ' with data ', start_pos)
                else: # elif padding == 'mirror':
                    data[i] = original_data[max(0, min(start_pos + translate - j, len(original_data)-1))]
                    # print('padding repeat: replacing data ', i, ' with data ', start_pos + translate - j)
                j+=1
        return data
    def _augment_pseudo_fake(self, data, time_len,
                             minimum_fake_length=2, maximum_fake_length=-1,
                             other_data=None):
        # t_seq = 1 x channel x time x h x w
        # type_0_hyperparam = [number repeat]

        # select the pseudo fake time length (minimum 2)
        # select the starting time
        # select the ending time
        fake_len, start_pos, end_pos = self._select_pseudo_fake_window(data_length=time_len,
                                                                       minimum=minimum_fake_length,
                                                                       maximum=maximum_fake_length)

        # For example, real data: I1, I2, I3, I4, I5, I6, I7, I8. Pseudo fake from I3 to I6.
        # replace with other clip
        data = self._replace_with_other(data, fake_len, start_pos, end_pos, other_data)

        return data

    def _get_other_item(self, index):
        if self.using_pseudo_fake and self.mode == 'train':
            success = 0

            while success == 0:
                try:
                    other_index = random.randint(0, self.__len__())
                    while index == other_index:
                        other_index = random.randint(0, self.__len__())

                    other_vpath, other_audiopath, other_label = self.video_info.iloc[other_index]
                    other_vpath = os.path.join(self.out_dir, other_vpath)
                    other_audiopath = os.path.join(self.out_dir, other_audiopath)

                    other_seq = [pil_loader(os.path.join(other_vpath, img)) for img in
                                 sorted(os.listdir(other_vpath))]
                    other_sample_rate, other_audio = wavfile.read(
                        other_audiopath)  # audio size: 48000 (range, so far checking some samples, can be -ten thousands to + ten thousands)

                    other_t_seq = self.transform(other_seq)  # apply same transform

                    (other_C, other_H, other_W) = other_t_seq[0].size()
                    other_t_seq = torch.stack(other_t_seq, 0)

                    # normalize audio. Bcos seems like each audio data has mean roughly 0, just the range is different (maybe some audio is louder than the others), better to normalize to -1 to 1 based on each data range.
                    other_normalized_raw_audio = min_max_normalize(other_audio, int(other_audio.min()),
                                                                   int(other_audio.max()))
                    other_normalized_raw_audio = torch.autograd.Variable(
                        torch.from_numpy(other_normalized_raw_audio.astype(float)).float())
                    other_normalized_raw_audio = (other_normalized_raw_audio - 0.5) / 0.5

                    if len(other_normalized_raw_audio) < 48000 or len(other_t_seq) < 30:
                        print('WARNING: other_index ', other_index, ' has weird length')
                        continue

                    success = 1
                except:
                    other_index = random.randint(0, self.__len__())
                    while index == other_index:
                        other_index = random.randint(0, self.__len__())  # select other data
                    print('WARNING: other_index ', other_index, ' has something wrong with data fetching')
                    continue

        else:
            other_t_seq = None
            other_normalized_raw_audio = None

        return other_t_seq, other_normalized_raw_audio

    def __getitem__(self, index):
        success = 0
        while success == 0:
            try:
                vpath, audiopath, label = self.get_vpath_audiopath_label(index)
                vpath = os.path.join(self.out_dir, vpath)
                audiopath = os.path.join(self.out_dir, audiopath)

                seq = [pil_loader(os.path.join(vpath, img)) for img in
                       sorted(os.listdir(vpath))]

                sample_rate, audio = wavfile.read(
                    audiopath)  # audio size: 48000 (range, so far checking some samples, can be -ten thousands to + ten thousands)

                t_seq = self.transform(seq)  # apply same transform

                (C, H, W) = t_seq[0].size()
                t_seq = torch.stack(t_seq, 0)

                # normalize audio. Bcos seems like each audio data has mean roughly 0, just the range is different (maybe some audio is louder than the others), better to normalize to -1 to 1 based on each data range.
                normalized_raw_audio = min_max_normalize(audio, int(audio.min()), int(audio.max()))
                normalized_raw_audio = torch.autograd.Variable(
                    torch.from_numpy(normalized_raw_audio.astype(float)).float())
                normalized_raw_audio = (normalized_raw_audio - 0.5) / 0.5


                success = 1
            except:
                index = random.randint(0, self.__len__())  # select other data
                print('WARNING: index ', index, ' has something wrong with data fetching')
                continue

        if self.using_pseudo_fake:  # still want pseudo fake augmentation
            use_pseudo_fake = random.randint(0, 1)  # 0: not using augment, 1: use augment
            if use_pseudo_fake:
                other_t_seq, other_normalized_raw_audio = self._get_other_item(index)
                t_seq, normalized_raw_audio, chosen_type = self._generate_pseudo_fake(
                    t_seq, normalized_raw_audio, other_t_seq, other_normalized_raw_audio)
                label = 'fake'


        t_seq = t_seq.view(1, 30, C, H, W).transpose(1, 2)

        vid = self.encode_label(label)  # fake = 0; real = 1

        return t_seq, normalized_raw_audio, torch.LongTensor([vid]), audiopath

    def get_vpath_audiopath_label(self, index):
        vpath, audiopath, label = self.video_info.iloc[index]

        return vpath, audiopath, label

    def __len__(self):
        return len(self.video_info)

    def encode_label(self, label_name):
        return self.label_dict_encode[label_name]

    def decode_label(self, label_code):
        return self.label_dict_decode[label_code]

