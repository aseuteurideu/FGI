import os
import glob
import random

random.seed(23)

source_folder = 'FakeAVCeleb_v1.2'
target_folder = 'FakeAVCeleb_v1.2_inDFDCformat'
races = ['African', 'Asian (East)', 'Asian (South)', 'Caucasian (American)', 'Caucasian (European)']
fake_types = ['FakeVideo-FakeAudio', 'FakeVideo-RealAudio', 'RealVideo-FakeAudio']
genders = ['men', 'women']
import shutil

# mkdir train, test inside target folder
# based on https://arxiv.org/pdf/2109.02993.pdf, test set is 70 real and 70 fake videos
# The test videos belongs to the individuals not in training set so that it would not have any bias in the results.
# if 70 real data and there are 5 races, then 70/5 = 14 id for each races. Then 7 men 7 women.

train_folder = os.path.join(target_folder, 'train')
test_folder = os.path.join(target_folder, 'test')  # 70 real and 70 fake. For each real, select one fake either from FakeVideo-FakeAudio, FakeVideo-RealAudio, or RealVideo-FakeAudio
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

train_real_folder = os.path.join(target_folder, 'train', 'real')
test_real_folder = os.path.join(target_folder, 'test', 'real')
os.makedirs(train_real_folder, exist_ok=True)
os.makedirs(test_real_folder, exist_ok=True)

train_fake_folder = os.path.join(target_folder, 'train', 'fake')
test_fake_folder = os.path.join(target_folder, 'test', 'fake')
os.makedirs(train_fake_folder, exist_ok=True)
os.makedirs(test_fake_folder, exist_ok=True)

for race in races:
    race_alphanum = ''.join(e for e in race if e.isalnum())
    for gender in genders:
        real_folder = os.path.join(source_folder, 'RealVideo-RealAudio', race, gender)

        real_counter = 0
        for id_folder in os.listdir(real_folder):
            real_counter += 1
            if real_counter <= 7:  # test
                # real
                source_file = glob.glob(os.path.join(real_folder, id_folder, '*.mp4'))
                assert len(source_file) == 1
                source_file = source_file[0]

                target_filename = 'RealVideo-RealAudio-' + race_alphanum + '-' + gender + '-' + id_folder + '_' + os.path.basename(source_file)
                target_file = os.path.join(test_real_folder, target_filename)
                shutil.copyfile(source_file, target_file)

                # fake balance
                fake_type = random.choice(fake_types)
                source_file = glob.glob(os.path.join(source_folder, fake_type, race, gender, id_folder, '*.mp4'))
                source_file = random.choice(source_file)
                target_filename = fake_type + '-' + race_alphanum + '-' + gender + '-' + id_folder + '_' + os.path.basename(source_file)
                target_file = os.path.join(test_fake_folder, target_filename)
                shutil.copyfile(source_file, target_file)


            else: # train
                # real
                source_file = glob.glob(os.path.join(real_folder, id_folder, '*.mp4'))
                assert len(source_file) == 1
                source_file = source_file[0]

                target_filename = 'RealVideo-RealAudio-' + race_alphanum + '-' + gender + '-' + id_folder + '_' + os.path.basename(source_file)
                target_file = os.path.join(train_real_folder, target_filename)
                shutil.copyfile(source_file, target_file)

                # fake (imbalanced)
                for fake_type in fake_types:
                    source_files = glob.glob(os.path.join(source_folder, fake_type, race, gender, id_folder, '*.mp4'))
                    for source_file in source_files:
                        target_filename = fake_type + '-' + race_alphanum + '-' + gender + '-' + id_folder + '_' + os.path.basename(source_file)
                        target_file = os.path.join(train_fake_folder, target_filename)
                        shutil.copyfile(source_file, target_file)


