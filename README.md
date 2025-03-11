# Detecting Audio-Visual Deepfakes with Fine-Grained Inconsistencies
This repository contains the implementation of Audio Visual Deepfake Detection method proposed in the paper:  
  
Marcella Astrid, Enjie Ghorbel, and Djamila Aouada, Detecting Audio-Visual Deepfakes with Fine-Grained Inconsistencies, BMVC 2024.  
<!---Links: [[PDF]](https://www.arxiv.org/pdf/2407.11650)-->

# Dependencies
Create conda environment with package inside the package list `conda create -n myenv --file package-list.txt`
  
# Prepare data (DFDC)
1) Download DFDC dataset from [here](https://www.kaggle.com/c/deepfake-detection-challenge/data). 
  
2) Store the train and test videos as follows:  

   ```
   train/real/{videoname}.mp4  
   train/fake/{videoname}.mp4  
   test/real/{videoname}.mp4  
   test/fake/{videoname}.mp4
   ```
  
   If you wish to use the same videos as used by the paper authors, please refer to `train_fake_list.txt`,  `train_real_list.txt`,  `test_fake_list.txt`, and `test_real_list.txt`. 
  
3) Once the videos have been placed at the above mentioned paths, run `python pre-process.py --out_dir train` and `python pre-process.py --out_dir test` for pre-processing the videos.  
  
4) After the above step, you can delete `pyavi`, `pywork`, `pyframes` and `pycrop` directories under `train` and `test` folders. (Do not delete `pytmp` folder please!)  
  
5) Collect video paths in csv files by running `python write_csv.py --out_dir . ` command. Also can create the small set version (e.g., used in Table 3) with `python write_csv.py --out_dir . --small`  

# Prepare data (FakeAVCeleb)
1) Download FakeAVCeleb dataset from [here](https://github.com/DASH-Lab/FakeAVCeleb/blob/main/dataset/README.md)
2) Run `python preprocess_FakeAVCeleb_to_DFDCformat.py`. See `fakeavceleb_test_fake.txt` and `fakeavceleb_test_real.txt` for list of videos we are using.
3) Use instructions in Prepare data (DFDC). Adjust the `--out_dir` in step 3 respectively. Also add `--dont_crop_face` option in step 3. Use `write_csv_fakeavceleb.py` for step 5

# Training
```
python my_train.py --out_dir . --epochs 100 --num_workers 7 --with_att --using_pseudo_fake 
```
Remove `--with_att` for model without attention.

Remove `--using_pseudo_fake` for training without pseudo-fake.

Add `--spatial_size 14` to change the spatial size to 14. Change the 14 to different number. (Figure 5(c) ablation)

Add `--aud_min_fake_len 0.5` and `--vis_min_fake_len 0.5` to change the minimum length of pseudo fake. (Figure 5(b) ablation). For ~0, change the 0.5 to 2 (the default value). Ratio value between 0 to 1. More than 1, it is regarded as number of frames instead

Similarly for the maximum length with `--aud_max_fake_len 0.5` and `--vis_max_fake_len 0.5`. (Figure 5(a) ablation). Value -1 represent the maximum length (same to ratio = 1)

# Testing
Final model weight file: [drive](https://drive.google.com/drive/folders/1ecffYG6KWZjTnAupP75JndOz8saYLrW5?usp=sharing)

```
python my_train.py --out_dir . --test log_tmp/v5_mydf-natt1.0-upf-PF0,1,2_A5_afl2,-1_vfl2,-1_224_r18_bs8_lr0.0001/model/model_best_epoch99.pth.tar --with_att
```
Change the path of model file accordingly in the --test argument.  
  
For computing AUC score, run `python my_test.py --folder test_results/v5_mydf-natt1.0-upf-PF0,1,2_A5_afl2,-1_vfl2,-1_224_r18_bs8_lr0.0001/model_best_epoch99/imbalance` after executing the above command, and see the result in test_results folder. 

See the results in the `output.txt` inside `test_results/v5_mydf-natt1.0-upf-PF0,1,2_A5_afl2,-1_vfl2,-1_224_r18_bs8_lr0.0001/model_best_epoch99/imbalance`
  
For testing with FakeAVCeleb:

```
python my_train.py --out_dir . --test log_tmp/v5_mydf-natt1.0-upf-PF0,1,2_A5_afl2,-1_vfl2,-1_224_r18_bs8_lr0.0001/model/model_best_epoch99.pth.tar --with_att --dataset fakeavceleb
python my_test.py --folder test_results/v5_mydf-natt1.0-upf-PF0,1,2_A5_afl2,-1_vfl2,-1_224_r18_bs8_lr0.0001/model_best_epoch99/balance --dataset fakeavceleb
```
See the results in the `output.txt` inside `test_results/v5_mydf-natt1.0-upf-PF0,1,2_A5_afl2,-1_vfl2,-1_224_r18_bs8_lr0.0001/model_best_epoch99/balance`

# Reference
If you use the code, please cite the paper -
```
@InProceedings{astrid2024detecting,
  author       = "Astrid, Marcella and Ghorbel, Enjie and Aouada, Djamila",
  title        = "Detecting Audio-Visual Deepfakes with Fine-Grained Inconsistencies",
  booktitle    = "British Machine Vision Conference (BMVC)",
  year         = "2024",
}
```
# Acknowledgements
Thanks to the code available at https://github.com/abhinavdhall/deepfake/tree/main/ACM_MM_2020, https://github.com/TengdaHan/DPC and https://github.com/joonson/syncnet_python.  
  
