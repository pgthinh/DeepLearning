Link drive h5 + pickles folder: https://drive.google.com/drive/folders/1PjHxYK8o2S7YX-Jn8C9Fgyr8-LYvR_Yv?usp=sharing

Trước khi chạy copy Deep_Learning/Clotho/hdf5s vào /m-LTM-Audio-Text-Retrieval/data/Clotho

Tương tự với pickles Deep_Learning/Clotho/pickles vào /m-LTM-Audio-Text-Retrieval/data/Clotho

Tương tự với resnet38: copy folder Deep_Learning/m-LTM-Audio-Text-Retrieval/pretrained_models vào /m-LTM-Audio-Text-Retrieval

Nhớ là chạy các file train ở:
cd /m-LTM-Audio-Text-Retrieval/

Create conde environment with dependencies:  conda env create -f environment.yaml -n [env-name]&&conda activate [env-name]

python train.py -n exp -c m-ltm-settings 
