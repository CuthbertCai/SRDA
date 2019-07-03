## Learning Smooth Representation for Unsupervised Domain Adaptation  
This is the implementation of [SRDA][4] for VisDA 2017 in Pytorch.  

### Requirements
> `pip install -r requirements.txt`

### Dataset  
> Refer to [VisDA 2017][1] repo.

### Train
> 1. To train SRDA (fgsm), `python res_train_main.py --save_model --attack fgsm --train_path TRAIN_PATH --val_path VAL_PATH`.  
> 2. To train SRDA (vat), `python res_train_main.py --save_model --attack vat --train_path TRAIN_PATH --val_path VAL_PATH`.
> 3. To train SRDA (ran), `python random_noise_exp.py --save_model --train_path TRAIN_PATH --val_path VAL_PATH`.
> `TRAIN_PATH` and `VAL_PATH` refer to the real path of your dataset. If you want to tune other hyper-parameters,  
> see args in `res_train_main.py` and `random_noise_exp.py`.

### Reference
> We refer to some other repos.
> 1. [MCD_DA][2]
> 2. [advertorch][3]

[1]:https://github.com/VisionLearningGroup/taskcv-2017-public
[2]:https://github.com/mil-tokyo/MCD_DA
[3]:https://github.com/BorealisAI/advertorch
[4]:https://arxiv.org/abs/1905.10748