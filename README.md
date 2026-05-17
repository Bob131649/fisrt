# Latent-variable Advantage-weighted Policy Optimization for Offline Reinforcement Learning

目前这个仓库是更改过的LAPO不带actor的部分 实现VAE sample from inital noise and generate action from actorvae decoder。

在这个代码开发的基础之上 要注意align different dataset’s normalize

![LAPO-framwork](https://github.com/pcchenxi/LAPO-offlienRL/blob/main/figs/LAPO.jpg)

## Requirements

- python=3.7.11
- [Datasets for Deep Data-Driven Reinforcement Learning (D4RL)](https://github.com/rail-berkeley/d4rl)
- torch=1.10.0

## Scripts for D4RL dataset

visualize model result 
```shell
$ python visualize_model_result.py --results_dir ./results/Exp0010/
```

train ope and pay attention to the args target policy and dataset_path
```shell
$ python train_ope.py --ExpID 514 --env_name maze2d-large-v1 --target_policy_mode vae --plot --target_policy_dir ~/first/LAPO-offlienRL_without_Actor/results/Exp0010/maze2d-large-v1-1000/
```

train v2
```shell
$ python train_v2.py --ExpID 513 --env_name antmaze-large-diverse-v2 --dataset_path ~/first/dataset/generated_dataset/antmaze/expert/antmaze-expert-success250.hdf5 --plot
```

eval model
```
$ python eval_policy.py --model_dir ./results/Exp0010/maze2d-large-v1-1000/ --render
```


## Expected results

You will get following results using --seed: 123(red) 456(green) 789(blue)

![LAPO-framwork](https://github.com/pcchenxi/LAPO-offlienRL/blob/main/figs/result_3seeds.jpg)

## Citing
If you find this code useful, please cite our paper:
```
@article{chen2022lapo,
  title={Lapo: Latent-variable advantage-weighted policy optimization for offline reinforcement learning},
  author={Chen, Xi and Ghadirzadeh, Ali and Yu, Tianhe and Wang, Jianhao and Gao, Alex Yuan and Li, Wenzhe and Bin, Liang and Finn, Chelsea and Zhang, Chongjie},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={36902--36913},
  year={2022}
}
```

## Note
+ If you have any questions, please contact me: pcchenxi@gmail.com
