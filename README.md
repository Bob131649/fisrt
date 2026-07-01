## train q until loss converge, check eval_expert_q.png and eval_random_q.png
```shell
$ python train_reference.py --ExpID 2 --env_name maze2d-large-v1 --plot --mode q --device cuda:0
$ python train_reference.py --ExpID 2 --env_name antmaze-large-diverse --plot --mode q --device cuda:0

```

## train v until loss converge, check eval_expert_v.png and eval_random_v.png
```shell
$ python train_reference.py --ExpID 3 --env_name maze2d-large-v1 --plot --mode v --device cuda:0 --load_model 2
$ python train_reference.py --ExpID 3 --env_name antmaze-large-diverse --plot --mode v --device cuda:0 --load_model 2

```

## train lapo with reference v, check vae_eval_fig.png
```shell
$ python train_all.py --ExpID 100 --env_name maze2d-large-v1 --plot --device cuda:0 --load_model 3
$ python train_all.py --ExpID 100 --env_name antmaze-large-diverse-v2 --plot --device cuda:0 --load_model 3
```