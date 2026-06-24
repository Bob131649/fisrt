## train BC vae, check vae_eval_fig.png
```shell
$ python train_reference.py --ExpID 1 --env_name maze2d-large-v1 --plot --mode pi --device cuda:0
```

## train q, check eval_expert_q.png and eval_random_q.png
```shell
$ python train_reference.py --ExpID 2 --env_name maze2d-large-v1 --plot --mode q --device cuda:0 --load_model 1

```

## train v, check eval_expert_v.png and eval_random_v.png
```shell
$ python train_reference.py --ExpID 3 --env_name maze2d-large-v1 --plot --mode v --device cuda:0 --load_model 2

```

## train lapo with reference v, check vae_eval_fig.png
```shell
$ python train_all.py --ExpID 100 --env_name maze2d-large-v1 --plot --device cuda:0 --load_model 3

```