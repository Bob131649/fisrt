import torch

import algos.vision_algos_guide as algos


def resolve_device(cli_device, trained_device):
    if cli_device:
        return cli_device
    if trained_device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return trained_device or ("cuda" if torch.cuda.is_available() else "cpu")


def _image_tensor(image, device):
    image_t = torch.as_tensor(image, dtype=torch.float32, device=device)
    if image_t.ndim == 3:
        image_t = image_t.unsqueeze(0)
    if image_t.shape[-1] == 3:
        image_t = image_t.permute(0, 3, 1, 2)
    if image_t.max() > 1:
        image_t = image_t / 255.0
    return image_t.float()


class DeployPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.device = policy.device
        self.vae = False

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def _obs(self, policy_state, image):
        state_t = torch.as_tensor(policy_state.reshape(1, -1), dtype=torch.float32, device=self.device)
        return {"proprio": state_t, "image": _image_tensor(image, self.device)}

    def select_action(self, policy_state, image):
        return self.policy.select_action(self._obs(policy_state, image))


def load_latent_policy(args, variant, stats, image_size):
    if args.load_best:
        raise ValueError("deploy supports vision_train_all checkpoints; --load-best is not supported.")
    device = resolve_device(args.device, variant.get("device"))
    state_dim = int(stats.get("state_dim", variant.get("state_dim", len(stats["state_mean"]))))
    action_dim = int(stats.get("action_dim", variant.get("action_dim", len(stats["action_mean"]))))

    policy = algos.Latent(
        state_dim,
        action_dim,
        action_dim * 2,
        0.0,
        100.0,
        device=device,
        max_latent_action=float(variant.get("max_latent_action", 0.675)),
        expectile=float(variant.get("expectile", 0.9)),
        kl_beta=float(variant.get("kl_beta", 1.0)),
        doubleq_min=float(variant.get("doubleq_min", 1.0)),
    )
    policy.load_policy(args.model_name, str(args.model_dir))
    policy.eval()
    return DeployPolicy(policy), device
