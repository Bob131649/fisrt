"""
Based on https://github.com/sfujim/BCQ
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.net import Critic, Value, ActorVAE
import copy


class Latent(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, min_v, max_v,
                 device, discount=0.99, tau=0.001, vae_lr=1e-4, actor_lr=1e-4, critic_lr=1e-4,
                 max_latent_action=3, expectile=0.9, kl_beta=0.5, doubleq_min=0.8):
        super(Latent, self).__init__()

        self.device = torch.device(device)

        self.qnet = Critic(state_dim, action_dim).to(self.device)
        encoded_state_dim = self.qnet.encoder.output_dim

        self.actor_vae = ActorVAE(encoded_state_dim, action_dim, latent_dim, max_latent_action, self.device).to(self.device)
        self.actorvae_optimizer = torch.optim.Adam(
            list(self.qnet.encoder.parameters()) + list(self.actor_vae.parameters()), lr=vae_lr
        )

        # self.qnet_target = Critic(state_dim, action_dim).to(self.device)
        self.qnet_target = copy.deepcopy(self.qnet)

        # self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.qnet_optimizer = torch.optim.Adam(
            self.qnet.parameters(), lr=critic_lr
        )

        self.vnet = Value(state_dim).to(self.device)
        self.vnet_optimizer = torch.optim.Adam(
            self.vnet.parameters(), lr=critic_lr
        )

        self.latent_dim = latent_dim
        self.max_latent_action = max_latent_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau

        self.kl_beta = kl_beta
        self.doubleq_min = doubleq_min

        self.g_clip = 0.5
        self.min_v, self.max_v = min_v, max_v

    def encode_state(self, state, target=False):
        qnet = self.qnet_target if target else self.qnet
        return qnet.encoder(state)

    def select_action(self, state, need_q=True):
        with torch.no_grad():
            raw_state = state
            state = self.encode_state(state)
            latent_a = None
            action = self.actor_vae.decode(state, z=latent_a)

            if need_q:
                q1, q2 = self.qnet(raw_state, action)
                q1, q2 = q1.item(), q2.item()
            else:
                q1, q2 = None, None

        return action.cpu().data.numpy().flatten(), q1, q2

    def add_proprio_noise(self, state, noise_std):
        noisy_state = dict(state)
        noisy_state["proprio"] = noisy_state["proprio"] + torch.randn_like(noisy_state["proprio"]) * noise_std
        return noisy_state

    def get_a_q(self, state, action):
        target_q1, target_q2 = self.qnet(state, action)
        target_q = torch.min(target_q1, target_q2)
        return target_q

    def get_v(self, state):
        return self.vnet(state)

    def get_pi_q(self, state, q_net, gen_net, type='mean', use_noise=True):
        encoded_state = q_net.encoder(state)
        latent_action = None
        actor_action = gen_net.decode(encoded_state, z=latent_action)
        target_q1, target_q2 = q_net(state, actor_action)

        if type == 'none':
            target_q = self.get_min_q(target_q1, target_q2)
        if type == 'min':
            target_q = torch.min(target_q1, target_q2)
        if type == 'max':
            target_q = torch.max(target_q1, target_q2)
        if type == 'mean':
            target_q = (target_q1 + target_q2) / 2
        if type == 'q1':
            target_q = target_q1
        return target_q

    def get_min_q(self, q1, q2):
        q = torch.min(q1, q2) * self.doubleq_min + torch.max(q1, q2) * (1 - self.doubleq_min)
        return q

    def train_value(self, state_random_inm, state_expert_inm, state_expert_ine, a_expert_ine):
        with torch.no_grad():
            next_q1, next_q2 = self.qnet(state_expert_ine, a_expert_ine)
            target_v_expert = (next_q1 + next_q2) / 2

        state_expert_inm = self.add_proprio_noise(state_expert_inm, 0.05)

        current_v_expert = self.vnet.forward_norm(state_expert_inm)
        current_v_random = self.vnet.forward_norm(state_random_inm)
        target_v_expert = target_v_expert / self.vnet.scale

        v_loss_expert = F.mse_loss(current_v_expert, target_v_expert)
        v_loss_random = torch.mean(current_v_random.pow(2))
        v_loss = v_loss_expert + v_loss_random * 1.0

        self.vnet_optimizer.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vnet.parameters(), max_norm=self.g_clip)
        self.vnet_optimizer.step()
        return (
            v_loss.item(),
            current_v_random.mean().item() * self.vnet.scale.item(),
            current_v_expert.mean().item() * self.vnet.scale.item(),
        )

    def train_q(self, batch, idx):
        state, action, next_state, reward, not_done, next_action = batch

        with torch.no_grad():
            next_q1, next_q2 = self.qnet_target(next_state, next_action)
            next_target_v = (next_q1 + next_q2) / 2
            target_q = reward + not_done * self.discount * next_target_v.clamp(self.min_v, self.max_v)

        current_q1, current_q2 = self.qnet.forward_norm(state, action)
        target_q = target_q / self.qnet.scale

        critic_loss_1 = F.mse_loss(current_q1, target_q)
        critic_loss_2 = F.mse_loss(current_q2, target_q)
        critic_loss = critic_loss_1 + critic_loss_2

        for i in range(len(action)):
            if reward[i] != 0:
                print(reward[i], not_done[i], target_q[i].item(), next_target_v[i].item(), current_q1[i], current_q2[i])
                print(next_q1[i].item(), next_q2[i].item())
                
        self.qnet_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=self.g_clip)
        self.qnet_optimizer.step()

        if idx % 1 == 0:
            for param, target_param in zip(self.qnet.parameters(), self.qnet_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), current_q1.mean().item() * self.qnet.scale.item()

    def train_policy(self, state, action):
        state = self.qnet.encoder(state)
        recons_action, mu, log_var = self.actor_vae(state, action)
        recons_loss_ori = F.mse_loss(recons_action, action, reduction='none')
        recon_loss = torch.sum(recons_loss_ori, 1).view(-1, 1).mean()

        free_bits = 0.4
        kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        free_bits_tensor = torch.tensor(free_bits, device=kl_per_dim.device)
        kl_freebits = torch.maximum(kl_per_dim, free_bits_tensor)
        kld_loss = kl_freebits.sum(dim=1).view(-1, 1).mean()
        actor_vae_loss = recon_loss + self.kl_beta * kld_loss

        self.actorvae_optimizer.zero_grad()
        actor_vae_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.qnet.encoder.parameters()) + list(self.actor_vae.parameters()), max_norm=self.g_clip
        )
        self.actorvae_optimizer.step()

        return recon_loss.item(), kld_loss.item()

    def save(self, filename, directory):
        torch.save(self.qnet.state_dict(), '%s/%s_qnet.pth' % (directory, filename))
        torch.save(self.qnet_optimizer.state_dict(), '%s/%s_qnet_optimizer.pth' % (directory, filename))
        torch.save(self.qnet_target.state_dict(), '%s/%s_qnet_target.pth' % (directory, filename))

        torch.save(self.vnet.state_dict(), '%s/%s_vnet.pth' % (directory, filename))
        torch.save(self.vnet_optimizer.state_dict(), '%s/%s_vnet_optimizer.pth' % (directory, filename))

        torch.save(self.actor_vae.state_dict(), '%s/%s_actor_vae.pth' % (directory, filename))
        torch.save(self.actorvae_optimizer.state_dict(), '%s/%s_actor_vae_optimizer.pth' % (directory, filename))

    def load_scaled_state_dict(self, module, path):
        state_dict = torch.load(path)
        if "scale" not in state_dict and "scale" in module.state_dict():
            state_dict["scale"] = torch.ones_like(module.state_dict()["scale"])
        module.load_state_dict(state_dict, strict=False)
        return state_dict

    def load(self, filename, directory):
        self.load_scaled_state_dict(self.qnet, '%s/%s_qnet.pth' % (directory, filename))

        self.qnet_optimizer.load_state_dict(torch.load('%s/%s_qnet_optimizer.pth' % (directory, filename)))
        self.load_scaled_state_dict(self.qnet_target, '%s/%s_qnet_target.pth' % (directory, filename))

        self.load_scaled_state_dict(self.vnet, '%s/%s_vnet.pth' % (directory, filename))
        try:
            self.vnet_optimizer.load_state_dict(torch.load('%s/%s_vnet_optimizer.pth' % (directory, filename)))
        except ValueError as exc:
            print(f"Skipping vnet optimizer state due to optimizer parameter mismatch: {exc}")
        self.actor_vae.load_state_dict(torch.load('%s/%s_actor_vae.pth' % (directory, filename)))
        self.actorvae_optimizer.load_state_dict(torch.load('%s/%s_actor_vae_optimizer.pth' % (directory, filename)))
