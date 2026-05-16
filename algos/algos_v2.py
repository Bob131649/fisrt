"""
Based on https://github.com/sfujim/BCQ
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.net_v2 import Actor, Critic, ActorVAE, OPEValue


class FrozenPolicy(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        latent_dim,
        max_latent_action,
        device,
        policy_mode="vae",
        ope_state_mean=None,
        ope_state_std=None,
        ope_action_mean=None,
        ope_action_std=None,
        target_policy_state_mean=None,
        target_policy_state_std=None,
        target_policy_action_mean=None,
        target_policy_action_std=None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.policy_mode = policy_mode
        self.ope_state_mean = self._build_stat_tensor(ope_state_mean)
        self.ope_state_std = self._build_stat_tensor(ope_state_std)
        self.ope_action_mean = self._build_stat_tensor(ope_action_mean)
        self.ope_action_std = self._build_stat_tensor(ope_action_std)
        self.target_policy_state_mean = self._build_stat_tensor(target_policy_state_mean)
        self.target_policy_state_std = self._build_stat_tensor(target_policy_state_std)
        self.target_policy_action_mean = self._build_stat_tensor(target_policy_action_mean)
        self.target_policy_action_std = self._build_stat_tensor(target_policy_action_std)
        self.actor_vae = ActorVAE(
            state_dim, action_dim, latent_dim, max_latent_action, self.device
        ).to(self.device)
        self.actor = None
        if policy_mode == "lapo":
            self.actor = Actor(state_dim, latent_dim, max_latent_action).to(self.device)

    def _build_stat_tensor(self, value):
        if value is None:
            return None
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _normalize_with_stats(self, value, mean, std):
        return (value - mean) / (std + 1e-6)

    def _unnormalize_with_stats(self, value, mean, std):
        return value * (std + 1e-6) + mean

    def load(self, filename, directory):
        self.actor_vae.load_state_dict(
            torch.load(f"{directory}/{filename}_actor_vae.pth", map_location=self.device)
        )
        if self.actor is not None:
            self.actor.load_state_dict(
                torch.load(f"{directory}/{filename}_actor.pth", map_location=self.device)
            )

        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    def select_action_tensor(self, state):
        if self.actor is None:
            latent_action = None
        else:
            latent_action = self.actor(state)
        return self.actor_vae.decode(state, z=latent_action)

    def select_action_tensor_ope(self, ope_state):
        if self.ope_state_mean is None or self.target_policy_state_mean is None:
            return self.select_action_tensor(ope_state)

        raw_state = self._unnormalize_with_stats(
            ope_state, self.ope_state_mean, self.ope_state_std
        )
        target_policy_state = self._normalize_with_stats(
            raw_state, self.target_policy_state_mean, self.target_policy_state_std
        )
        target_policy_action = self.select_action_tensor(target_policy_state)
        raw_action = self._unnormalize_with_stats(
            target_policy_action,
            self.target_policy_action_mean,
            self.target_policy_action_std,
        )
        return self._normalize_with_stats(
            raw_action, self.ope_action_mean, self.ope_action_std
        )

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.select_action_tensor(state)
        return action.cpu().data.numpy().flatten(), 0.0, 0.0


class Latent(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, min_v, max_v, 
                 device, discount=0.99, tau=0.001, vae_lr=1e-4, actor_lr=1e-4, critic_lr=1e-4, 
                 max_latent_action=3, expectile=0.5, kl_beta=0.5, doubleq_min=0.8,
                 target_policy_dir="", target_policy_name="model", target_policy_mode="lapo",
                 policy_mode="vae", ope_state_mean=None, ope_state_std=None,
                 ope_action_mean=None, ope_action_std=None,
                 target_policy_state_mean=None, target_policy_state_std=None,
                 target_policy_action_mean=None, target_policy_action_std=None):
        super(Latent, self).__init__()

        self.device = torch.device(device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-5)

        self.latent_dim = latent_dim
        self.max_latent_action = max_latent_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.tau_act = tau
        self.tau_vae = tau

        self.expectile = expectile
        self.kl_beta = kl_beta
        self.initial_kl_beta = kl_beta
        self.doubleq_min = doubleq_min

        self.g_clip = 0.5

        self.min_v, self.max_v = min_v, max_v 
        self.ope_mode = bool(target_policy_dir)
        if policy_mode == "vae":
            policy_mode = "vae_bc"
        if policy_mode not in ("lapo", "vae_bc"):
            raise ValueError(f"Unsupported policy_mode: {policy_mode}")
        self.policy_mode = policy_mode

        if self.ope_mode:
            if target_policy_mode == "vae":
                target_policy_mode = "vae_bc"
            if target_policy_mode not in ("lapo", "vae_bc"):
                raise ValueError(f"Unsupported target_policy_mode: {target_policy_mode}")
            self.target_policy = FrozenPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                latent_dim=latent_dim,
                max_latent_action=max_latent_action,
                device=device,
                policy_mode="lapo" if target_policy_mode == "lapo" else "vae_bc",
                ope_state_mean=ope_state_mean,
                ope_state_std=ope_state_std,
                ope_action_mean=ope_action_mean,
                ope_action_std=ope_action_std,
                target_policy_state_mean=target_policy_state_mean,
                target_policy_state_std=target_policy_state_std,
                target_policy_action_mean=target_policy_action_mean,
                target_policy_action_std=target_policy_action_std,
            )
            self.target_policy.load(target_policy_name, target_policy_dir)
            self.value = OPEValue(state_dim).to(self.device)
            self.value_target = copy.deepcopy(self.value)
            self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=critic_lr, weight_decay=1e-5)
            self.actor_vae = None
            self.actor_vae_target = None
            self.actorvae_optimizer = None
            self.actor = None
            self.actor_target = None
            self.actor_optimizer = None
        else:
            self.target_policy = None
            self.value = None
            self.value_target = None
            self.value_optimizer = None
            self.actor_vae = ActorVAE(state_dim, action_dim, latent_dim, max_latent_action, self.device).to(self.device)
            self.actor_vae_target = copy.deepcopy(self.actor_vae)
            self.actorvae_optimizer = torch.optim.Adam(self.actor_vae.parameters(), lr=vae_lr, weight_decay=1e-5)

            self.actor = Actor(state_dim, latent_dim, max_latent_action).to(self.device)
            self.actor_target = copy.deepcopy(self.actor)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=1e-5)

    def copy_bn_param(self):
        if self.ope_mode:
            return None
        source_state = self.actor_vae.state_dict()
        for name, module in self.actor_vae_target.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                module.running_mean.copy_(source_state[f'{name}.running_mean'])
                module.running_var.copy_(source_state[f'{name}.running_var'])

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            if self.ope_mode:
                action = self.target_policy.select_action_tensor_ope(state)
            else:
                latent_a = None if self.policy_mode == "vae_bc" else self.actor(state)
                action = self.actor_vae.decode(state, z=latent_a)
            q1, q2 = self.critic(state, action)
            # v = self.critic.v(state)
            
        return action.cpu().data.numpy().flatten(), q1.item(), q2.item()

    def kl_loss(self, mu, log_var):
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).view(-1, 1)
        return kld_loss

    def get_pi_q(self, state, actor_net, critic_net, gen_net, type='none', use_noise=True):
        latent_action = actor_net(state)
        if use_noise:
            latent_action += (torch.randn_like(latent_action) * 0.1).clamp(-0.3, 0.3)

        actor_action = gen_net.decode(state, z=latent_action)
        target_q1, target_q2 = critic_net(state, actor_action)
        
        if type == 'none':
            target_q = self.get_min_q(target_q1, target_q2)
        if type == 'min':
            target_q = torch.min(target_q1, target_q2)
        if type == 'max':
            target_q = torch.max(target_q1, target_q2)
        if type == 'q1':
            target_q = target_q1
        return target_q

    def get_min_q(self, q1, q2):
        q = torch.min(q1, q2)*self.doubleq_min + torch.max(q1, q2)*(1-self.doubleq_min)
        return q

    def estimate_value(self, states):
        if not self.ope_mode:
            raise RuntimeError("estimate_value is only available in OPE mode.")
        with torch.no_grad():
            return self.value(states)

    def train_step(self, batch, iter_id, ):
        state = batch['state'].to(self.device).float()
        action = batch['action'].to(self.device).float()
        next_state = batch['next_state'].to(self.device).float()
        reward = batch['reward'].to(self.device).view(-1,1).float()
        not_done = batch['not_done'].to(self.device).view(-1,1)

        if self.ope_mode:
            with torch.no_grad():
                next_target_v = self.value_target(next_state)
                target_q = reward + not_done * self.discount * next_target_v.clamp(
                    self.min_v, self.max_v
                )

            current_q1, current_q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.g_clip)
            self.critic_optimizer.step()

            policy_action = self.target_policy.select_action_tensor_ope(state)
            with torch.no_grad():
                q1_pi, q2_pi = self.critic(state, policy_action)
                q_pi = self.get_min_q(q1_pi, q2_pi)

            current_v = self.value(state)
            value_loss = F.mse_loss(current_v, q_pi)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=self.g_clip)
            self.value_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.value.parameters(), self.value_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            sample_w = np.ones(state.shape[0], dtype=np.float32)
            return sample_w, current_v.mean().item(), critic_loss.item(), value_loss.item(), None

        with torch.no_grad():
            next_target_v = self.get_pi_q(next_state, self.actor_target, self.critic_target, 
                                          self.actor_vae_target, use_noise=True)     
            target_q = reward + not_done * self.discount * next_target_v.clamp(self.min_v, self.max_v)

        # Critic Training
        current_q1, current_q2 = self.critic(state, action)
        critic_loss_1 = F.mse_loss(current_q1, target_q)
        critic_loss_2 = F.mse_loss(current_q2, target_q)
        critic_loss = critic_loss_1 + critic_loss_2
    
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.g_clip)
        self.critic_optimizer.step()
        
        loss_rc, loss_kl, a_loss, loss_std, loss_mean= None, None, None, None, None

        if iter_id % 1 == 0:
            with torch.no_grad():
                q1_a, q2_a = self.critic(state, action)
                # q_a = torch.min(q1_a, q2_a)
                q_a = (q1_a + q2_a)/2
                q_pi = self.get_pi_q(state, self.actor, self.critic, self.actor_vae,
                                     type='min', use_noise=False)
                adv = q_a - q_pi
                weight = torch.where(adv < 0, 1-self.expectile, self.expectile).detach()
            
            # train weighted CVAE
            recons_action, mu, log_var = self.actor_vae(state, action)
            recons_loss_ori = F.mse_loss(recons_action, action, reduction='none')
            recon_loss = torch.sum(recons_loss_ori, 1).view(-1, 1)

            free_bits = 0.5
            kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # shape: [batch_size, latent_dim]

            free_bits_tensor = torch.tensor(free_bits, device=kl_per_dim.device)
            kl_freebits = torch.maximum(kl_per_dim, free_bits_tensor)
            kld_loss = kl_freebits.sum(dim=1).view(-1, 1)
            actor_vae_loss = recon_loss + self.kl_beta * kld_loss

            actor_vae_loss = actor_vae_loss.mean()
            self.actorvae_optimizer.zero_grad()
            actor_vae_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_vae.parameters(), max_norm=self.g_clip)
            self.actorvae_optimizer.step()

            # Update Target Networks
            loss_rc = recons_loss_ori.mean().item()
            loss_kl = kld_loss.mean().item()
            a_loss = q_pi.mean().item()

        if iter_id % 2 == 0:
            # train latent policy 
            latent_actor_action = self.actor(state)
            latent_actor_action = latent_actor_action

            actor_action = self.actor_vae.decode(state, z=latent_actor_action)
            q1_pi, q2_pi = self.critic(state, actor_action)
            q_pi = torch.min(q1_pi, q2_pi)

            actor_qloss = -q_pi.mean()
            actor_reg_loss = torch.mean(latent_actor_action ** 2)
            actor_loss = actor_qloss + actor_reg_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.g_clip)
            self.actor_optimizer.step()
            a_loss = -actor_loss.item()

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau_act * param.data + (1 - self.tau_act) * target_param.data)
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor_vae.parameters(), self.actor_vae_target.parameters()):
            target_param.data.copy_(self.tau_vae * param.data + (1 - self.tau_vae) * target_param.data)

        return weight.cpu().numpy(), a_loss, critic_loss.item(), loss_rc, loss_kl
    
    def save(self, filename, directory):
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.critic_optimizer.state_dict(), '%s/%s_critic_optimizer.pth' % (directory, filename))
        torch.save(self.critic_target.state_dict(), '%s/%s_critic_target.pth' % (directory, filename))

        if self.ope_mode:
            torch.save(self.value.state_dict(), '%s/%s_value.pth' % (directory, filename))
            torch.save(self.value_optimizer.state_dict(), '%s/%s_value_optimizer.pth' % (directory, filename))
            torch.save(self.value_target.state_dict(), '%s/%s_value_target.pth' % (directory, filename))
            return

        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, filename))

        torch.save(self.actor_vae.state_dict(), '%s/%s_actor_vae.pth' % (directory, filename))
        torch.save(self.actorvae_optimizer.state_dict(), '%s/%s_actor_vae_optimizer.pth' % (directory, filename))
        torch.save(self.actor_vae_target.state_dict(), '%s/%s_actor_vae_target.pth' % (directory, filename))

    def load(self, filename, directory):
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename), map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load('%s/%s_critic_optimizer.pth' % (directory, filename), map_location=self.device))
        self.critic_target.load_state_dict(torch.load('%s/%s_critic_target.pth' % (directory, filename), map_location=self.device))

        if self.ope_mode:
            self.value.load_state_dict(torch.load('%s/%s_value.pth' % (directory, filename), map_location=self.device))
            self.value_optimizer.load_state_dict(torch.load('%s/%s_value_optimizer.pth' % (directory, filename), map_location=self.device))
            self.value_target.load_state_dict(torch.load('%s/%s_value_target.pth' % (directory, filename), map_location=self.device))
            return

        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename), map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load('%s/%s_actor_optimizer.pth' % (directory, filename), map_location=self.device))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, filename), map_location=self.device))

        self.actor_vae.load_state_dict(torch.load('%s/%s_actor_vae.pth' % (directory, filename), map_location=self.device))
        self.actorvae_optimizer.load_state_dict(torch.load('%s/%s_actor_vae_optimizer.pth' % (directory, filename), map_location=self.device))
        self.actor_vae_target.load_state_dict(torch.load('%s/%s_actor_vae_target.pth' % (directory, filename), map_location=self.device))
