"""
Based on https://github.com/sfujim/BCQ
"""
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.net_v2 import Actor, Critic, ActorVAE, OPEValue
from networks.obs_encoder import FiLMObsEncoder, build_obs_encoder


class FrozenPolicy(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        latent_dim,
        max_latent_action,
        device,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.actor_vae = ActorVAE(
            state_dim, action_dim, latent_dim, max_latent_action, self.device
        ).to(self.device)
        self.actor = None
        self.actor = Actor(state_dim, latent_dim, max_latent_action).to(self.device)

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

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.select_action_tensor(state)
        return action.cpu().data.numpy().flatten(), 0.0, 0.0


class Latent(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, min_v, max_v, vae,
                 device, discount=0.99, tau=0.005, vae_lr=2e-4, actor_lr=2e-4, critic_lr=2e-4, 
                 max_latent_action=0.675, expectile=0.9, kl_beta=1.0, doubleq_min=1.0,
                 target_policy_dir="", target_policy_name="model", target_policy_mode="lapo",
                 ope_ref_dir="", ope_ref_name="model", ope_ref_clip=None,
                 ope_ref_state_mean=None, ope_ref_state_std=None,
                 obs_encoder=None, obs_feature_dim=None, obs_encoder_lr=None,
                 obs_image_key="image", obs_proprio_key="proprio",
                 use_robomimic_obs_encoder=False, image_shape=(3, 224, 224),
                 robomimic_feature_dim=256, robomimic_crop_shape=None,
                 robomimic_backbone_class="ResNet18Conv",
                 robomimic_pool_class="SpatialSoftmax",
                 encoder_mode="robomimic",
                 concat_mode="default",
                 zipper_backbone="resnet18",
                 zipper_normalize_image=True):
        super(Latent, self).__init__()

        self.device = torch.device(device)
        self.vae = vae

        # print("Initializing Latent model with encoder_mode:", encoder_mode)
        if concat_mode == "default":
            obs_encoder = build_obs_encoder(
                encoder_mode=encoder_mode,
                image_shape=image_shape,
                proprio_dim=state_dim,
                image_key=obs_image_key,
                proprio_key=obs_proprio_key,
                robomimic_feature_dim=robomimic_feature_dim,
                robomimic_crop_shape=robomimic_crop_shape,
                robomimic_backbone_class=robomimic_backbone_class,
                robomimic_pool_class=robomimic_pool_class,
                zipper_backbone=zipper_backbone,
                zipper_normalize_image=zipper_normalize_image,
            )
        elif concat_mode == "film":
            obs_encoder = FiLMObsEncoder(
                encoder_mode=encoder_mode,
                image_shape=image_shape,
                proprio_dim=state_dim,
                image_key=obs_image_key,
                proprio_key=obs_proprio_key,
                feature_dim=robomimic_feature_dim,
                crop_shape=robomimic_crop_shape,
                backbone_class=robomimic_backbone_class,
                pool_class=robomimic_pool_class,
                zipper_backbone=zipper_backbone,
                zipper_normalize_image=zipper_normalize_image,
            )
        else:
            raise ValueError("concat_mode must be 'default' or 'film'.")
        
        obs_feature_dim = obs_encoder.output_dim
        if isinstance(obs_feature_dim, (tuple, list)):
            obs_feature_dim = int(np.prod(obs_feature_dim))

        self.obs_encoder = obs_encoder.to(self.device)
        self.obs_encoder_target = copy.deepcopy(self.obs_encoder)
        self.obs_image_key = obs_image_key
        self.obs_proprio_key = obs_proprio_key
        self.raw_state_dim = state_dim
        self.obs_encoder_optimizer = torch.optim.Adam(
            self.obs_encoder.parameters(),
            lr=obs_encoder_lr if obs_encoder_lr is not None else critic_lr,
            weight_decay=1e-5,
        )
        self.model_state_dim = obs_feature_dim

        self.critic = Critic(self.model_state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-5)

        self.state_dim = self.model_state_dim
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
        self.ope_ref_value = None
        self.ope_ref_clip = ope_ref_clip
        self.ope_ref_state_mean = None
        self.ope_ref_state_std = None
        if ope_ref_state_mean is not None and ope_ref_state_std is not None:
            self.ope_ref_state_mean = torch.as_tensor(
                ope_ref_state_mean, dtype=torch.float32, device=self.device
            )
            self.ope_ref_state_std = torch.as_tensor(
                ope_ref_state_std, dtype=torch.float32, device=self.device
            )

        if self.ope_mode:
            if target_policy_mode == "vae":
                target_policy_mode = "vae_bc"
            if target_policy_mode not in ("lapo", "vae_bc"):
                raise ValueError(f"Unsupported target_policy_mode: {target_policy_mode}")
            self.target_policy = FrozenPolicy(
                state_dim=self.model_state_dim,
                action_dim=action_dim,
                latent_dim=latent_dim,
                max_latent_action=max_latent_action,
                device=device,
                policy_mode="lapo" if target_policy_mode == "lapo" else "vae_bc",
            )
            self.target_policy.load(target_policy_name, target_policy_dir)
            self.value = OPEValue(self.model_state_dim).to(self.device)
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
            self.actor_vae = ActorVAE(self.model_state_dim, action_dim, latent_dim, max_latent_action, self.device).to(self.device)
            self.actor_vae_target = copy.deepcopy(self.actor_vae)
            self.actorvae_optimizer = torch.optim.Adam(self.actor_vae.parameters(), lr=vae_lr, weight_decay=1e-5)

            self.actor = Actor(self.model_state_dim, latent_dim, max_latent_action).to(self.device)
            self.actor_target = copy.deepcopy(self.actor)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=1e-5)

            if ope_ref_dir:
                self.load_ope_value(ope_ref_name, ope_ref_dir, ope_ref_clip)

    def copy_bn_param(self):
        if self.ope_mode:
            return None
        source_state = self.actor_vae.state_dict()
        for name, module in self.actor_vae_target.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                module.running_mean.copy_(source_state[f'{name}.running_mean'])
                module.running_var.copy_(source_state[f'{name}.running_var'])

    def _make_obs_dict(self, state, image=None, obs=None):
        if obs is not None:
            obs_dict = {}
            for key, value in obs.items():
                if not torch.is_tensor(value):
                    obs_dict[key] = value
                    continue
                value = value.to(self.device)
                obs_dict[key] = value if key == self.obs_image_key else value.float()
            return obs_dict

        obs_dict = {self.obs_proprio_key: state}
        if image is not None:
            obs_dict[self.obs_image_key] = image
        return obs_dict

    def _obs_feature(self, state, image=None, obs=None, target=False):
        encoder = self.obs_encoder_target if target else self.obs_encoder
        feature = encoder(self._make_obs_dict(state, image=image, obs=obs))
        if isinstance(feature, dict):
            if "obs" in feature:
                feature = feature["obs"]
            elif "feature" in feature:
                feature = feature["feature"]
            else:
                raise KeyError("obs_encoder returned a dict without 'obs' or 'feature'.")
        if isinstance(feature, (tuple, list)):
            feature = feature[0]
        return feature

    def _batch_feature(self, batch, next_obs=False, target=False):
        state_key = "next_state" if next_obs else "state"
        image_key = "next_image" if next_obs else "image"
        obs_key = "next_obs" if next_obs else "obs"

        if image_key in batch and state_key in batch:
            state = batch[state_key].to(self.device).float()
            image = batch[image_key].to(self.device)
            return self._obs_feature(state, image=image, target=target)

        obs = batch.get(obs_key, None)
        if obs is not None:
            return self._obs_feature(None, obs=obs, target=target)

        raise KeyError(f"Batch is missing '{state_key}/{image_key}' or '{obs_key}'.")

    def select_action(self, state, image=None, obs=None):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            if image is not None:
                image = torch.as_tensor(image, dtype=torch.float32, device=self.device)
                if image.ndim == 3:
                    image = image.unsqueeze(0)
                if image.shape[-1] == 3:
                    image = image.permute(0, 3, 1, 2)
                if image.max() > 1:
                    image = image / 255.0
            state = self._obs_feature(state, image=image, obs=obs)
            
            if self.vae:
                latent_a = None
            else:
                latent_a = self.actor(state)
            # print("latent_a", latent_a)
            
            if self.ope_mode:
                action = self.target_policy.select_action_tensor(state)
            else:
                action = self.actor_vae.decode(state, z=latent_a)
            q1, q2 = self.critic(state, action)
            # v = self.critic.v(state)
            
        return action.cpu().data.numpy().flatten(), q1.item(), q2.item()

    def kl_loss(self, mu, log_var):
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).view(-1, 1)
        return kld_loss

    def get_pi_q(self, state, actor_net, critic_net, gen_net, type='none', use_noise=True):

        if self.vae:
            latent_action = None
            # print("Using VAE with no latent action.")
        else:
            latent_action = actor_net(state)
            # print("actor_net output (latent_action)", latent_action)
        # print("latent_action", latent_action)
        if use_noise and latent_action is not None:
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

    def load_ope_value(self, filename, directory, clip_value=None):
        if self.ope_mode:
            raise RuntimeError("Reference OPE value is only used in offline RL mode.")

        self.ope_ref_value = OPEValue(self.state_dim).to(self.device)
        self.ope_ref_value.load_state_dict(
            torch.load(f"{directory}/{filename}_value.pth", map_location=self.device)
        )
        self.ope_ref_value.eval()
        for param in self.ope_ref_value.parameters():
            param.requires_grad_(False)
        self.ope_ref_clip = clip_value
        return self.ope_ref_value

    def normalize_ope_ref_state(self, raw_state):
        if self.ope_ref_state_mean is None or self.ope_ref_state_std is None:
            return raw_state
        return (raw_state - self.ope_ref_state_mean) / (self.ope_ref_state_std + 0.000001)

    def train_step(self, batch, iter_id, ):
        action = batch['action'].to(self.device).float()
        reward = batch['reward'].to(self.device).view(-1,1).float()
        not_done = batch['not_done'].to(self.device).view(-1,1)

        if self.ope_mode:
            with torch.no_grad():
                next_target_state = self._batch_feature(batch, next_obs=True, target=True)
                target_q_state = self._batch_feature(batch)
                next_target_v = self.value_target(next_target_state)
                target_q = reward + not_done * self.discount * next_target_v.clamp(
                    self.min_v, self.max_v
                )

            current_state = self._batch_feature(batch)
            current_q1, current_q2 = self.critic(current_state, action)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            self.obs_encoder_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.g_clip)
            torch.nn.utils.clip_grad_norm_(self.obs_encoder.parameters(), max_norm=self.g_clip)
            self.critic_optimizer.step()
            self.obs_encoder_optimizer.step()

            value_state = self._batch_feature(batch)
            policy_action = self.target_policy.select_action_tensor(target_q_state)
            with torch.no_grad():
                q1_pi, q2_pi = self.critic(target_q_state, policy_action)
                q_pi = self.get_min_q(q1_pi, q2_pi)

            current_v = self.value(value_state)
            value_loss = F.mse_loss(current_v, q_pi)

            self.value_optimizer.zero_grad()
            self.obs_encoder_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=self.g_clip)
            torch.nn.utils.clip_grad_norm_(self.obs_encoder.parameters(), max_norm=self.g_clip)
            self.value_optimizer.step()
            self.obs_encoder_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.value.parameters(), self.value_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.obs_encoder.parameters(), self.obs_encoder_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            sample_w = np.ones(action.shape[0], dtype=np.float32)
            return sample_w, current_v.mean().item(), critic_loss.item(), value_loss.item(), None

        with torch.no_grad():
            target_next_state = self._batch_feature(batch, next_obs=True, target=True)
            next_target_v = self.get_pi_q(target_next_state, self.actor_target, self.critic_target, 
                                          self.actor_vae_target, use_noise=True)     
            next_backup_v = next_target_v
            if self.ope_ref_value is not None:
                if self.ope_ref_state_mean is not None and 'raw_next_state' in batch:
                    next_ref_state = self.normalize_ope_ref_state(
                        batch['raw_next_state'].to(self.device).float()
                    )
                else:
                    next_ref_state = self._batch_feature(batch, next_obs=True, target=True)
                next_ref_v = self.ope_ref_value(next_ref_state)
                if self.ope_ref_clip is not None and self.ope_ref_clip > 0:
                    next_ref_v = next_ref_v.clamp(0, self.ope_ref_clip)
                next_backup_v = torch.max(next_backup_v, next_ref_v)

            # Reference-anchored Bellman backup:
            # y = r_task + gamma * max(Q_targ(s', pi(s')), bounded V_ref(s'))
            target_q = reward + not_done * self.discount * next_backup_v.clamp(self.min_v, self.max_v)

        # Critic Training
        current_state = self._batch_feature(batch)
        current_q1, current_q2 = self.critic(current_state, action)
        critic_loss_1 = F.mse_loss(current_q1, target_q)
        critic_loss_2 = F.mse_loss(current_q2, target_q)
        critic_loss = critic_loss_1 + critic_loss_2
    
        self.critic_optimizer.zero_grad()
        self.obs_encoder_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.g_clip)
        torch.nn.utils.clip_grad_norm_(self.obs_encoder.parameters(), max_norm=self.g_clip)
        self.critic_optimizer.step()
        self.obs_encoder_optimizer.step()
        
        loss_rc, loss_kl, a_loss, loss_std, loss_mean= None, None, None, None, None

        if iter_id % 1 == 0:
            with torch.no_grad():
                eval_state = self._batch_feature(batch)
                q1_a, q2_a = self.critic(eval_state, action)
                # q_a = torch.min(q1_a, q2_a)
                q_a = (q1_a + q2_a)/2
                q_pi = self.get_pi_q(eval_state, self.actor, self.critic, self.actor_vae,
                                     type='min', use_noise=False)
                adv = q_a - q_pi
                weight = torch.where(adv < 0, 1-self.expectile, self.expectile).detach()
            
            # train weighted CVAE
            vae_state = self._batch_feature(batch)
            recons_action, mu, log_var = self.actor_vae(vae_state, action)
            recons_loss_ori = F.mse_loss(recons_action, action, reduction='none')
            recon_loss = torch.sum(recons_loss_ori, 1).view(-1, 1)

            free_bits = 0.5
            kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # shape: [batch_size, latent_dim]

            free_bits_tensor = torch.tensor(free_bits, device=kl_per_dim.device)
            kl_freebits = torch.maximum(kl_per_dim, free_bits_tensor)
            kld_loss = kl_freebits.sum(dim=1).view(-1, 1)
            # actor_vae_loss = recon_loss + self.kl_beta * kld_loss
            actor_vae_loss = (recon_loss + self.kl_beta * kld_loss)*weight.view(-1,1)

            actor_vae_loss = actor_vae_loss.mean()
            self.actorvae_optimizer.zero_grad()
            self.obs_encoder_optimizer.zero_grad()
            actor_vae_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_vae.parameters(), max_norm=self.g_clip)
            torch.nn.utils.clip_grad_norm_(self.obs_encoder.parameters(), max_norm=self.g_clip)
            self.actorvae_optimizer.step()
            self.obs_encoder_optimizer.step()

            # Update Target Networks
            # loss_rc = recons_loss_ori.mean().item()
            # loss_kl = kld_loss.mean().item()
            # a_loss = q_pi.mean().item()

            loss_rc = (recons_loss_ori*weight.view(-1,1)).mean().item()
            loss_kl = (kld_loss*weight.view(-1,1)).mean().item()
            a_loss = q_pi.mean().item()

        if iter_id % 2 == 0:
            # train latent policy 
            actor_state = self._batch_feature(batch)
            latent_actor_action = self.actor(actor_state)
            latent_actor_action = latent_actor_action

            actor_action = self.actor_vae.decode(actor_state, z=latent_actor_action)
            q1_pi, q2_pi = self.critic(actor_state, actor_action)
            q_pi = torch.min(q1_pi, q2_pi)

            actor_qloss = -q_pi.mean()
            actor_reg_loss = torch.mean(latent_actor_action ** 2)
            actor_loss = actor_qloss + actor_reg_loss

            self.actor_optimizer.zero_grad()
            self.obs_encoder_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.g_clip)
            torch.nn.utils.clip_grad_norm_(self.obs_encoder.parameters(), max_norm=self.g_clip)
            self.actor_optimizer.step()
            self.obs_encoder_optimizer.step()
            a_loss = -actor_loss.item()

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau_act * param.data + (1 - self.tau_act) * target_param.data)
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor_vae.parameters(), self.actor_vae_target.parameters()):
            target_param.data.copy_(self.tau_vae * param.data + (1 - self.tau_vae) * target_param.data)
        for param, target_param in zip(self.obs_encoder.parameters(), self.obs_encoder_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return weight.cpu().numpy(), a_loss, critic_loss.item(), loss_rc, loss_kl

    def compute_loss(self, batch, iter_id=0):
        return self.train_step(batch, iter_id)
    
    def save(self, filename, directory, is_best=False, best_metric=None, best_epoch=None):
        torch.save(self.obs_encoder.state_dict(), '%s/%s_obs_encoder.pth' % (directory, filename))
        torch.save(self.obs_encoder_optimizer.state_dict(), '%s/%s_obs_encoder_optimizer.pth' % (directory, filename))
        torch.save(self.obs_encoder_target.state_dict(), '%s/%s_obs_encoder_target.pth' % (directory, filename))

        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.critic_optimizer.state_dict(), '%s/%s_critic_optimizer.pth' % (directory, filename))
        torch.save(self.critic_target.state_dict(), '%s/%s_critic_target.pth' % (directory, filename))

        if self.ope_mode:
            torch.save(self.value.state_dict(), '%s/%s_value.pth' % (directory, filename))
            torch.save(self.value_optimizer.state_dict(), '%s/%s_value_optimizer.pth' % (directory, filename))
            torch.save(self.value_target.state_dict(), '%s/%s_value_target.pth' % (directory, filename))
            if is_best:
                best_filename = "best_%s" % filename
                self.save(best_filename, directory, is_best=False)
                best_meta = {
                    "filename": best_filename,
                    "metric": best_metric,
                    "epoch": best_epoch,
                }
                with open(os.path.join(directory, "%s_best_meta.json" % filename), "w") as meta_file:
                    json.dump(best_meta, meta_file, indent=2)
            return

        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, filename))

        torch.save(self.actor_vae.state_dict(), '%s/%s_actor_vae.pth' % (directory, filename))
        torch.save(self.actorvae_optimizer.state_dict(), '%s/%s_actor_vae_optimizer.pth' % (directory, filename))
        torch.save(self.actor_vae_target.state_dict(), '%s/%s_actor_vae_target.pth' % (directory, filename))

        if is_best:
            best_filename = "best_%s" % filename
            self.save(best_filename, directory, is_best=False)
            best_meta = {
                "filename": best_filename,
                "metric": best_metric,
                "epoch": best_epoch,
            }
            with open(os.path.join(directory, "%s_best_meta.json" % filename), "w") as meta_file:
                json.dump(best_meta, meta_file, indent=2)

    def load(self, filename, directory, load_best=False):
        if load_best:
            filename = "best_%s" % filename

        self.obs_encoder.load_state_dict(torch.load('%s/%s_obs_encoder.pth' % (directory, filename), map_location=self.device))
        self.obs_encoder_optimizer.load_state_dict(torch.load('%s/%s_obs_encoder_optimizer.pth' % (directory, filename), map_location=self.device))
        self.obs_encoder_target.load_state_dict(torch.load('%s/%s_obs_encoder_target.pth' % (directory, filename), map_location=self.device))

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
