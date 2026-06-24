"""
Based on https://github.com/sfujim/BCQ
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from networks.net_v2 import Critic, Value, ActorVAE
from collections import deque

class Latent(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, min_v, max_v, 
                 device, discount=0.99, tau=0.001, vae_lr=1e-4, actor_lr=1e-4, critic_lr=1e-4, 
                 max_latent_action=3, expectile=0.9, kl_beta=0.5, doubleq_min=0.8):
        super(Latent, self).__init__()

        self.device = torch.device(device)
        self.actor_vae = ActorVAE(state_dim, action_dim, latent_dim, max_latent_action, self.device).to(self.device)
        self.actorvae_optimizer = torch.optim.Adam(self.actor_vae.parameters(), lr=vae_lr, weight_decay=1e-9)

        self.qnet = Critic(state_dim, action_dim).to(self.device)
        self.qnet_target = copy.deepcopy(self.qnet)
        self.qnet_optimizer = torch.optim.Adam(self.qnet.parameters(), lr=critic_lr, weight_decay=1e-9)

        self.vnet = Value(state_dim).to(self.device)
        self.vnet_optimizer = torch.optim.Adam(self.vnet.parameters(), lr=critic_lr, weight_decay=1e-5)

        self.latent_dim = latent_dim
        self.max_latent_action = max_latent_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.tau_act = tau
        self.tau_vae = tau

        self.expectile = expectile
        self.kl_beta = kl_beta
        self.doubleq_min = doubleq_min

        self.g_clip = 0.5
        self.min_v, self.max_v = min_v, max_v 

    def select_action(self, state, need_q=True):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            # latent_a = self.actor(state)
            latent_a = None
            action = self.actor_vae.decode(state, z=latent_a)

            if need_q:
                q1, q2 = self.qnet(state, action)
                q1, q2 = q1.item(), q2.item()
            else:
                q1, q2 = None, None

        return action.cpu().data.numpy().flatten(), q1, q2

    def get_a_q(self, state, action):
        target_q1, target_q2 = self.qnet(state, action)
        target_q = torch.min(target_q1, target_q2)
        return target_q
        
    def get_pi_q(self, state, q_net, gen_net, type='mean', use_noise=True):
        # latent_action = actor_net(state)
        # if use_noise:
        #     latent_action += (torch.randn_like(latent_action) * 0.1).clamp(-0.3, 0.3)
        latent_action = None

        actor_action = gen_net.decode(state, z=latent_action)
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
        q = torch.min(q1, q2)*self.doubleq_min + torch.max(q1, q2)*(1-self.doubleq_min)
        return q

    def train_value(self, state_random_inr, state_expert_inr, state_expert_ine):
        with torch.no_grad():
            target_v_expert = self.get_pi_q(state_expert_ine, self.qnet, self.actor_vae, 
                                            type='mean', use_noise=True)
            # target_v_random = torch.zeros(len(state_random_inr), device=state_random_inr.device).view(-1, 1)
            # target_v_random = self.critic.v(state_random_inr) * 0.999

        # Critic Training
        current_v_expert = self.vnet(state_expert_inr)
        current_v_random = self.vnet(state_random_inr)

        v_loss_expert = F.mse_loss(current_v_expert, target_v_expert)
        v_loss_random = torch.mean(current_v_random.pow(2))

        v_loss = v_loss_expert + v_loss_random*0.001

        self.vnet_optimizer.zero_grad()
        v_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.g_clip)
        self.vnet_optimizer.step()
        return v_loss.item(), current_v_random.mean().item(), current_v_expert.mean().item()

    def train_q(self, batch, idx):
        state, action, next_state, reward, not_done = batch

        with torch.no_grad():
            next_target_v = self.qnet.v(next_state)
            # next_target_v = self.get_pi_q(next_state, self.qnet_target, self.actor_vae, 
                                        #   type="mean", use_noise=True)  
            target_q = (reward + not_done * self.discount * next_target_v).clamp(-np.inf, self.max_v)
            q1_a, q2_a = self.qnet_target(state, action)
            # target_v = torch.min(q1_a, q2_a)
            target_v = (q1_a + q2_a) / 2
        # Critic Training
        current_q1, current_q2 = self.qnet(state, action)
        current_v = self.qnet.v(state)

        critic_loss_1 = F.mse_loss(current_q1, target_q)
        critic_loss_2 = F.mse_loss(current_q2, target_q)
        v_loss = F.mse_loss(current_v, target_v)
        critic_loss = critic_loss_1 + critic_loss_2 + v_loss
    
        self.qnet_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=self.g_clip)
        self.qnet_optimizer.step()
        
        if idx % 5 == 0:
            for param, target_param in zip(self.qnet.parameters(), self.qnet_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return critic_loss.item(), current_q1.mean().item()

    def train_policy(self, state, action):
        # state, action, next_state, reward, not_done = batch
        
        # train weighted CVAE
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
        # torch.nn.utils.clip_grad_norm_(self.actor_vae.parameters(), max_norm=self.g_clip)
        self.actorvae_optimizer.step()
        
        return recon_loss.item(), kld_loss.item()
    
    def save(self, filename, directory):
        torch.save(self.qnet.state_dict(), '%s/%s_qnet.pth' % (directory, filename))
        torch.save(self.qnet_optimizer.state_dict(), '%s/%s_qnet_optimizer.pth' % (directory, filename))
        torch.save(self.qnet_target.state_dict(), '%s/%s_qnet_target.pth' % (directory, filename))

        torch.save(self.vnet.state_dict(), '%s/%s_vnet.pth' % (directory, filename))
        torch.save(self.vnet_optimizer.state_dict(), '%s/%s_vnet_optimizer.pth' % (directory, filename))

        # torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        # torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))
        # torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, filename))

        torch.save(self.actor_vae.state_dict(), '%s/%s_actor_vae.pth' % (directory, filename))
        torch.save(self.actorvae_optimizer.state_dict(), '%s/%s_actor_vae_optimizer.pth' % (directory, filename))
        # torch.save(self.actor_vae_target.state_dict(), '%s/%s_actor_vae_target.pth' % (directory, filename))

    def load(self, filename, directory):
        self.qnet.load_state_dict(torch.load('%s/%s_qnet.pth' % (directory, filename)))
        self.qnet_optimizer.load_state_dict(torch.load('%s/%s_qnet_optimizer.pth' % (directory, filename)))
        self.qnet_target.load_state_dict(torch.load('%s/%s_qnet_target.pth' % (directory, filename)))

        self.vnet.load_state_dict(torch.load('%s/%s_vnet.pth' % (directory, filename)))
        self.vnet_optimizer.load_state_dict(torch.load('%s/%s_vnet_optimizer.pth' % (directory, filename)))

        # self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        # self.actor_optimizer.load_state_dict(torch.load('%s/%s_actor_optimizer.pth' % (directory, filename)))
        # self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, filename)))

        self.actor_vae.load_state_dict(torch.load('%s/%s_actor_vae.pth' % (directory, filename)))
        self.actorvae_optimizer.load_state_dict(torch.load('%s/%s_actor_vae_optimizer.pth' % (directory, filename)))
        # self.actor_vae_target.load_state_dict(torch.load('%s/%s_actor_vae_target.pth' % (directory, filename)))