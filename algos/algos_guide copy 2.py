"""
Based on https://github.com/sfujim/BCQ
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from networks.net import Actor, Critic, ActorVAE, Value
from collections import deque

class Latent(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, min_v, max_v, 
                 device, discount=0.99, tau=0.001, vae_lr=1e-4, actor_lr=1e-4, critic_lr=1e-4, 
                 max_latent_action=3, expectile=0.9, kl_beta=0.5, doubleq_min=0.8):
        super(Latent, self).__init__()

        self.device = torch.device(device)
        self.actor_vae = ActorVAE(state_dim, action_dim, latent_dim, max_latent_action, self.device).to(self.device)
        
        self.actor_vae_target = copy.deepcopy(self.actor_vae)
        self.actorvae_optimizer = torch.optim.Adam(self.actor_vae.parameters(), lr=vae_lr)

        self.actor = Actor(state_dim, latent_dim, max_latent_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=1e-5)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.vnet = Value(state_dim).to(self.device)

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
        self.q_buffer = deque(maxlen=2000)

        self.min_v, self.max_v = min_v, max_v 

    def copy_bn_param(self):
        source_state = self.actor_vae.state_dict()
        for name, module in self.actor_vae_target.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                module.running_mean.copy_(source_state[f'{name}.running_mean'])
                module.running_var.copy_(source_state[f'{name}.running_var'])

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            latent_a = self.actor(state)

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
        # latent_action = None
        
        actor_action = gen_net.decode(state, z=latent_action)
        target_q1, target_q2 = critic_net(state, actor_action)
        
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

    def train_step(self, batch, iter_id):
        state, action, next_state, reward, not_done = batch

        with torch.no_grad():
            next_target_v = self.get_pi_q(next_state, self.actor_target, self.critic_target, 
                                          self.actor_vae_target, use_noise=True)  
            ref_value_next = self.vnet(next_state)
            next_target_v = torch.max(next_target_v, ref_value_next)
            target_q = reward + not_done * self.discount * next_target_v.clamp(-np.inf, self.max_v)

        # Critic Training
        current_q1, current_q2 = self.critic(state, action)
        # current_v = self.critic.v(state)

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
                q_a = torch.max(q1_a, q2_a)
                # # q_a = (q1_a.clamp(-np.inf, self.max_v) + q2_a.clamp(-np.inf, self.max_v))/2
                # q_a = (q1_a + q2_a)/2
                # q_a = target_q
                # ref_value = reward + not_done * self.discount * self.vnet(next_state)
                # q_a = torch.max(q_a, ref_value)

                q_pi = self.get_pi_q(state, self.actor_target, self.critic_target, self.actor_vae_target,
                                     type='min', use_noise=False)
                # q_pi = current_v #self.critic.v(state)
                # q_pi = torch.max(q_pi, self.vnet(state))

                adv = q_a - q_pi
                # q_pi_abs = q_pi.abs().detach()
                threshold_reduce = 0 #-q_pi_abs*0.002
                weight = torch.where(adv < threshold_reduce, 1-self.expectile, self.expectile).detach()
                # threshold_out = -q_pi_abs*0.04
                # weight = torch.where(adv < threshold_out, 0.0, weight).detach()
                w_check = torch.where(adv < threshold_reduce, 0.0, 1.0).detach()

            # train weighted CVAE
            recons_action, mu, log_var = self.actor_vae(state, action)
            recons_loss_ori = F.mse_loss(recons_action, action, reduction='none')
            recon_loss = torch.sum(recons_loss_ori, 1).view(-1, 1)

            free_bits = 0.4
            kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # shape: [batch_size, latent_dim]

            free_bits_tensor = torch.tensor(free_bits, device=kl_per_dim.device)
            kl_freebits = torch.maximum(kl_per_dim, free_bits_tensor)
            kld_loss = kl_freebits.sum(dim=1).view(-1, 1)
            actor_vae_loss = (recon_loss + self.kl_beta * kld_loss)*weight.view(-1,1)
            # actor_vae_loss = recon_loss + self.kl_beta * kld_loss

            actor_vae_loss = actor_vae_loss.mean()
            self.actorvae_optimizer.zero_grad()
            actor_vae_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_vae.parameters(), max_norm=self.g_clip)
            self.actorvae_optimizer.step()
            
            # for param, target_param in zip(self.actor_vae.parameters(), self.actor_vae_target.parameters()):
            #     target_param.data.copy_(self.tau_vae * param.data + (1 - self.tau_vae) * target_param.data)

            # Update Target Networks
            loss_rc = (recons_loss_ori*weight.view(-1,1)).mean().item()
            loss_kl = (kld_loss*weight.view(-1,1)).mean().item()
            a_loss = q_pi.mean().item()

        if iter_id % 5 == 0:
            # train latent policy 
            latent_actor_action = self.actor(state)
            # latent_actor_action = latent_actor_action+ (torch.randn_like(latent_actor_action) * 0.1).clamp(-0.3, 0.3)

            actor_action = self.actor_vae.decode(state, z=latent_actor_action)
            # q1_pi = self.critic.q1(state, actor_action)
            # q_pi = q1_pi
            q1_pi, q2_pi = self.critic(state, actor_action)
            q_pi = torch.min(q1_pi, q2_pi)
            # q_pi = (q1_pi + q2_pi) / 2

            q_abs = q_pi.detach().abs().mean().item()
            self.q_buffer.append(q_abs)

            dist_z = torch.mean(latent_actor_action ** 2)
            # dist_norm = dist_z / (self.max_latent_action ** 2 + 1e-6)

            q_scale = np.mean(self.q_buffer)
            lambda_reg = max(0.02 * q_scale, 0.3)
            actor_loss = -q_pi.mean() + dist_z * 0.3 #lambda_reg

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.g_clip)
            self.actor_optimizer.step()
            a_loss = -actor_loss.item()

        # if iter_id % 5 == 0:
            for param, target_param in zip(self.actor_vae.parameters(), self.actor_vae_target.parameters()):
                target_param.data.copy_(self.tau_vae * param.data + (1 - self.tau_vae) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau_act * param.data + (1 - self.tau_act) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
        return w_check.sum().cpu().numpy(), a_loss, critic_loss.item(), loss_rc, loss_kl, adv.mean().item()
    
    def save(self, filename, directory):
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.critic_optimizer.state_dict(), '%s/%s_critic_optimizer.pth' % (directory, filename))
        torch.save(self.critic_target.state_dict(), '%s/%s_critic_target.pth' % (directory, filename))

        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, filename))

        torch.save(self.actor_vae.state_dict(), '%s/%s_actor_vae.pth' % (directory, filename))
        torch.save(self.actorvae_optimizer.state_dict(), '%s/%s_actor_vae_optimizer.pth' % (directory, filename))
        torch.save(self.actor_vae_target.state_dict(), '%s/%s_actor_vae_target.pth' % (directory, filename))

    def load(self, filename, directory):
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.critic_optimizer.load_state_dict(torch.load('%s/%s_critic_optimizer.pth' % (directory, filename)))
        self.critic_target.load_state_dict(torch.load('%s/%s_critic_target.pth' % (directory, filename)))

        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.actor_optimizer.load_state_dict(torch.load('%s/%s_actor_optimizer.pth' % (directory, filename)))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, filename)))

        self.actor_vae.load_state_dict(torch.load('%s/%s_actor_vae.pth' % (directory, filename)))
        self.actorvae_optimizer.load_state_dict(torch.load('%s/%s_actor_vae_optimizer.pth' % (directory, filename)))
        self.actor_vae_target.load_state_dict(torch.load('%s/%s_actor_vae_target.pth' % (directory, filename)))

    def load_reference(self, filename, directory):
        self.vnet.load_state_dict(torch.load('%s/%s_vnet.pth' % (directory, filename)))
        # self.vnet_optimizer.load_state_dict(torch.load('%s/%s_vnet_optimizer.pth' % (directory, filename)))
