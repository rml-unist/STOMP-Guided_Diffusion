"""
Adapted from https://github.com/jannerm/diffuser
"""
import abc
import time
from collections import namedtuple
from copy import copy

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from abc import ABC

from torch.nn import DataParallel

from mpd.models.diffusion_models.helpers import cosine_beta_schedule, Losses, exponential_beta_schedule
from mpd.models.diffusion_models.sample_functions import extract, apply_hard_conditioning, mean_ddpm_sample_fn, \
                                                            noise_ddpm_sample_fn, noise_ddim_sample_fn, resample_model_mean
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import to_numpy


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


def build_context(model, dataset, input_dict):
    # input_dict is already normalized
    context = None
    if model.context_model is not None:
        context = dict()
        # (normalized) features of variable environments
        if dataset.variable_environment:
            env_normalized = input_dict[f'{dataset.field_key_env}_normalized']
            context['env'] = env_normalized

        # tasks
        task_normalized = input_dict[f'{dataset.field_key_task}_normalized']
        context['tasks'] = task_normalized
    return context


class GaussianDiffusionModel(nn.Module, ABC):

    def __init__(self,
                 model=None,
                 variance_schedule='exponential',
                 n_diffusion_steps=100,
                 clip_denoised=True,
                 predict_epsilon=False,
                 loss_type='l2',
                 context_model=None,
                 **kwargs):
        super().__init__()

        self.model = model

        self.context_model = context_model

        self.n_diffusion_steps = n_diffusion_steps

        self.state_dim = self.model.state_dim

        if variance_schedule == 'cosine':
            betas = cosine_beta_schedule(n_diffusion_steps, s=0.008, a_min=0, a_max=0.999)
        elif variance_schedule == 'exponential':
            betas = exponential_beta_schedule(n_diffusion_steps, beta_start=1e-4, beta_end=1.0)
        else:
            raise NotImplementedError

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        
        ## get loss coefficients and initialize objective
        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#
    def predict_noise_from_start(self, x_t, t, x0):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return x0
        else:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
            ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, hard_conds, context, t, noise=None, return_recon=False):
        if context is not None:
            context = self.context_model(context)

        if noise is None:
            noise = self.model(x, t, context)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        if return_recon:
            return x_recon

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample_loop(self, shape, hard_conds, context=None, return_chain=False,
                      sample_ver='mean_ddpm',
                      n_diffusion_steps_without_noise=0,
                      x0=True,
                      structured_noise=True,
                      init_structured_noise=False,
                      sampler=None,
                      recurrencing=False,
                      step_size=0.5,
                      negative_step_size=0.5,
                      **sample_kwargs):
        
        device = self.betas.device
        if sample_ver=='mean_ddpm':
            sample_fn=mean_ddpm_sample_fn
        elif sample_ver=='noise_ddpm':
            sample_fn=noise_ddpm_sample_fn
        else:
            raise NotImplementedError

        batch_size = shape[0]
        if init_structured_noise:
            x = sampler.sample((shape[0],)).reshape(shape)
        else:
            x = torch.randn(shape, device=device)
        # No hard conditioning

        chain = [] if return_chain else None

        for i in reversed(range(-n_diffusion_steps_without_noise, self.n_diffusion_steps)):
            if i >= 0:
                t = make_timesteps(batch_size, i, device)
            else:
                t = torch.zeros_like(t)

            # To guide nose
            sqrt_one_minus_alpha_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            x, values = sample_fn(self, x, hard_conds, context, t, 
                                  noise_scale=sqrt_one_minus_alpha_cumprod, 
                                  x0=x0,
                                  sampler=sampler,
                                  structured_noise=structured_noise, # True : SGDcosine
                                  step_size=step_size,
                                  negative_step_size=negative_step_size,
                                  **sample_kwargs)
            
            if recurrencing and 0 >= i > -n_diffusion_steps_without_noise:
                betas = extract(self.betas, t, x.shape)
                x = torch.sqrt(1-betas)*x + torch.sqrt(betas)*torch.rand_like(x)
            x = apply_hard_conditioning(x, hard_conds)

            if return_chain:
                if x0:
                    values = apply_hard_conditioning(values, hard_conds)
                    chain.append(values)
                else:
                    chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)
            return x, chain

        return x

    @torch.no_grad()
    def ddim_sample(
        self, shape, hard_conds, 
        context=None, return_chain=False,
        recurrencing = False,
        n_diffusion_steps_without_noise=0,
        init_structured_noise=False,
        structured_noise=False,
        sampler=None,
        x0=True,
        time_jump=2,
        **sample_kwargs
    ):
    
        device = self.betas.device
        batch_size = shape[0]

        # times = torch.tensor([0]*n_diffusion_steps_without_noise + [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24], device=device)
        # times = torch.tensor(list(range(self.n_diffusion_steps-1, -1, -time_jump)) + [0]*(n_diffusion_steps_without_noise), device=device)
        times = torch.tensor([24, 20, 18, 17, 16, 15, 14, 12, 10, 6, 3] + [0]*n_diffusion_steps_without_noise, device=device)
        # times = torch.tensor([24, 20, 18, 16, 14, 10, 5,] + [0]*n_diffusion_steps_without_noise, device=device)
        time_pairs = list(zip(times[:-1], times[1:]))
        total_step_num = len(times)

        if init_structured_noise:
            x = sampler.sample((shape[0],)).reshape(shape)
        else:
            x = torch.randn(shape, device=device)
        chain = [x] if return_chain else None

        for i, (time, time_next) in enumerate(time_pairs):
            t = make_timesteps(batch_size, time, device)
            t_next = make_timesteps(batch_size, time_next, device)

            alpha_next = extract(self.alphas_cumprod, t_next, x.shape)
            sqrt_one_minus_alpha_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

            x, values = noise_ddim_sample_fn(self, x, hard_conds, context, t, alpha_next,
                                noise_scale=sqrt_one_minus_alpha_cumprod,
                                x0=x0,
                                sampler=sampler,
                                structured_noise=structured_noise,
                                eta=1.0, # DDIM
                                **sample_kwargs)
            
            if recurrencing and time < 2:
                betas = extract(self.betas, t, x.shape)
                x = torch.sqrt(1-betas)*x + torch.sqrt(betas)*torch.rand_like(x)

            x = apply_hard_conditioning(x, hard_conds)

            if return_chain:
                if x0:
                    values = apply_hard_conditioning(values, hard_conds)
                    chain.append(values)
                else:
                    chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)
            return x, chain

        return x
    
    @torch.no_grad()
    def re_ddpm_sample_loop(
        self, shape, hard_conds, 
        context=None, return_chain=False,
        n_diffusion_steps_without_noise=0,
        init_structured_noise=False,
        structured_noise=False,
        recurrencing=False,
        sampler=None,
        x0=True,
        **sample_kwargs
    ):
        device = self.betas.device
        batch_size = shape[0]

        times = torch.tensor([0]*n_diffusion_steps_without_noise + list(range(0, self.n_diffusion_steps, 1)), device=device)
        next_times = torch.tensor([0]*(n_diffusion_steps_without_noise + 1) + list(range(0, self.n_diffusion_steps-2, 1)), device=device)
        times = list(reversed(times.int().tolist()))
        next_times = list(reversed(next_times.int().tolist()))
        time_pairs = list(zip(times, next_times))  # [(T-1, T-3), (T-2, T-4), ..., (1, 0), (0, 0), ..., (0, 0)]

        if init_structured_noise:
            x = sampler.sample((shape[0],)).reshape(shape)
        else:
            x = torch.randn(shape, device=device)

        chain = [x] if return_chain else None
        for time, time_next in time_pairs:
            direct_forward = True
            eta = 1

            t = make_timesteps(batch_size, time, device)
            t_next = make_timesteps(batch_size, time_next, device)

            alpha_next = extract(self.alphas_cumprod, t_next, x.shape)
            sqrt_one_minus_alpha_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

            x, values = noise_ddim_sample_fn(self, x, hard_conds, context, t, alpha_next,
                                  noise_scale=sqrt_one_minus_alpha_cumprod, 
                                  x0=x0,
                                  sampler=sampler,
                                  structured_noise=structured_noise,
                                  eta=eta,
                                  direct_forward=direct_forward,
                                  **sample_kwargs)
            
            # recurrencing : go back in one time step.
            if time >= 2: # edge cases
                betas = extract(self.betas, t_next, x.shape)
                x = torch.sqrt(1-betas)*x + torch.sqrt(betas)*torch.rand_like(x)

            if recurrencing and time < 2:
                betas = extract(self.betas, t, x.shape)
                x = torch.sqrt(1-betas)*x + torch.sqrt(betas)*torch.rand_like(x)

            x = apply_hard_conditioning(x, hard_conds)

            if return_chain:
                if x0:
                    values = apply_hard_conditioning(values, hard_conds)
                    chain.append(values)
                else:
                    chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)
            return x, chain

        return x

    @torch.no_grad()
    def diffusion_es_sample_loop(
        self, shape, hard_conds, 
        context=None, return_chain=False,
        recurrencing = False,
        n_diffusion_steps_without_noise=0,
        structured_noise=False,
        sampler=None,
        x0=True,
        time_jump=1,
        loop_num=20,
        cost=None,
        **sample_kwargs
    ):
        assert cost is not None # check resample=True
        # es doesn't utilize guidance
        sample_kwargs.update({'gradient_free_guide_ver': None})
        sample_kwargs.update({'guide': None})

        device = self.betas.device
        batch_size = shape[0]

        last_time = make_timesteps(batch_size, torch.tensor(self.n_diffusion_steps-1, device=device), device)
        times = torch.tensor(list(range(self.n_diffusion_steps-1, -1, -time_jump)) + [0]*n_diffusion_steps_without_noise, device=device)
        time_pairs = list(zip(times[:-1], times[1:]))

        if structured_noise:
            x = sampler.sample((shape[0],)).reshape(shape)
        else:
            x = torch.randn(shape, device=device)
        chain = [x] if return_chain else None

        alpha_last = extract(self.alphas_cumprod, last_time, x.shape)
        sqrt_one_minus_alpha_cumprod_last = extract(self.sqrt_one_minus_alphas_cumprod, last_time, x.shape)

        for k in range(loop_num):
            for i, times in enumerate(time_pairs):
                time, time_next = times
                t = make_timesteps(batch_size, time, device)
                t_next = make_timesteps(batch_size, time_next, device)

                alpha_next = extract(self.alphas_cumprod, t_next, x.shape)
                sqrt_one_minus_alpha_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

                x, values = noise_ddim_sample_fn(self, x, hard_conds, context, t, alpha_next,
                                    noise_scale=sqrt_one_minus_alpha_cumprod, 
                                    x0=x0,
                                    sampler=sampler,
                                    structured_noise=False,
                                    eta = 1 * (1-k/loop_num), # DDIM
                                    **sample_kwargs)
                if recurrencing and time < 2:
                    betas = extract(self.betas, t, x.shape)
                    x = torch.sqrt(1-betas)*x + torch.sqrt(betas)*torch.rand_like(x)

                x = apply_hard_conditioning(x, hard_conds)

                if return_chain:
                    if x0:
                        values = apply_hard_conditioning(values, hard_conds)
                        if i == len(time_pairs) - 1:
                            chain.append(values)
                    else:
                        if i == len(time_pairs) - 1:
                            chain.append(x)
                
            # get x cost and sample
            noise = torch.randn(shape, device=device)
            x = resample_model_mean(cost, x, x, temperature=10)
            x = torch.sqrt(alpha_last) * x + sqrt_one_minus_alpha_cumprod_last * noise

            # last_time = make_timesteps(batch_size, torch.tensor(self.n_diffusion_steps * (1 - k/self.n_diffusion_steps)-1, device=device), device)
            # times = torch.tensor(list(range(self.n_diffusion_steps-1, -1, -time_jump)) + [0]*n_diffusion_steps_without_noise, device=device)
            # time_pairs = list(zip(times[:-1], times[1:]))

            # alpha_last = extract(self.alphas_cumprod, last_time, x.shape)
            # sqrt_one_minus_alpha_cumprod_last = extract(self.sqrt_one_minus_alphas_cumprod, last_time, x.shape)

        if return_chain:
            chain = torch.stack(chain, dim=1)
            return x, chain

        return x
    
    @torch.no_grad()
    def diffusion_es_sample_loop_ver2(
        self, shape, hard_conds, 
        context=None, return_chain=False,
        recurrencing = False,
        n_diffusion_steps_without_noise=0,
        structured_noise=False,
        sampler=None,
        x0=True,
        time_jump=1,
        loop_num=20,
        cost=None,
        **sample_kwargs
    ):
        # This function tries to follow author code at most.

        assert cost is not None # check resample=True
        # es doesn't utilize guidance
        sample_kwargs.update({'gradient_free_guide_ver': None})
        sample_kwargs.update({'guide': None})
        sample_kwargs.update({'sample_ver':'noise_ddpm'})

        device = self.betas.device
        batch_size = shape[0]

        x, init_chain = self.p_sample_loop(shape, hard_conds, context=context, return_chain=return_chain,
                      n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,
                      x0=x0,
                      structured_noise=False, 
                      sampler=None,
                      recurrencing=False,
                      **sample_kwargs)
        
        chain = [init_chain] if return_chain else None
        #mutate
        max_mutate_timestep = 5
        noise = torch.randn(shape, device=device)
        mutate_time = make_timesteps(batch_size, torch.tensor(max_mutate_timestep-1, device=device), device)
        alpha_mutate = extract(self.alphas_cumprod, mutate_time, x.shape)
        sqrt_one_minus_alpha_cumprod_mutate = extract(self.sqrt_one_minus_alphas_cumprod, mutate_time, x.shape)
        x = torch.sqrt(alpha_mutate) * x + sqrt_one_minus_alpha_cumprod_mutate * noise    

        sample_kwargs.update({'noise_std_extra_schedule_fn':lambda x: 0.0})
        for k in range(loop_num):
            mutate_timestep = max(int(max_mutate_timestep * (1 - k/loop_num)), 1)
            timestep_list = list(reversed(range(mutate_timestep)))
            for tstep, i in enumerate(timestep_list):
                if i >= 0:
                    t = make_timesteps(batch_size, i, device)
                else:
                    t = torch.zeros_like(t)

                # To guide nose
                sqrt_one_minus_alpha_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
                x, values = noise_ddpm_sample_fn(self, x, hard_conds, context, t, 
                                                noise_scale=sqrt_one_minus_alpha_cumprod, 
                                                x0=x0,
                                                sampler=sampler,
                                                structured_noise=False,
                                                **sample_kwargs)
                
                if recurrencing and 0 >= i > -n_diffusion_steps_without_noise:
                    betas = extract(self.betas, t, x.shape)
                    x = torch.sqrt(1-betas)*x + torch.sqrt(betas)*torch.rand_like(x)
                x = apply_hard_conditioning(x, hard_conds)

                if return_chain:
                    if x0:
                        if tstep == len(timestep_list) - 1:
                            values = apply_hard_conditioning(values, hard_conds)
                            chain.append(values.unsqueeze(1))
                    else:
                        chain.append(x.unsqueeze(1))

            noise = torch.randn(shape, device=device)
            mutate_time = make_timesteps(batch_size, torch.tensor(mutate_timestep-1, device=device), device)
            alpha_mutate = extract(self.alphas_cumprod, mutate_time, x.shape)
            sqrt_one_minus_alpha_cumprod_mutate = extract(self.sqrt_one_minus_alphas_cumprod, mutate_time, x.shape)

            x = resample_model_mean(cost, x, x, temperature=10)
            x = torch.sqrt(alpha_mutate) * x + sqrt_one_minus_alpha_cumprod_mutate * noise    

        if return_chain:
            chain = torch.concat(chain, dim=1)
            return x, chain
        
        return x
        
    @torch.no_grad()
    def conditional_sample(self, hard_conds, horizon=None, batch_size=1, time_sample_ver='ddpm', **sample_kwargs):
        '''
            hard conditions : hard_conds : { (time, state), ... }
        '''
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.state_dim)

        if time_sample_ver == 'ddim':
            return self.ddim_sample(shape, hard_conds, **sample_kwargs)
        
        elif time_sample_ver == 'ddpm_recurrencing':
            return self.re_ddpm_sample_loop(shape, hard_conds, **sample_kwargs)
        
        elif time_sample_ver == 'ddpm':
            return self.p_sample_loop(shape, hard_conds, **sample_kwargs)

        elif time_sample_ver == 'diffusion-es':
            return self.diffusion_es_sample_loop(shape, hard_conds, **sample_kwargs)
            # return self.diffusion_es_sample_loop_ver2(shape, hard_conds, **sample_kwargs)
        
        else:
            raise NotImplementedError

    def forward(self, cond, *args, **kwargs):
        raise NotImplementedError
        return self.conditional_sample(cond, *args, **kwargs)

    @torch.no_grad()
    def warmup(self, horizon=64, device='cuda'):
        shape = (2, horizon, self.state_dim)
        x = torch.randn(shape, device=device)
        t = make_timesteps(2, 1, device)
        self.model(x, t, context=None)

    @torch.no_grad()
    def run_inference(self, context=None, hard_conds=None, n_samples=1, return_chain=False, **diffusion_kwargs):
        # context and hard_conds must be normalized
        hard_conds = copy(hard_conds)
        context = copy(context)

        # repeat hard conditions and contexts for n_samples
        for k, v in hard_conds.items():
            new_state = einops.repeat(v, 'd -> b d', b=n_samples)
            hard_conds[k] = new_state

        if context is not None:
            for k, v in context.items():
                context[k] = einops.repeat(v, 'd -> b d', b=n_samples)

        # Sample from diffusion model
        samples, chain = self.conditional_sample(
            hard_conds, context=context, batch_size=n_samples, return_chain=True, **diffusion_kwargs
        )

        # chain: [ n_samples x (n_diffusion_steps + 1) x horizon x (state_dim)]
        # extract normalized trajectories
        trajs_chain_normalized = chain

        # trajs: [ (n_diffusion_steps + 1) x n_samples x horizon x state_dim ]
        trajs_chain_normalized = einops.rearrange(trajs_chain_normalized, 'b diffsteps h d -> diffsteps b h d')

        if return_chain:
            return trajs_chain_normalized

        # return the last denoising step
        return trajs_chain_normalized[-1]

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, context, t, hard_conds):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_hard_conditioning(x_noisy, hard_conds)

        # context model
        if context is not None:
            context = self.context_model(context)

        # diffusion model
        x_recon = self.model(x_noisy, t, context)
        x_recon = apply_hard_conditioning(x_recon, hard_conds)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, context, *args):
        batch_size = x.shape[0]
        t = torch.randint(0, self.n_diffusion_steps, (batch_size,), device=x.device).long()
        return self.p_losses(x, context, t, *args)

