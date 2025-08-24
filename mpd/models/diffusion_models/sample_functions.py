import torch
import math
import time

def apply_hard_conditioning(x, conditions):
    for t, val in conditions.items():
        x[:, t, :] = val.clone()
    return x


def extract(a, t, x_shape):
    
    # Gather elements directly from 1D tensor `a` using `t`
    out = a.gather(0, t)  # Gather based on indices in `t`
    if len(x_shape) == 3:
        # For 3D case
        b = x_shape[0]
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    elif len(x_shape) == 4:
        # For 4D case
        c, b = x_shape[:2]
        return out.view(1, b).expand(c, b).reshape(c, b, 1, 1)

@torch.no_grad()
def mean_ddpm_sample_fn(
        model, x, hard_conds, context, t,
        guide=None,
        n_guide_steps=1,
        scale_grad_by_std=True,
        t_start_guide=torch.inf,
        noise_std_extra_schedule_fn=None,  # 'linear'
        x0=True,
        latent_guide=True,
        sample_num=10,
        debug=False,
        **kwargs
):

    t_single = t[0]
    if t_single < 0:
        t = torch.zeros_like(t)

    pred_noise = model.model(x, t, context)
    model_mean, _, _, x_recon = model.p_mean_variance(x=x, noise=pred_noise, hard_conds=hard_conds, context=context, t=t) 

    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    if guide.__class__.__name__ == 'GuideManagerTrajectoriesWithSTOMP':
        # Get shapes and prepare variables for multi-sampling
        batch, traj_len, state_dim = pred_noise.shape
        multi_shape = (sample_num * batch, traj_len, state_dim)
        
        # Repeat tensors once, instead of multiple operations
        multi_t = t.unsqueeze(0).repeat(sample_num, 1).view(-1)
        multi_x = x.unsqueeze(0).expand(sample_num, -1, -1, -1).reshape(multi_shape)
        multi_pred_noise = pred_noise.unsqueeze(0).expand(sample_num, -1, -1, -1).reshape(multi_shape)
        
        # Efficient sampling direction
        # multi_std = model_std.unsqueeze(0).expand(sample_num, -1, -1, -1).reshape(multi_shape[0], 1, 1)
        sampled_direction = torch.randn_like(multi_pred_noise)
        multi_pred_noise += sampled_direction

        # Efficient computation of p_mean_variance
        multi_model_mean, _, _, multi_x_recon = model.p_mean_variance(
            x=multi_x, noise=multi_pred_noise, 
            hard_conds=hard_conds, context=context, t=multi_t
        )
        
        # Conditional assignment of guider to avoid redundant evaluation
        guider = multi_model_mean if latent_guide else multi_x_recon
    else:
        guider = model_mean if latent_guide else x_recon
        sampled_direction = None
    x = model_mean

    if guide is not None and t_single < t_start_guide:
        x = mean_guide_gradient_steps(
            guider, x,
            sampled_direction=sampled_direction,
            sample_num=sample_num,
            hard_conds=hard_conds,
            guide=guide,
            n_guide_steps=n_guide_steps,
            scale_grad_by_std=scale_grad_by_std,
            model_var=model_var,
            debug=False,
            **kwargs
        )

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    # For smoother results, we can decay the noise standard deviation throughout the diffusion
    # this is roughly equivalent to using a temperature in the prior distribution
    if noise_std_extra_schedule_fn is None:
        noise_std = 1.0
    else:
        noise_std = noise_std_extra_schedule_fn(t_single)

    values = None
    if x0:
        _, _, _, x_recon = model.p_mean_variance(x=x, noise=pred_noise, hard_conds=hard_conds, context=context, t=t)
        return x + model_std * noise * noise_std, x_recon
    return x + model_std * noise * noise_std, values

@torch.no_grad()
def noise_ddpm_sample_fn(
        model, x, hard_conds, context, t,
        step_size=0.5,
        negative_step_size=0.5,
        guide=None,
        gradient_free_guide_ver=None,
        sampler=None,
        structured_noise=False,
        n_guide_steps=1,
        t_start_guide=torch.inf,
        noise_std_extra_schedule_fn=None,  # 'linear'
        noise_scale=None,
        x0=True,
        latent_guide=True,
        sample_num=10,
        cost=None,
        debug=False,
        tensor_args=None,
        **kwargs
):
    
    # inputs:
    # model: GuassianDiffusionModel itself
    # x: x_{t-1}, latent at the previous timestep
    # noise_std_extra_schedule_fn: maybe about noise scheduls (alpha... ,etc)

    # currently required,
    # model noise: To change this code as noise guidance (obtained) -> from p_mean _variance
    # correspoing scaler: sqrt(1-alpha_cumprod) -> noise_scale

    t_single=t[0]
    if t_single <=0:
        step_size = min(negative_step_size, step_size)
    pred_noise = model.model(x, t, context)
    batch, traj_len, state_dim = pred_noise.shape

    if gradient_free_guide_ver is not None:
        model_mean, _, _, _ = model.p_mean_variance(x=x, noise=pred_noise, hard_conds=hard_conds, context=context, t=t)
        # sample_num = max(int(sample_num * (1 - t_single/25)), 5)
        multi_shape = (sample_num, batch, traj_len, state_dim)
        
        # Efficient batch sampling of direction with log probability computation
        direction = sampler.sample(multi_shape[:-2])
        direction_logprob = sampler.log_prob(direction)
        direction = noise_scale * direction.view(multi_shape)

        # Efficiently initialize multi_pred_noise without expand and clone
        guider = model_mean.repeat(sample_num, 1, 1, 1) + direction
        
    else:
        model_mean, _, _, x_recon = model.p_mean_variance(x=x, noise=pred_noise, hard_conds=hard_conds, context=context, t=t)
        guider = model_mean
        direction = None
        direction_logprob = None

    if guide is not None and t_single < t_start_guide:
        pred_noise = noise_guide_gradient_steps(
            guider,
            guide=guide,
            latent_guide=latent_guide,
            gradient_free_guide_ver=gradient_free_guide_ver,
            model=model,
            step_size=step_size,
            sampled_direction=direction,
            sampled_logprob=direction_logprob,
            sample_num=sample_num,
            n_guide_steps=n_guide_steps,
            noise=pred_noise,
            noise_scale=noise_scale,
            debug=False,
            **kwargs
        )

    model_mean, model_variance, _, x_recon = model.p_mean_variance(x=x, noise=pred_noise, hard_conds=hard_conds, context=context, t=t)
    x = model_mean

    if cost is not None:
        x = resample_model_mean(cost, x_recon, x)

    model_std = torch.sqrt(model_variance)

    if structured_noise:
        alpha = math.cos(torch.pi/2 * t_single/model.n_diffusion_steps)
        noise = (1-alpha) * sampler.sample((batch,)).reshape(batch, traj_len, state_dim) + alpha * torch.randn_like(x)

        # alpha =  t_single/model.n_diffusion_steps
        # noise = alpha * sampler.sample((batch,)).reshape(batch, traj_len, state_dim) + (1 - alpha) *torch.randn_like(x)
        
        # noise = sampler.sample((batch,)).reshape(batch, traj_len, state_dim)
    else:
        noise = torch.randn_like(x)
    noise[t == 0] = 0

    # For smoother results, we can decay the noise standard deviation throughout the diffusion
    # this is roughly equivalent to using a temperature in the prior distribution
    if noise_std_extra_schedule_fn is None:
        noise_std = 1.0
    else:
        noise_std = noise_std_extra_schedule_fn(t_single)
    
    values = None
    if x0:
        return x + model_std * noise * noise_std, x_recon
    return x + model_std * noise * noise_std, values

@torch.no_grad()
def noise_ddim_sample_fn(
        model, x, hard_conds, context, t, alpha_next,
        guide=None,
        gradient_free_guide_ver=None,
        sampler=None,
        structured_noise=False,
        n_guide_steps=1,
        eta=None,
        t_start_guide=torch.inf,
        noise_std_extra_schedule_fn=None,  # 'linear'
        noise_scale=None,
        x0=True,
        latent_guide=True,
        sample_num=10,
        direct_forward=False,
        step_size=0.5,
        negative_step_size=0.5,
        cost=None,
        debug=False,
        tensor_args=None,
        **kwargs
):
    t_single=t[0]
    if t_single <=0:
        step_size = min(negative_step_size, step_size)

    pred_noise = model.model(x, t, context)
    batch, traj_len, state_dim = pred_noise.shape
    # gradient_free_guide_ver = kwargs['gradient_free_guide_ver']

    if gradient_free_guide_ver is not None:
        model_mean, _, _, _ = model.p_mean_variance(x=x, noise=pred_noise, hard_conds=hard_conds, context=context, t=t)
        sample_num = max(int(sample_num * (1 - t_single/model.n_diffusion_steps)), 2)
        multi_shape = (sample_num, batch, traj_len, state_dim)

        # Efficient batch sampling of direction with log probability computation
        direction = sampler.sample(multi_shape[:-2])
        # direction_logprob = sampler.log_prob(direction)
        direction = noise_scale * direction.view(multi_shape)

        # Efficiently initialize multi_pred_noise without expand and clone
        guider = model_mean.repeat(sample_num, 1, 1, 1) + direction
        
    else:
        model_mean, _, _, _ = model.p_mean_variance(x=x, noise=pred_noise, hard_conds=hard_conds, context=context, t=t)
        guider = model_mean
        direction = None
    direction_logprob = None

    if guide is not None and t_single < t_start_guide:
        pred_noise = noise_guide_gradient_steps(
            guider,
            guide=guide,
            latent_guide=latent_guide,
            gradient_free_guide_ver=gradient_free_guide_ver,
            model=model,
            step_size=step_size,
            sampled_direction=direction,
            sampled_logprob=direction_logprob,
            sample_num=sample_num,
            n_guide_steps=n_guide_steps,
            noise=pred_noise,
            noise_scale=noise_scale,
            debug=False,
            **kwargs
        )

    x_recon = torch.clamp((x - noise_scale*pred_noise) / (torch.sqrt(1-noise_scale**2) +1e-10), min=-1, max=1)

    # For non Markovian DDPM,
    # no noise when t == 0
    if structured_noise:
        alpha = math.cos(torch.pi/2 * t_single/model.n_diffusion_steps)
        noise = (1-alpha) * sampler.sample((batch,)).reshape(batch, traj_len, state_dim) + alpha * torch.randn_like(x)

        # alpha =  t_single/model.n_diffusion_steps
        # noise = alpha * sampler.sample((batch,)).reshape(batch, traj_len, state_dim) + (1 - alpha) *torch.randn_like(x)
        
        # noise = sampler.sample((batch,)).reshape(batch, traj_len, state_dim)

    else:
        noise = torch.randn_like(x)
    noise[t == 0] = 0

    if direct_forward:
        return alpha_next.sqrt() * x_recon + (1-alpha_next).sqrt() * noise, x_recon

    cur_next_ratio = (1-noise_scale**2)/alpha_next
    model_std = eta * (torch.sqrt(1-alpha_next) / noise_scale) * torch.sqrt(1-cur_next_ratio) # Non Markovian DDPM
    model_variance = model_std**2

    c_square = torch.clamp(1 - alpha_next - model_variance, min=0) if t_single > 1 else torch.zeros_like(model_variance) # Non Markovian DDPM
    c = c_square.sqrt()

    x = alpha_next.sqrt() * x_recon + c * pred_noise

    # For smoother results, we can decay the noise standard deviation throughout the diffusion
    # this is roughly equivalent to using a temperature in the prior distribution
    if noise_std_extra_schedule_fn is None:
        noise_std = 1.0
    else:
        noise_std = noise_std_extra_schedule_fn(t_single)
    
    values = None
    if x0:
        return x + model_std * noise * noise_std, x_recon # Non Markovian DDPM
    return x + model_std * noise * noise_std, values

def mean_guide_gradient_steps(
    guider, x,
    sampled_direction=None,
    sample_num=None,
    hard_conds=None,
    guide=None,
    n_guide_steps=1, 
    scale_grad_by_std=False,
    step_size=None,
    model_var=None,
    debug=False,
    **kwargs
):
    for _ in range(n_guide_steps):
        if sampled_direction is None:
            grad_scaled = guide(guider)
        else:
            raise NotImplementedError
            #grad_scaled = guide(guider, sampled_direction, sample_num=sample_num, temperature=model_var)

        if scale_grad_by_std: 
            grad_scaled = model_var * grad_scaled

        x = x + step_size*grad_scaled
    return x

def noise_guide_gradient_steps(
    guider, noise,
    perturbed_noise=None,
    gradient_free_guide_ver=None,
    latent_guide=False,
    sampled_direction=None,
    sampled_logprob=None,
    sample_num=None,
    guide=None,
    n_guide_steps=1,
    noise_scale=None,
    step_size=None,
    debug=False,
    **kwargs
):
    for _ in range(n_guide_steps):
        if gradient_free_guide_ver is None:
            if latent_guide:
                grad_scaled = guide(guider)
            else:
                grad_scaled = guide(guider, noise, noise_scale)
        else:
            if gradient_free_guide_ver == 'STOMP':
                grad_scaled = guide(guider, noise, sampled_direction, noise_scale, sample_num=sample_num, temperature=noise_scale*1e-4) # 1e-4
            elif gradient_free_guide_ver == 'Adv':
                grad_scaled = guide(guider, noise, sampled_direction, sampled_logprob, noise_scale, sample_num=sample_num, temperature=noise_scale*1e-2)
            
        #noise_guidance
        if noise_scale is not None:
            if gradient_free_guide_ver is not None:
                # grad_scaled : - grad of cost fn
                grad_scaled = (1/noise_scale + noise_scale)*grad_scaled
            else:
                # grad_scaled : - grad of cost fn
                if latent_guide:
                    grad_scaled = noise_scale * grad_scaled
                else:
                    grad_scaled = ((1-noise_scale**2)/noise_scale + noise_scale) * grad_scaled
                    # grad_scaled = noise_scale * grad_scaled # Universal Guidance

        noise = noise - step_size*grad_scaled

    return noise


def resample_model_mean(cost_fn, x_recon, model_mean, temperature=1):
    num_traj = x_recon.shape[0]

    cost = cost_fn(x_recon, return_invidual_costs_and_weights=False)
    # maxmin = cost.max() - cost.min()
    # normalized_cost = (cost - cost.min()) / maxmin if maxmin > 1 else cost-cost.min()
    
    prob = torch.softmax(-cost/temperature, dim=0)
    resampled_indices = torch.multinomial(prob, num_traj, replacement=True)
    return model_mean[resampled_indices]
