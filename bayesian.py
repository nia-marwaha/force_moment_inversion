from turtle import lt

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

#noise variance calculation for uncertainty based on a given window and event start/end
def noise_variance(st_noise, noise_window, event_starttime, event_endtime):
    noise_variances = []
    for tr in st_noise:
        noise_slice = tr.slice(starttime=event_starttime, endtime=event_endtime)
        noise_variances.append(np.var(noise_slice.data))
    return noise_variances

#BIC calculation for model selection (estimate)
def BIC(data, misfit, Cd, Cd_inv, Cm_post, G, term1, inversion_type=None):
    N = data.d.shape[0]
    chi = misfit.T @ Cd_inv @ misfit
    log_det_Cd = np.sum(np.log(np.diag(Cd)))
    bic = N * np.log(2 * np.pi) + log_det_Cd + chi
    if inversion_type=='joint':
        R=Cm_post @ G.T @ Cd_inv @ G
    else:
        R = Cm_post @ term1
    #because of time correlation within model, effective number of parameters is the trace of the hat matrix R
    k_eff = np.trace(R)
    return k_eff * np.log(N) + bic

#log evidence calculation for model selection
def log_evidence(data, m, m_prior, Cd, Cd_inv, Cm, Cm_post, G):
    N = data.d.shape[0]
    d_tilde= G @ m
    exp_term = (data.d-d_tilde).T @ Cd_inv @ data.d + (m_prior-m).T @ np.linalg.inv(Cm) @ m_prior
    sign_d, log_det_Cd = np.linalg.slogdet(Cd)
    sign_m, log_det_Cm = np.linalg.slogdet(Cm)
    sign_p, log_det_Cm_post = np.linalg.slogdet(Cm_post)
    if sign_d <= 0 or sign_m <= 0 or sign_p <= 0:
        raise np.linalg.LinAlgError('a covariance matrix is not positive definite')
    log_evidence_value = -0.5 * (N * np.log(2 * np.pi) + log_det_Cd + exp_term - log_det_Cm_post + log_det_Cm)
    return log_evidence_value

#temporal kernel function for time correlation, relates time steps to eachother based on a correlation length
def temporal_kernel(gl, dt, corr_length, kernel='gaussian'):
    t = np.arange(gl) * dt
    diff = t[:, None] - t[None, :]
    if kernel == 'gaussian':
        K = np.exp(-0.5 * (diff / corr_length) ** 2)
    elif kernel == 'exponential':
        K = np.exp(-np.abs(diff) / corr_length)
    else:
        raise ValueError("kernel must be 'gaussian' or 'exponential'")
    return K

#force inversion 
def force_bayesian(data, theta, force_mag, uncertainty, dt, nugget, corr_length, m_prior, noise_variances):
    G_matrix=data.G
    d_matrix=data.d
    n = G_matrix.shape[1]
    gl = int(n / 3) 
    #take a seperate uncertainty for each component, as they may be different
    R_uncertainty=uncertainty[0]
    T_uncertainty=uncertainty[1]
    Z_uncertainty=uncertainty[2]
    sigma_R = R_uncertainty*force_mag ** 2
    sigma_T = T_uncertainty*force_mag ** 2
    cov_RT = np.diag([sigma_R, sigma_T])
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
    #create a rotatied covariance matrix for the North and East components based on the given angle theta and force magnitude
    cov_NE = rot @ cov_RT @ rot.T

    #noise variance uncertainty
    Cd_diag = np.concatenate([np.full((gl), var) for var in noise_variances])
    Cd = np.diag(Cd_diag)
    Cd_inv = np.diag(1 / np.diag(Cd))

    G_T = G_matrix.T
    data_residual = d_matrix - (G_matrix @ m_prior)

    f_corr = temporal_kernel(gl, dt, corr_length, kernel='gaussian')

    Cm_Z = (Z_uncertainty*force_mag ** 2) * f_corr
    Cm_NE = np.kron(cov_NE, f_corr)
    Cm = block_diag(Cm_Z, Cm_NE)
    #final Cm matrix is the block diagonal of the Z component and the NE components, with a nugget added to account for uncertainty in the model
    Cm += nugget * force_mag ** 2 * np.eye(Cm.shape[0])
    Cm_inv = np.linalg.inv(Cm)

    #baysian inversion calculations
    term1 = G_T @ Cd_inv @ G_matrix
    term2 = Cm_inv
    Cm_post = np.linalg.inv(term1 + term2)
    m = m_prior + Cm_post @ G_T @ Cd_inv @ data_residual

    #calculate misfit, misfit norm, model norm, BIC, and log evidence for model selection
    misfit = data_residual - G_matrix @ (m - m_prior)
    misfit_norm = np.sqrt(misfit.T @ Cd_inv @ misfit) 
    model_norm = np.sqrt((m - m_prior).T @ Cm_inv @ (m - m_prior))
    bic=BIC(data, misfit, Cd, Cd_inv, Cm_post, G_matrix, term1, inversion_type='single') 
    log_evidence_value = log_evidence(data, m, m_prior, Cd, Cd_inv, Cm, Cm_post, G_matrix)
    print(f"Log Evidence: {log_evidence_value}")
    return m, Cm_post, Cd, misfit, misfit_norm, model_norm, term1, bic, log_evidence_value

#force inversion plots
def force_bayesian_plot(m, dt, Cm_post, given_forces, component_labels, test='yes'):
    N_steps = len(m) // 3
    time = np.arange(N_steps) * dt
    F_z = -m[0 : N_steps]
    F_n = m[N_steps : 2 * N_steps]
    F_e = m[2 * N_steps : 3 * N_steps]

    uncertainties = np.sqrt(np.diag(Cm_post))
    std_z = uncertainties[0 : N_steps]
    std_n = uncertainties[N_steps : 2 * N_steps]
    std_e = uncertainties[2 * N_steps : 3 * N_steps]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    forces = [F_z, F_n, F_e]
    stds = [std_z, std_n, std_e]
    colors = ['black', 'red', 'blue'] 

    for i, ax in enumerate(axes):
        ax.plot(time, forces[i], color=colors[i], linewidth=2, label='bayesian')
        
        ax.fill_between(time, 
                        forces[i] - stds[i], 
                        forces[i] + stds[i], 
                        color=colors[i], alpha=0.2, edgecolor='none', label=r'bayesian uncertainty')

        #to compare with given forces, plot them if test=='yes'
        if test=='yes':
            ax.plot(given_forces[i], color='black', linestyle='--', label='given')
        else:
            continue

        ax.set_ylabel(f'{component_labels[i]}\nForce (N)')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    axes[-1].set_xlabel('Time (s)')
    axes[-1].set_xlim(time[0], time[-1])
    
    return fig, F_z, F_n, F_e

#function to compare the given and model created seismograms for each station and component, and calculate the variance reduction and overall fit for each trace
def comparison(data, m, stations, components, st_obs_filt, n):
    fig, axes = plt.subplots(nrows=len(stations), ncols=3, figsize=(14, 2.5*len(stations)), sharex=True, squeeze=False)
    vr_records = []

    test_d=data.G @ m
    for i, tr in enumerate(st_obs_filt):
        start=i*n
        end=start+n
        comp=tr.stats.channel[-1]
        net, sta=tr.stats.network, tr.stats.station
        if (net, sta) not in stations:
            continue 
        row_idx=stations.index((net, sta))
        col_idx=components.index(comp) if comp in components else None
        if col_idx is None:
            continue

        ax = axes[row_idx, col_idx]
        syn_data=test_d[start:end]
        obs_data=data.d[start:end]

        ax.plot(tr.times()[:n], syn_data, color='tab:red', label='synthetic')
        ax.plot(tr.times()[:n], obs_data, color='black', label='observed', alpha=0.7)

        resid_norm_sq = np.sum((obs_data - syn_data) ** 2)
        obs_norm_sq = np.sum(obs_data ** 2)
        vr = 100 * (1 - resid_norm_sq / obs_norm_sq) if obs_norm_sq > 0 else np.nan
        correlation = np.corrcoef(obs_data, syn_data)[0, 1]
        fit_pct = max(correlation, 0) * 100
        vr_records.append((net, sta, comp, vr))
        ax.set_title(f'{net}.{sta} {comp}  VR={vr:.1f}% fit={fit_pct:.1f}%', fontsize=18)

    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    plt.show()

    overall_vr = 100 * (1 - np.sum((data.d - test_d) ** 2) / np.sum(data.d ** 2))
    print(f'Overall VR: {overall_vr:.2f}%')
    return fig, overall_vr, vr_records

#moment inversion
def moment_bayesian(data, mw, uncertainty, dt, nugget, corr_length, m_prior, noise_variances):
    G_matrix=data.G
    d_matrix=data.d

    n = G_matrix.shape[1]

    #conversion from moment magnitude to seismic moment,
    m0 = 10**(1.5 * mw + 9.1)
    sigma = uncertainty*m0
    gl = int(n / 6)  

    mt_corr = temporal_kernel(gl, dt, corr_length, kernel='gaussian')

    #frobenius weights for the moment tensor components
    frobenius_weights = np.array([1, 1, 1, 2, 2, 2])    

    sigma = sigma **2 / (np.sum(frobenius_weights))
    variances = np.ones(6) * sigma
    Cm = np.zeros((6 * gl, 6 * gl))
    for i in range(6):
        start_idx = i * gl
        end_idx = start_idx + gl
        Cm[start_idx:end_idx, start_idx:end_idx] = mt_corr * variances[i]
    #population of the Cm matrix with the temporal correlation and variances for each moment tensor component, with a nugget added to account for uncertainty in the model
    Cm += nugget * sigma * np.eye(Cm.shape[0])

    Cd_diag = np.concatenate([
        np.full((gl), var) for var in noise_variances
    ])
    Cd = np.diag(Cd_diag)

    Cd_inv = np.diag(1.0 / np.diag(Cd))
    Cm_inv = np.linalg.inv(Cm)

    G_T = G_matrix.T
    data_residual = d_matrix - (G_matrix @ m_prior)

    term1 = G_matrix.T @ Cd_inv @ G_matrix
    term2 = Cm_inv

    Cm_post= np.linalg.inv(term1 + term2)
    m= m_prior + Cm_post @ G_matrix.T @ Cd_inv @ data_residual
    misfit = data_residual - G_matrix @ (m - m_prior)
    misfit_norm = np.sqrt(misfit.T @ Cd_inv @ misfit)
    model_norm = np.sqrt((m - m_prior).T @ Cm_inv @ (m - m_prior))
    bic=BIC(data, misfit, Cd, Cd_inv, Cm_post, G_matrix, term1, inversion_type='single')
    log_evidence_value = log_evidence(data, m, m_prior, Cd, Cd_inv, Cm, Cm_post, G_matrix)
    print(f"Log Evidence: {log_evidence_value}")
    return m, Cm_post, Cd, misfit, misfit_norm, model_norm, bic, log_evidence_value

#momnent inversion plots
def moment_bayesian_plot(m, dt, Cm_post, given_moment, mt_labels, test='yes'):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

    bayesian_vals = m
    bayesian_errs = np.sqrt(np.diag(Cm_post))
    N_steps = len(m) // 6
    time = np.arange(N_steps) * dt

    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(10,12), sharex=True)
    moment_functions={}
    for i, (label, ax) in enumerate(zip(mt_labels, axes)):
        vals=bayesian_vals[i*N_steps:(i+1)*N_steps]
        errs=bayesian_errs[i*N_steps:(i+1)*N_steps]
        moment_functions[label] = vals

        ax.plot(time, vals, color=colors[i], label='bayseian inversion')
        ax.fill_between(time, vals-errs, vals+errs, alpha=0.3, color='gray')
        if test=='yes':
            ax.plot(given_moment[i], color='black', label='given')
        else:
            continue
        
        ax.set_ylabel(label)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('Time (s)')
    return fig, moment_functions

#joint inversion, combining both force and moment inversions into a single inversion, with the same calculations as the individual inversions
def joint_bayesian(data, theta, force_mag, mw, uncertainty_f, uncertainty_m, dt, nugget, corr_length_f, corr_length_m, m_prior, noise_variances):
    G_matrix = data.G          
    d_matrix = data.d
    n = G_matrix.shape[1]
    gl = int(n / 9)
    R_uncertainty=uncertainty_f[0]
    T_uncertainty=uncertainty_f[1]
    Z_uncertainty=uncertainty_f[2]
    sigma_R = (R_uncertainty*force_mag) ** 2
    sigma_T = (T_uncertainty*force_mag) ** 2
    cov_RT = np.diag([sigma_R, sigma_T])
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
    cov_NE = rot @ cov_RT @ rot.T

    force_corr = temporal_kernel(gl, dt, corr_length_f, kernel='exponential')
    Cm_Z = (Z_uncertainty*force_mag ** 2) * force_corr
    Cm_NE = np.kron(cov_NE, force_corr)
    Cm_force = block_diag(Cm_Z, Cm_NE)

    m0 = 10 ** (1.5 * mw + 9.1)  
    sigma_mt = uncertainty_m* m0

    mt_corr = temporal_kernel(gl, dt, corr_length_m, kernel='gaussian')
    frobenius_weights = np.array([1, 1, 1, 2, 2, 2])    

    sigma_mt = sigma_mt **2 / (np.sum(frobenius_weights))
    variances = np.ones(6) * sigma_mt
    Cm_mt = np.zeros((6 * gl, 6 * gl))
    for i in range(6):
        start_idx = i * gl
        end_idx = start_idx + gl
        Cm_mt[start_idx:end_idx, start_idx:end_idx] = mt_corr * variances[i]
    Cm_mt += nugget * sigma_mt * np.eye(Cm_mt.shape[0])


    Cm = block_diag(Cm_force,Cm_mt)
    idx = 3 * gl
    Cm[:idx, :idx] += nugget * (force_mag ** 2) * np.eye(idx)
    Cm[idx:, idx:] += nugget * (sigma_mt) * np.eye(Cm.shape[0] - idx)
    assert Cm.shape == (n, n), f'Cm shape {Cm.shape} does not match n={n}'

    #scaling of the G matrix and Cm matrix to improve numerical stability, by normalizing the force and moment components separately
    force_scale = np.linalg.norm(G_matrix[:, :3*gl])
    mt_scale = np.linalg.norm(G_matrix[:, 3*gl:])
    scale_vec = np.hstack([
        np.full(3 * gl, force_scale),
        np.full(6 * gl, mt_scale),
    ])

    G_scaled = G_matrix / scale_vec
    Cm_scaled = (scale_vec[:, None] * Cm) * scale_vec[None, :]
    Cm_inv_scaled = np.linalg.inv(Cm_scaled)


    Cd_diag = np.concatenate([
        np.full(gl, var) for var in noise_variances
    ])
    Cd = np.diag(Cd_diag)
    Cd_inv = np.diag(1.0 / np.diag(Cd))

    m_prior_scaled = m_prior * scale_vec 
    data_residual = d_matrix - (G_matrix @ m_prior) 

    term1 = G_scaled.T @ Cd_inv @ G_scaled
    term2 = Cm_inv_scaled

    Cm_post_scaled = np.linalg.inv(term1 + term2)
    m_scaled = m_prior_scaled + Cm_post_scaled @ G_scaled.T @ Cd_inv @ data_residual

    #unscale the model parameters and covariance matrix to return to the original scale
    m = m_scaled / scale_vec
    Cm_post = Cm_post_scaled / np.outer(scale_vec, scale_vec) 
    m_both_bayes, Cm_post_both_bayes = m.copy(), Cm_post.copy()
    misfit = data_residual - G_matrix @ (m - m_prior)
    misfit_norm = np.sqrt(misfit.T @ Cd_inv @ misfit)
    model_norm = np.sqrt((m - m_prior).T @ Cm_inv_scaled @ (m - m_prior))
    bic=BIC(data, misfit, Cd, Cd_inv, Cm_post_scaled, G_scaled, term1, inversion_type='joint')
    log_evidence_value = log_evidence(data, m_scaled, m_prior, Cd, Cd_inv, Cm_scaled, Cm_post_scaled, G_scaled)
    print(f"Log Evidence: {log_evidence_value}")
    return m, Cm_post, misfit,misfit_norm, model_norm, bic, log_evidence_value

#joint inversion plots
def joint_plots(component_labels, m_both, m_force, m_moment, Cm_post, dt, given, test='yes'):
    N_steps_both = len(m_both) // 9
    N_steps_force = len(m_force) // 3
    N_steps_moment = len(m_moment) // 6

    time_both = np.arange(N_steps_both) * dt
    time_force = np.arange(N_steps_force) * dt
    time_moment = np.arange(N_steps_moment) * dt

    both_bayes_errs = np.sqrt(np.diag(Cm_post))

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'darkblue', 'turquoise', 'magenta']
    fig, axes = plt.subplots(nrows=9, ncols=1, figsize=(10, 22), sharex=True)

    for i, (label, ax) in enumerate(zip(component_labels, axes)):
        if i ==0:
            vals_both = -m_both[i*N_steps_both:(i+1)*N_steps_both]
            errs_both = -both_bayes_errs[i*N_steps_both:(i+1)*N_steps_both]
        else: 
            vals_both = m_both[i*N_steps_both:(i+1)*N_steps_both]
            errs_both = both_bayes_errs[i*N_steps_both:(i+1)*N_steps_both]
        ax.plot(time_both, vals_both, color=colors[i], linewidth=2, label='joint results')
        if test=='yes':
            ax.plot(given[i], color='black', linestyle='--', label='given')
        else:
            continue
        ax.fill_between(time_both, vals_both - errs_both, vals_both + errs_both,
                        alpha=0.25, color='gray', label='uncertainty (joint)')

        if i == 0:
            vals_single = -m_force[i*N_steps_force:(i+1)*N_steps_force]
            t_single = time_force
        elif i < 3:
            vals_single = m_force[i*N_steps_force:(i+1)*N_steps_force]
            t_single = time_force
        else:
            vals_single = m_moment[(i-3)*N_steps_moment:(i-2)*N_steps_moment]
            t_single = time_moment

        ax.plot(t_single, vals_single, colors[i], linestyle='--',label='single inversion results')
        ax.set_ylabel(label)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    axes[0].legend(loc='upper right', fontsize=7, ncol=2)
    axes[-1].set_xlabel('Time (s)')
    return fig 
