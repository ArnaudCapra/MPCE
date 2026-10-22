import ast
import inspect
import textwrap
import numpy as np
from numba import njit, prange
from numpy.polynomial.legendre import leggauss

@njit(cache=True)
def kernel_weights(h, lag, m, nodes, quadrature_weights):
    """Poids sans unité pour f_d(r)=(d-1+r)^(h-1/2), 0<r<1.

    Les m premiers modes sont sqrt(m)*1_{[j/m,(j+1)/m]}.
    Le dernier est le résidu de f_1 après projection sur ces modes.
    L'intégrale réelle multiplie ces poids par dt**h.
    """
    beta = h + 0.5
    alpha = h - 0.5
    a1 = np.empty(m)
    out = np.zeros(m + 1)
    root_m = np.sqrt(m)
    norm1 = 1.0 / (2.0 * h)
    projected_norm1 = 0.0
    cross_projected = 0.0
    for j in range(m):
        lo = j / m
        hi = (j + 1.0) / m
        a1[j] = root_m * (hi ** beta - lo ** beta) / beta
        out[j] = root_m * ((lag - 1.0 + hi) ** beta - (lag - 1.0 + lo) ** beta) / beta
        projected_norm1 += a1[j] * a1[j]
        cross_projected += a1[j] * out[j]
    if h == 0.5:
        return out
    residual_variance = norm1 - projected_norm1
    if residual_variance <= 0.0:
        raise FloatingPointError('Résidu singulier non positif : précision insuffisante.')
    b1 = np.sqrt(residual_variance)
    if lag == 1:
        out[m] = b1
    else:
        cross_exact = 0.0
        for j in range(len(nodes)):
            r = nodes[j] ** (1.0 / beta)
            cross_exact += quadrature_weights[j] * (lag - 1.0 + r) ** alpha / beta
        out[m] = (cross_exact - cross_projected) / b1
    return out

@njit(cache=True)
def permanent_modes(coarse_z, e):
    """m normales de sous-cellules conditionnées par leur somme, plus un résidu.

    sum(z_sub)/sqrt(m) = coarse_z. Une base de Helmert utilise les
    m-1 premières normales de e. La dernière normale porte le résidu singulier.
    Les mêmes modes sont réutilisés pour toutes les dates futures.
    """
    m = len(e)
    modes = np.empty(m + 1)
    for j in range(m):
        modes[j] = coarse_z / np.sqrt(m)
    for ell in range(m - 1):
        scale = 1.0 / np.sqrt((ell + 1.0) * (ell + 2.0))
        for j in range(ell + 1):
            modes[j] += scale * e[ell]
        modes[ell + 1] -= (ell + 1.0) * scale * e[ell]
    modes[m] = e[m - 1]
    return modes

@njit(cache=True, parallel=True)
def _integrate(z, e, h, y, c, dt, hist, spot, carry, y_speed, y_mean, y_diffusion, initial_log_variance, nodes, weights):
    p, n = z.shape[1:]
    m = e.shape[2]
    x = np.empty((p, n + 1))
    v = np.empty((p, n + 1))
    for q in prange(p):
        modes = np.empty((n, m + 1))
        for k in range(n):
            modes[k] = permanent_modes(z[1, q, k], e[q, k])
        x[q, 0] = spot
        v[q, 0] = initial_log_variance
        for i in range(1, n + 1):
            value = initial_log_variance + hist[i]
            for k in range(i):
                beta = h[q, k] + 0.5
                lag = i - k
                it = dt ** beta * (lag ** beta - (lag - 1.0) ** beta) / beta
                yp = max(y[q, k], 0.0)
                value += c[q, k] * y_speed * (y_mean - yp) * it
                if y_diffusion != 0.0:
                    a = kernel_weights(h[q, k], lag, m, nodes, weights)
                    stochastic_integral = 0.0
                    for ell in range(m + 1):
                        stochastic_integral += a[ell] * modes[k, ell]
                    value += c[q, k] * y_diffusion * np.sqrt(yp) * dt ** h[q, k] * stochastic_integral
            v[q, i] = value
            vol = np.exp(0.5 * v[q, i - 1])
            x[q, i] = x[q, i - 1] * np.exp((carry - 0.5 * vol * vol) * dt + vol * np.sqrt(dt) * z[2, q, i - 1])
    return (x, v)

"""MPCE : code utilisateur avec correction du bruit brownien par cellule.
Dépendance de projet conservée : rough_heston.py.
Noyaux browniens incorporés dans ce module.
"""
import numpy as np
import pandas as pd
from scipy.special import gamma, ndtri
from numba import njit, prange, set_num_threads
from rough_heston import validate_market, implied_vol

from numpy.polynomial.legendre import leggauss
_BROWNIAN_SUBCELLS = 8
_BROWNIAN_QUADRATURE_ORDER = 64
_BROWNIAN_NODES, _BROWNIAN_WEIGHTS = leggauss(_BROWNIAN_QUADRATURE_ORDER)
_BROWNIAN_NODES = (_BROWNIAN_NODES + 1.0) / 2.0
_BROWNIAN_WEIGHTS = _BROWNIAN_WEIGHTS / 2.0

@njit(cache=True)
def drivers(z, dt, h0, y0, h_mean, h_speed, h_diffusion, h_lower, h_upper, y_speed, y_mean, y_diffusion):
    n = z.shape[1]
    h = np.empty(n + 1)
    y = np.empty(n + 1)
    h[0] = h0
    y[0] = y0
    clips = 0
    for k in range(n):
        raw = h[k] + h_speed * (h_mean - h[k]) * dt + h_diffusion * np.sqrt(h[k] * (1 - h[k]) * dt) * z[0, k]
        if raw < h_lower or raw > h_upper:
            clips += 1
        h[k + 1] = min(h_upper, max(h_lower, raw))
        yp = max(y[k], 0.0)
        y[k + 1] = y[k] + y_speed * (y_mean - yp) * dt + y_diffusion * np.sqrt(yp * dt) * z[1, k]
    return (h, y, clips)

def coefficient(h):
    return np.sqrt(2 * h * np.sin(np.pi * h) * gamma(2 * h)) / gamma(h + 0.5)

def integrate_and_price(z, e, h, y, c, dt, hist, spot, carry, y_speed, y_mean, y_diffusion, initial_log_variance):
    if e.ndim != 3 or e.shape != (z.shape[1], z.shape[2], _BROWNIAN_SUBCELLS):
        raise ValueError('Bruit e attendu de forme (paths, steps, subcells).')
    return _integrate(z, e, h, y, c, dt, hist, spot, carry, y_speed, y_mean, y_diffusion, initial_log_variance, _BROWNIAN_NODES, _BROWNIAN_WEIGHTS)

def validate_mc(model, simulation, market, grid):
    maturities, moneyness = validate_market(market, grid)
    scalar = [v for k, v in model.items() if k != 'correlation']
    if not np.isfinite(scalar).all():
        raise ValueError('Paramètres MPCE non finis.')
    if not 0 < model['H_min'] < model['H_max'] < 1:
        raise ValueError('Bornes de h invalides.')
    if not model['H_min'] <= model['H0'] <= model['H_max']:
        raise ValueError('h initial hors bornes.')
    if not model['H_min'] <= model['eta'] <= model['H_max']:
        raise ValueError('Moyenne de h hors bornes.')
    if min(model['lambda'], model['kappa'], model['theta'], model['y0']) < 0:
        raise ValueError('Vitesses et niveaux de y non négatifs requis.')
    if min(model['zeta'], model['xi']) < 0:
        raise ValueError('Diffusions non négatives requises.')
    for key in ['n_paths', 'n_steps', 'batch_size', 'threads']:
        if not isinstance(simulation[key], int) or simulation[key] < 1:
            raise ValueError(f'{key} entier positif requis.')
    if not isinstance(simulation['seed'], int) or simulation['seed'] < 0:
        raise ValueError('Graine entiere non negative requise.')
    if not np.isfinite(simulation['horizon']) or simulation['horizon'] <= 0:
        raise ValueError('Horizon positif fini requis.')
    if simulation['n_paths'] < 2:
        raise ValueError('Au moins deux trajectoires requises pour les erreurs standards.')
    if simulation['horizon'] < max(maturities):
        raise ValueError('Horizon trop court.')
    if not 0 < simulation['confidence_level'] < 1:
        raise ValueError('Niveau de confiance invalide.')
    dt = simulation['horizon'] / simulation['n_steps']
    indices = np.rint(maturities / dt).astype(int)
    if np.max(abs(indices * dt - maturities)) > simulation['alignment_tolerance']:
        raise ValueError('Maturités non alignées sur le pas MC ; aucune interpolation automatique.')
    corr = np.asarray(model['correlation'], float)
    if corr.shape != (3, 3) or not np.isfinite(corr).all() or (not np.allclose(corr, corr.T)) or (not np.allclose(np.diag(corr), 1)):
        raise ValueError('Matrice de corrélation invalide.')
    return (dt, indices, np.linalg.cholesky(corr))

def build_history(model, simulation, dt, L):
    mode=simulation.get('history_mode','fixed')
    if mode not in ('fixed','zero'):raise ValueError('history_mode doit etre fixed ou zero.')
    if mode=='zero':
        return (np.zeros(simulation['n_steps']+1),np.array([model['H0']]),np.array([model['y0']]),0)
    nh = int(round(simulation.get('history_length', simulation.get('history_length', simulation['history_multiple'] * simulation['horizon'])) / dt))
    if nh < 1: raise ValueError('Historique positif requis.')
    rng = np.random.default_rng(np.random.SeedSequence([simulation.get('history_seed',simulation['seed']), 0]))
    z = L @ rng.standard_normal((3, nh))
    e = rng.standard_normal((nh, _BROWNIAN_SUBCELLS))
    h, y, clips = drivers(z, dt, model['H0'], model['y0'], model['eta'], model['lambda'], model['zeta'], model['H_min'], model['H_max'], model['kappa'], model['theta'], model['xi'])
    c = coefficient(h[:-1])
    yp = np.maximum(y[:-1], 0.0)
    hist = np.zeros(simulation['n_steps'] + 1)
    modes = np.array([permanent_modes(z[1, k], e[k]) for k in range(nh)])
    for k in range(nh):
        beta = h[k] + 0.5
        lag0 = nh - k
        a0 = kernel_weights(h[k], lag0, _BROWNIAN_SUBCELLS, _BROWNIAN_NODES, _BROWNIAN_WEIGHTS)
        it0 = dt ** beta * (lag0 ** beta - (lag0 - 1.0) ** beta) / beta
        for i in range(1, len(hist)):
            lag = lag0 + i
            a = kernel_weights(h[k], lag, _BROWNIAN_SUBCELLS, _BROWNIAN_NODES, _BROWNIAN_WEIGHTS)
            it = dt ** beta * (lag ** beta - (lag - 1.0) ** beta) / beta
            hist[i] += c[k] * (model['kappa'] * (model['theta'] - yp[k]) * (it - it0) + model['xi'] * np.sqrt(yp[k]) * dt ** h[k] * np.dot(a - a0, modes[k]))
    if not np.isfinite(hist).all():
        raise FloatingPointError('Historique non fini.')
    return (hist, h, y, int(clips))

def check_integral(z, e, h, y, c, v, dt, hist, model, indices):
    error = 0.0
    for i in indices:
        if i < 1 or i >= v.shape[1]:
            continue
        value = model['initial_log_variance'] + hist[i]
        for k in range(i):
            beta = h[0, k] + 0.5
            lag = i - k
            yp = max(y[0, k], 0.0)
            it = dt ** beta * (lag ** beta - (lag - 1.0) ** beta) / beta
            a = kernel_weights(h[0, k], lag, _BROWNIAN_SUBCELLS, _BROWNIAN_NODES, _BROWNIAN_WEIGHTS)
            modes = permanent_modes(z[1, 0, k], e[0, k])
            value += c[0, k] * model['kappa'] * (model['theta'] - yp) * it
            value += c[0, k] * model['xi'] * np.sqrt(yp) * dt ** h[0, k] * np.dot(a, modes)
        error = max(error, abs(value - v[0, i]))
    return float(error)

def simulate_single_history(model, simulation, market, grid):
    configure_numerics(simulation)
    dt, indices, L = validate_mc(model, simulation, market, grid)
    set_num_threads(simulation['threads'])
    hist, hh, yh, hist_clips = build_history(model, simulation, dt, L)
    n = simulation['n_steps']
    count = simulation['n_paths']
    samples = np.empty((count, len(indices)))
    conditional_enabled=simulation.get('conditional_mc',False)
    corr=np.asarray(model['correlation'],float)
    regression=np.linalg.solve(corr[:2,:2],corr[:2,2])
    independent_variance=1.-np.dot(corr[2,:2],regression)
    if independent_variance<=0:raise ValueError('Variance conditionnelle non positive.')
    conditional={key:np.empty_like(samples) for key in ['forward','variance','proxy_forward','proxy_variance']} if conditional_enabled else None
    sumx = np.zeros(n + 1)
    sumx2 = np.zeros(n + 1)
    sumv = np.zeros(n + 1)
    sumv2 = np.zeros(n + 1)
    sumvar = np.zeros(n + 1)
    sumdrift=np.zeros(n+1)
    sumstochastic=np.zeros(n+1)
    hclips = 0
    ynegative = 0
    hsum = 0.0
    hsum2 = 0.0
    ymin = np.inf
    vmin = np.inf
    vmax = -np.inf
    integral_error = 0.0
    for start in range(0, count, simulation['batch_size']):
        size = min(simulation['batch_size'], count - start)
        z = np.empty((3, size, n))
        e = np.empty((size, n, _BROWNIAN_SUBCELLS))
        for q in range(size):
            rng = np.random.default_rng(np.random.SeedSequence([simulation['seed'], 1, start + q]))
            z[:, q] = L @ rng.standard_normal((3, n))
            e[q] = rng.standard_normal((n, _BROWNIAN_SUBCELLS))
        h = np.empty((size, n + 1))
        y = np.empty_like(h)
        for q in range(size):
            h[q], y[q], clips = drivers(z[:, q], dt, hh[-1], yh[-1], model['eta'], model['lambda'], model['zeta'], model['H_min'], model['H_max'], model['kappa'], model['theta'], model['xi'])
            hclips += clips
        c = coefficient(h[:, :-1])
        x, v = integrate_and_price(z, e, h, y, c, dt, hist, market['spot'], market['rate'] - market['dividend'], model['kappa'], model['theta'], model['xi'], model['initial_log_variance'])
        if not np.isfinite(x).all() or not np.isfinite(v).all() or np.any(x <= 0):
            raise FloatingPointError('Trajectoires non finies ou sous-flux à zéro : aucun clipping.')
        if start == 0:
            integral_error = check_integral(z, e, h, y, c, v, dt, hist, model, simulation['integral_check_indices'])
        samples[start:start + size] = x[:, indices]
        if conditional_enabled:
            cf,cv,pf,pv=conditional_terms(z,v,dt,market['spot'],market['rate']-market['dividend'],regression,independent_variance,model['initial_log_variance'])
            for key,values in [('forward',cf),('variance',cv),('proxy_forward',pf),('proxy_variance',pv)]:
                if not np.isfinite(values).all():raise FloatingPointError('Termes conditionnels non finis.')
                conditional[key][start:start+size]=values[:,indices]
        sumx += x.sum(0)
        sumx2 += (x * x).sum(0)
        sumv += v.sum(0)
        sumv2 += (v * v).sum(0)
        if simulation.get('diagnostic_decomposition',False):
            _,drift_v=integrate_and_price(z,e,h,y,c,dt,np.zeros_like(hist),market['spot'],market['rate']-market['dividend'],model['kappa'],model['theta'],0.,model['initial_log_variance'])
            drift=drift_v-model['initial_log_variance']
            stochastic=v-model['initial_log_variance']-hist[None,:]-drift
            sumdrift+=drift.sum(0);sumstochastic+=stochastic.sum(0)
        variance = np.exp(v)
        if not np.isfinite(variance).all():
            raise FloatingPointError('Variance non finie.')
        sumvar += variance.sum(0)
        ynegative += int((y[:, 1:] < 0).sum())
        ymin = min(ymin, float(y.min()))
        hsum += float(h[:, 1:].sum())
        hsum2 += float((h[:, 1:] ** 2).sum())
        vmin = min(vmin, float(v.min()))
        vmax = max(vmax, float(v.max()))
    times = np.arange(n + 1) * dt
    forward = market['spot'] * np.exp((market['rate'] - market['dividend']) * times)
    mean = sumx / count
    sample_variance = np.maximum((sumx2 - sumx * sumx / count) / (count - 1), 0.0)
    se = np.sqrt(sample_variance / count)
    zscore = np.divide(mean - forward, se, out=np.full_like(se, np.nan), where=se > 0)
    if not np.isfinite(sumx2).all() or not np.isfinite(sumvar).all():
        raise FloatingPointError('Moments non finis.')
    diagnostics = pd.DataFrame(dict(T=times, mean_X=mean, theoretical_F=forward, mean_X_se=se, forward_zscore=zscore, mean_log_variance=sumv / count, mean_variance=sumvar / count))
    if simulation.get('diagnostic_decomposition',False):
        diagnostics['history_log_contribution']=hist
        diagnostics['mean_drift_log_contribution']=sumdrift/count
        diagnostics['mean_stochastic_log_contribution']=sumstochastic/count
        diagnostics['log_variance_population_variance']=sumv2/count-(sumv/count)**2
        diagnostics['rms_volatility']=np.sqrt(sumvar/count)
        diagnostics['decomposition_residual']=diagnostics.mean_log_variance-model['initial_log_variance']-hist-sumdrift/count-sumstochastic/count
    metadata = dict(history_h_terminal=float(hh[-1]), history_y_terminal=float(yh[-1]), history_h_clip_fraction=hist_clips / max(len(hh)-1,1), main_h_clip_fraction=hclips / (count * n), main_y_negative_fraction=ynegative / (count * n), history_y_negative_fraction=float(np.mean(yh < 0)), minimum_y=ymin, mean_h=hsum / (count * n), second_moment_h=hsum2 / (count * n), minimum_log_variance=vmin, maximum_log_variance=vmax, integral_internal_error=integral_error, n_paths=count, conditional_on_single_history=True, terminal_forward_zscore=float(zscore[-1]), history_length_years=simulation.get('history_length', simulation['history_multiple'] * simulation['horizon']))
    assert integral_error < simulation['integral_check_tolerance']
    history = pd.DataFrame(dict(T=times, history_contribution=hist))
    metadata.update(brownian_subcells=_BROWNIAN_SUBCELLS, brownian_modes_per_cell=_BROWNIAN_SUBCELLS + 1, brownian_quadrature_order=_BROWNIAN_QUADRATURE_ORDER, brownian_scheme='subcell_projection_plus_permanent_singular_residual', brownian_covariance_status='APPROXIMATE_FINITE_PROJECTION', brownian_history_corrected=True)
    metadata.update(history_mode=simulation.get('history_mode','fixed'),history_seed=simulation.get('history_seed',simulation['seed']),
        initial_main_y=float(yh[-1]),initial_main_y_drift=float(model['kappa']*(model['theta']-max(yh[-1],0.))),
        coefficient_h_initial=float(coefficient(model['H0'])),coefficient_gamma_denominator_power=1)
    if conditional_enabled:
        metadata['conditional_mc']=conditional
        metadata['conditional_estimator']='exact_discrete_conditional_lognormal_with_constant_volatility_proxy'
    return (samples, diagnostics, metadata, history)

def build_surface(samples,market,grid,simulation,pricing):
    """Geometric Black surface from raw simulated terminal prices."""
    rows=[]
    for j,T in enumerate(grid['maturities']):
        F=market['spot']*np.exp((market['rate']-market['dividend'])*T)
        for ratio in grid['moneyness']:
            K=F*ratio;Y=np.maximum(samples[:,j]-K,0.)
            price=float(Y.mean());se=float(Y.std(ddof=1)/np.sqrt(len(Y)))
            vol,status,res=implied_vol(price,F,K,T,'black',pricing)
            rows.append(dict(model='mpce',T=T,moneyness=ratio,K=K,F=F,undiscounted_call=price,
                             black_iv=vol,black_status=status,price_se=se,exercise_count=int(np.count_nonzero(Y)),black_repricing_residual=res))
    return pd.DataFrame(rows)


def verify_reference(model, simulation, market, grid, verification):
    """Explicitly inconclusive sparse points; no division by zero or false pass.
    Legacy z gate retained only as a heuristic on sufficiently sampled points.
    """
    from scipy.stats import binomtest
    from scipy.special import ndtr
    from rough_heston import black_call
    sim=dict(simulation,n_paths=verification['mc_test_paths'],
             batch_size=verification['mc_test_batch'],seed=verification['mc_test_seed'],conditional_mc=False)
    constant=dict(model,y_initial=model['theta'],y_diffusion=0.)
    samples,diag,meta,history=simulate(constant,sim,market,grid)
    sigma=np.exp(.5*model['initial_log_variance'])
    if not np.isfinite(sigma) or sigma<=0:
        raise FloatingPointError('Volatilite de reference sous-flux/non finie : aucun quotient calcule.')
    rows=[]; M=len(grid['maturities'])*len(grid['moneyness'])
    for j,T in enumerate(grid['maturities']):
        F=market['spot']*np.exp((market['rate']-market['dividend'])*T)
        for ratio in grid['moneyness']:
            K=ratio*F; Y=np.maximum(samples[:,j]-K,0.); n=len(Y)
            mu=black_call(F,K,T,sigma); a=sigma*np.sqrt(T)
            d2=np.log(F/K)/a-.5*a; p=float(ndtr(d2))
            mean=float(Y.mean()); sd=float(Y.std(ddof=1)); se=sd/np.sqrt(n)
            count=int(np.count_nonzero(Y)); z=(mean-mu)/se if se>0 else np.nan
            bp=float(binomtest(count,n,p).pvalue)
            second=F*F*np.exp(a*a)*ndtr(d2+2*a)-2*F*K*ndtr(d2+a)+K*K*p
            variance=second-mu*mu
            bound=np.sqrt(variance/(n*(verification.get('reference_family_alpha',.01)/(2*M)))) if variance>0 else np.nan
            status='passed'
            if not np.isfinite([mean,se,mu,bound]).all() or variance<=0:status='numerical_failure'
            elif bp<verification.get('reference_family_alpha',.01)/(2*M) or abs(mean-mu)>bound:status='failed'
            elif se==0 and count>0:status='numerical_failure'
            elif count<verification.get('reference_min_exercises',30) or se==0:status='insufficient_sampling'
            elif abs(z)>=verification['mc_reference_z_limit']:status='failed'
            rows.append(dict(T=T,moneyness=ratio,K=K,F=F,n_paths=n,
                exercise_count=count,distinct_payoffs=len(np.unique(Y)),
                distinct_underlyings=len(np.unique(samples[:,j])),payoff_min=float(Y.min()),
                payoff_max=float(Y.max()),sample_mean=mean,sample_std=sd,price_se=se,
                analytical_price=mu,numerator=mean-mu,zscore=z,
                zscore_status='finite' if se>0 else ('zero_over_zero' if mean==mu else 'nonzero_over_zero'),
                exercise_probability=p,expected_exercises=n*p,
                probability_no_exercises=float(np.exp(n*np.log1p(-p))),
                binomial_pvalue=bp,population_variance=variance,mean_error_bound=bound,status=status))
    points=pd.DataFrame(rows)
    # Separate singular-kernel moment gate, preserving its own tolerance.
    from scipy.integrate import quad
    dt=simulation['horizon']/simulation['n_steps']; errors=[]
    for h in verification['kernel_test_h_values']:
        integral=quad(lambda u:u**(h-.5),0,dt)[0]
        square=quad(lambda u:u**(2*h-1),0,dt)[0]
        errors.append(abs(integral/np.sqrt(square*dt)-np.sqrt(2*h)/(h+.5)))
    kernel_error=float(max(errors))
    tiny=dict(sim,n_paths=verification['batch_test_paths'],batch_size=verification['batch_test_size_a'])
    aa=simulate(model,tiny,market,grid)[0]
    bb=simulate(model,dict(tiny,batch_size=verification['batch_test_size_b']),market,grid)[0]
    batch_error=float(np.max(abs(aa-bb)))
    states=set(points.status)
    status=next((s for s in ['numerical_failure','failed','insufficient_sampling'] if s in states),'passed')
    report=dict(status=status,all_passed=states=={'passed'},
        singular_kernel_correlation_error=kernel_error,
        kernel_passed=kernel_error<verification['kernel_moment_tolerance'],
        batch_invariance_error=batch_error,batch_passed=batch_error<verification['batch_tolerance'],
        constant_integral_error=meta['integral_internal_error'],
        method='legacy_z_heuristic_plus_Chebyshev_exact_binomial_and_explicit_coverage_guard',
        limitations='Not a calibrated simultaneous mean test; no general MPCE convergence proof.')
    return report,points


@njit(cache=True,parallel=True)
def conditional_terms(z,v,dt,spot,carry,regression,independent_variance,initial_log_variance):
    """Integrate out the underlying's independent Gaussian component exactly.
    v depends only on the first two Gaussian drivers and permanent bridge modes.
    Left-point volatility is retained; this does not change the discrete model.
    """
    p,n=z.shape[1:]
    cf=np.empty((p,n+1));cv=np.empty_like(cf);pf=np.empty_like(cf);pv=np.empty_like(cf)
    sigma0=np.exp(.5*initial_log_variance)
    for q in prange(p):
        cf[q,0]=spot;pf[q,0]=spot;cv[q,0]=0.;pv[q,0]=0.
        I=0.;Q=0.;Q0=0.
        for i in range(1,n+1):
            sigma=np.exp(.5*v[q,i-1])
            conditional_shock=regression[0]*z[0,q,i-1]+regression[1]*z[1,q,i-1]
            I+=sigma*sigma*dt
            Q+=sigma*np.sqrt(dt)*conditional_shock
            Q0+=sigma0*np.sqrt(dt)*conditional_shock
            B=independent_variance*I;I0=sigma0*sigma0*i*dt;B0=independent_variance*I0
            cf[q,i]=spot*np.exp(carry*i*dt-.5*I+Q+.5*B)
            cv[q,i]=B
            pf[q,i]=spot*np.exp(carry*i*dt-.5*I0+Q0+.5*B0)
            pv[q,i]=B0
    return cf,cv,pf,pv


def conditional_call_and_derivative(Fc,B,K):
    """Conditional undiscounted call and derivative w.r.t. log strike."""
    from scipy.special import ndtr
    Fc=np.asarray(Fc,float);B=np.asarray(B,float)
    if not np.isfinite(Fc).all() or not np.isfinite(B).all() or np.any(Fc<=0) or np.any(B<=0):
        raise FloatingPointError('Conditional forward/variance invalid.')
    a=np.sqrt(B);d2=np.log(Fc/K)/a-.5*a
    return Fc*ndtr(d2+a)-K*ndtr(d2),-K*ndtr(d2)


def estimate_point(samples,conditional,j,F,K,T,initial_log_variance):
    """Price and strike-derivative observations, preserving raw diagnostics.
    Constant-vol proxy has an analytic expectation. No fitted control coefficient,
    seed selection, clipping, path rescaling or parameter calibration is used.
    """
    from rough_heston import black_call
    from scipy.special import ndtr
    raw=np.maximum(samples[:,j]-K,0.)
    raw_derivative=-K*(samples[:,j]>K)
    sigma0=np.exp(.5*initial_log_variance)
    if conditional is None:
        values=raw;derivatives=raw_derivative;method='raw_payoff'
    else:
        cc,dc=conditional_call_and_derivative(conditional['forward'][:,j],conditional['variance'][:,j],K)
        pc,pd=conditional_call_and_derivative(conditional['proxy_forward'][:,j],conditional['proxy_variance'][:,j],K)
        a=sigma0*np.sqrt(T);d2=np.log(F/K)/a-.5*a
        values=cc-pc+black_call(F,K,T,sigma0)
        derivatives=dc-pd-K*ndtr(d2)
        method='conditional_mc_with_exact_constant_vol_proxy'
    if conditional is not None and 'observation_weights' in conditional:
        obs_weights=conditional['observation_weights']
        values=values*obs_weights;derivatives=derivatives*obs_weights
        raw=raw*obs_weights;raw_derivative=raw_derivative*obs_weights
    n=len(raw)
    if conditional is not None and 'history_group' in conditional:
        groups=conditional['history_group']; group_ids=np.unique(groups)
        # Undo weighting to estimate variance within each fixed historical group.
        price_se=np.sqrt(sum(np.var(values[groups==g]/conditional['observation_weights'][groups==g],ddof=1)/np.sum(groups==g) for g in group_ids))/len(group_ids)
    else: price_se=values.std(ddof=1)/np.sqrt(n)
    return dict(price=float(values.mean()),derivative=float(derivatives.mean()),
                price_se=float(price_se),values=values,derivatives=derivatives,uncertainty_context=conditional,
                raw_price=float(raw.mean()),raw_price_se=conditional_standard_error(raw,conditional),
                raw_values=raw,raw_derivatives=raw_derivative,exercise_count=int(np.count_nonzero(raw)),method=method)


def configure_numerics(simulation):
    global _BROWNIAN_SUBCELLS,_BROWNIAN_QUADRATURE_ORDER,_BROWNIAN_NODES,_BROWNIAN_WEIGHTS
    m=simulation.get('brownian_subcells',8);order=simulation.get('brownian_quadrature_order',64)
    if type(m) is not int or m<1 or type(order) is not int or order<2:raise ValueError('Projection brownienne invalide.')
    _BROWNIAN_SUBCELLS=m;_BROWNIAN_QUADRATURE_ORDER=order
    nodes,w=leggauss(order);_BROWNIAN_NODES=(nodes+1)/2;_BROWNIAN_WEIGHTS=w/2


def simulate(model, simulation, market, grid):
    """Ensemble d'historiques distincts, sans normalisation de la variance.

    n_histories = ceil(n_paths * history_paths_percent / 100).
    Chaque historique porte le même poids ; les erreurs conditionnelles ne
    mesurent pas l'incertitude entre historiques. Le pricing est une moyenne
    sur l'ensemble fini d'historiques, pas le prix conditionnel original.
    """
    from copy import deepcopy
    import math
    percent = simulation.get('history_paths_percent', 100.0 / simulation['n_paths'])
    if not np.isfinite(percent) or not 0 <= percent <= 100:
        raise ValueError('history_paths_percent doit être compris entre 0 et 100.')
    n = simulation['n_paths']
    if percent == 0:
        local = dict(simulation, history_mode='zero')
        result = simulate_single_history(model, local, market, grid)
        result[2].update(n_histories=0, history_paths_percent=0., pricing_conditioning='zero_history')
        return result
    count = int(math.ceil(n * percent / 100.))
    if count > n // 2:
        raise ValueError('Au moins deux futurs par historique : réduire le pourcentage (maximum 50%).')
    sizes = [n // count + (i < n % count) for i in range(count)]
    sample_parts=[]; conditional_parts=[]; diagnostics=[]; histories=[]; metadata=[]
    for i, size in enumerate(sizes):
        local = dict(simulation, n_paths=int(size), history_mode='fixed')
        if count > 1:
            local['history_seed'] = int(np.random.SeedSequence([simulation.get('history_seed',simulation['seed']), 73, i]).generate_state(1)[0])
            local['seed'] = int(np.random.SeedSequence([simulation['seed'], 97, i]).generate_state(1)[0])
        samples, diag, meta, hist = simulate_single_history(model, local, market, grid)
        conditional_parts.append(meta.pop('conditional_mc', None))
        sample_parts.append(samples); diagnostics.append(diag); histories.append(hist); metadata.append(meta)
    # Equal historical weights even when the allocation differs by one future.
    if count == 1:
        meta=metadata[0];meta.update(n_histories=1,history_paths_percent=float(percent),history_allocation=sizes,pricing_conditioning='single_fixed_history')
        if conditional_parts[0] is not None:meta['conditional_mc']=conditional_parts[0]
        return sample_parts[0],diagnostics[0],meta,histories[0]
    weights = np.concatenate([np.full(size, n / (count * size)) for size in sizes])
    samples = np.concatenate(sample_parts)
    diag = diagnostics[0].copy()
    for col in diag.columns:
        if col != 'T': diag[col] = np.mean([x[col].to_numpy() for x in diagnostics], axis=0)
    # A mean of z-scores or standard errors is not an ensemble standard error.
    diag['mean_X_se'] = np.nan; diag['forward_zscore'] = np.nan
    diag['log_variance_population_variance'] = np.nan
    hist = histories[0].copy()
    hist['history_contribution'] = np.mean([x.history_contribution.to_numpy() for x in histories],axis=0)
    meta = dict(n_paths=n,n_histories=count,history_paths_percent=float(percent),
        history_length_years=float(simulation.get('history_length',1.)),
        pricing_conditioning='single_fixed_history' if count==1 else 'equal_weight_finite_history_ensemble',
        conditional_on_single_history=count==1,history_allocation=sizes,
        histories_metadata=metadata,observation_weights=weights,
        ensemble_uncertainty_status='NOT_ESTABLISHED_BY_IID_PATH_STANDARD_ERRORS')
    if conditional_parts[0] is not None:
        meta['conditional_mc']={key:np.concatenate([x[key] for x in conditional_parts]) for key in conditional_parts[0]}
        meta['conditional_mc']['observation_weights']=weights
        meta['conditional_mc']['history_group']=np.repeat(np.arange(count),sizes)
    return samples,diag,meta,hist


def conditional_standard_error(values, context=None):
    """Erreur des futurs, conditionnelle à l'ensemble fini de passés."""
    if context is None or 'history_group' not in context:
        return float(np.std(values,ddof=1)/np.sqrt(len(values)))
    groups=context['history_group'];weights=context['observation_weights'];ids=np.unique(groups)
    return float(np.sqrt(sum(np.var(values[groups==g]/weights[groups==g],ddof=1)/np.sum(groups==g) for g in ids))/len(ids))
