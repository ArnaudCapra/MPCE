"""Rough Heston : Riccati fractionnaire implicite et inversion de Lewis.
xi multiplie sqrt(V)dB ; temps en années ; prix non actualisés.
"""
import numpy as np
import pandas as pd
from scipy.special import gamma, ndtr, roots_legendre
from scipy.optimize import brentq

def validate_market(market,grid):
    if not np.isfinite(list(market.values())).all() or market['spot']<=0:
        raise ValueError('Marché invalide.')
    t=np.asarray(grid['maturities'],float); k=np.asarray(grid['moneyness'],float)
    for a in (t,k):
        if a.ndim!=1 or not len(a) or not np.isfinite(a).all() or np.any(a<=0) or np.any(np.diff(a)<=0):
            raise ValueError('Grilles positives strictement croissantes requises.')
    if len(k)<3 or not np.any(k==1.): raise ValueError('Trois strikes minimum et ATM exact requis.')
    return t,k

def black_call(forward,strike,maturity,volatility):
    if volatility==0: return max(forward-strike,0.)
    a=volatility*np.sqrt(maturity); d=np.log(forward/strike)/a+.5*a
    return forward*ndtr(d)-strike*ndtr(d-a)

def implied_vol(price,forward,strike,maturity,convention,controls):
    if convention != 'black': raise ValueError('Convention inconnue.')
    vals=[controls[k] for k in ['initial_upper','maximum_upper','root_xtol','root_rtol','root_maxiter']]
    if not np.isfinite(vals).all() or min(vals)<=0 or controls['maximum_upper']<controls['initial_upper']:
        raise ValueError('Contrôles de racine invalides.')
    if controls['root_rtol']<4*np.finfo(float).eps or type(controls['root_maxiter']) is not int: raise ValueError('rtol/maxiter invalides.')
    if not np.isfinite([price,forward,strike,maturity]).all() or min(forward,strike,maturity)<=0: return np.nan,'invalid_input',np.nan
    intrinsic=max(forward-strike,0.)
    if price<intrinsic: return np.nan,'below_intrinsic',np.nan
    if convention=='black' and price>=forward: return np.nan,'above_black_bound',np.nan
    if price==intrinsic: return 0.,'at_intrinsic',0.
    fn=black_call; hi=controls['initial_upper']
    while fn(forward,strike,maturity,hi)<price and hi<controls['maximum_upper']: hi=min(2*hi,controls['maximum_upper'])
    if fn(forward,strike,maturity,hi)<price: return np.nan,'unbracketed',np.nan
    value,result=brentq(lambda s:fn(forward,strike,maturity,s)-price,0.,hi,xtol=controls['root_xtol'],rtol=controls['root_rtol'],maxiter=controls['root_maxiter'],full_output=True,disp=False)
    if not result.converged: return np.nan,'root_not_converged',np.nan
    return value,'ok',fn(forward,strike,maturity,value)-price

def characteristic_function(z,maturity,model,steps):
    if not np.isfinite(list(model.values())).all(): raise ValueError('Paramètres non finis.')
    if not 0<model['H']<=.5 or min(model['kappa'],model['theta'],model['v0'])<=0 or model['xi']<0 or abs(model['rho'])>1: raise ValueError('Paramètres rough Heston invalides.')
    if type(steps) is not int or steps<2 or not np.isfinite(maturity) or maturity<=0: raise ValueError('Discrétisation invalide.')
    z=np.atleast_1d(np.asarray(z,complex))
    if z.ndim!=1 or not np.isfinite(z).all(): raise ValueError('Fréquences invalides.')
    alpha=model['H']+.5; dt=maturity/steps
    constant=-.5*(z*z+1j*z); linear=1j*z*model['rho']*model['xi']-model['kappa']; quadratic=.5*model['xi']**2
    h=np.zeros((steps+1,len(z)),complex); f=np.zeros_like(h); f[0]=constant
    scale=dt**alpha/gamma(alpha+2); powers=np.arange(steps+2,dtype=float)**(alpha+1)
    weights=powers[2:]+powers[:-2]-2*powers[1:-1]
    for m in range(1,steps+1):
        first=(m-1)**(alpha+1)-(m-1-alpha)*m**alpha
        history=scale*(first*f[0]+np.sum(weights[:m-1,None]*f[m-1:0:-1],axis=0))
        rhs=history+scale*constant; b=1-scale*linear
        if quadratic==0: h[m]=rhs/b
        else:
            disc=np.sqrt(b*b-4*scale*quadratic*rhs)
            # Branche connectée à xi=0 : ne pas basculer vers l'autre racine.
            disc=np.where(np.real(disc*np.conj(b))<0,-disc,disc)
            denominator=b+disc
            if np.any(abs(denominator)<np.finfo(float).tiny): raise FloatingPointError('Racine Riccati singulière.')
            h[m]=2*rhs/denominator
        f[m]=constant+linear*h[m]+quadratic*h[m]**2
    exponent=model['kappa']*model['theta']*np.trapz(h,dx=dt,axis=0)+model['v0']*np.trapz(f,dx=dt,axis=0)
    result=np.exp(exponent)
    if not np.isfinite(h).all() or not np.isfinite(result).all(): raise FloatingPointError('Riccati non finie.')
    return result

def characteristic_function_pece(z,maturity,model,steps):
    z=np.atleast_1d(np.asarray(z,complex)); alpha=model['H']+.5; dt=maturity/steps
    constant=-.5*(z*z+1j*z); linear=1j*z*model['rho']*model['xi']-model['kappa']; q=.5*model['xi']**2
    h=np.zeros((steps+1,len(z)),complex); f=np.zeros_like(h); f[0]=constant; scale=dt**alpha/gamma(alpha+2)
    for m in range(1,steps+1):
        lag=np.arange(m,0,-1,dtype=float)
        predictor=dt**alpha/gamma(alpha+1)*np.sum((lag**alpha-(lag-1)**alpha)[:,None]*f[:m],axis=0)
        fp=constant+linear*predictor+q*predictor**2
        first=(m-1)**(alpha+1)-(m-1-alpha)*m**alpha
        if m>1:
            ell=np.arange(m-1,0,-1,dtype=float); weights=(ell+1)**(alpha+1)+(ell-1)**(alpha+1)-2*ell**(alpha+1)
            history=np.sum(weights[:,None]*f[1:m],axis=0)
        else: history=0.
        h[m]=scale*(first*f[0]+history+fp); f[m]=constant+linear*h[m]+q*h[m]**2
    return np.exp(model['kappa']*model['theta']*np.trapz(h,dx=dt,axis=0)+model['v0']*np.trapz(f,dx=dt,axis=0))

def normalized_calls(maturity,moneyness,model,numerical):
    if not np.isfinite(numerical['fourier_upper']) or numerical['fourier_upper']<=0 or type(numerical['fourier_nodes']) is not int or numerical['fourier_nodes']<2: raise ValueError('Quadrature invalide.')
    nodes,weights=roots_legendre(numerical['fourier_nodes']); u=(nodes+1)*numerical['fourier_upper']/2; weights=weights*numerical['fourier_upper']/2
    phi=characteristic_function(u-.5j,maturity,model,numerical['riccati_steps'])
    integrand=np.real(np.exp(-1j*np.outer(np.log(moneyness),u))*phi)/(u*u+.25)
    return 1-np.sqrt(moneyness)/np.pi*(integrand@weights)

def build_surface(model,numerical,market,grid,pricing):
    ts,ks=validate_market(market,grid); rows=[]
    for t in ts:
        forward=market['spot']*np.exp((market['rate']-market['dividend'])*t)
        for k,price in zip(ks,forward*normalized_calls(float(t),ks,model,numerical)):
            strike=forward*k; biv,status,res=implied_vol(price,forward,strike,t,'black',pricing)
            if status not in ('ok','at_intrinsic'): raise ArithmeticError(f'Rough Heston : {status}, T={t}, K/F={k}')
            rows.append(dict(model='rough_heston',T=t,moneyness=k,K=strike,F=forward,undiscounted_call=price,discounted_call=np.exp(-market['rate']*t)*price,black_iv=biv,black_status=status,black_repricing_residual=res))
    return pd.DataFrame(rows)

def verify_rough(model,numerical,market,grid,pricing,verification):
    t=verification['reference_maturity']; ks=np.asarray(grid['moneyness'])
    constant=dict(model,xi=0.,v0=model['theta'])
    bs=float(np.max(abs(normalized_calls(t,ks,constant,numerical)-np.array([black_call(1.,k,t,np.sqrt(constant['theta'])) for k in ks]))))
    z=np.array([.1,1.,3.])-.5j; steps=verification['reference_steps']
    pece=float(np.max(abs(characteristic_function(z,t,model,steps)-characteristic_function_pece(z,t,model,steps))))
    classic=dict(model,H=.5); b=classic['kappa']-1j*z*classic['rho']*classic['xi']
    if classic['xi']==0: classic['xi']=.1
    b=classic['kappa']-1j*z*classic['rho']*classic['xi']; d=np.sqrt(b*b+classic['xi']**2*(z*z+1j*z)); g=(b-d)/(b+d)
    D=(b-d)/classic['xi']**2*(1-np.exp(-d*t))/(1-g*np.exp(-d*t)); C=classic['kappa']*classic['theta']/classic['xi']**2*((b-d)*t-2*np.log((1-g*np.exp(-d*t))/(1-g)))
    classic_error=float(np.max(abs(characteristic_function(z,t,classic,steps)-np.exp(C+classic['v0']*D))))
    invariant=float(np.max(abs(characteristic_function([0,-1j],t,model,steps)-1)))
    symmetry=float(np.max(abs(characteristic_function(z.real,t,model,steps)-np.conj(characteristic_function(-z.real,t,model,steps)))))
    base=build_surface(model,numerical,market,grid,pricing)
    fine=build_surface(model,dict(numerical,riccati_steps=numerical['riccati_steps']*2),market,grid,pricing)
    fourier=build_surface(model,dict(numerical,fourier_upper=numerical['fourier_upper']*1.5,fourier_nodes=numerical['fourier_nodes']*2),market,grid,pricing)
    report=dict(black_scholes_price_error=bs,classical_heston_cf_error=classic_error,independent_pece_cf_error=pece,normalization_error=invariant,conjugate_symmetry_error=symmetry,time_refinement_max_iv_bp=float(abs(base.black_iv-fine.black_iv).max()*1e4),fourier_refinement_max_iv_bp=float(abs(base.black_iv-fourier.black_iv).max()*1e4))
    limits={'black_scholes_price_error':verification['analytic_price_tolerance'],'classical_heston_cf_error':verification['analytic_cf_tolerance'],'independent_pece_cf_error':verification['pece_cf_tolerance'],'normalization_error':verification['invariant_tolerance'],'conjugate_symmetry_error':verification['invariant_tolerance'],'time_refinement_max_iv_bp':verification['rough_iv_tolerance_bp'],'fourier_refinement_max_iv_bp':verification['rough_iv_tolerance_bp']}
    report['checks']={k:dict(value=report[k],tolerance=v,passed=report[k]<v) for k,v in limits.items()}
    report['all_passed']=all(x['passed'] for x in report['checks'].values())
    # Un échec est conservé et publié comme échec, jamais converti en succès.
    return report



def price_and_logstrike_derivative(T, ratios, model, numerical):
    """Normalized undiscounted call and derivative with respect to log(K/F).
    Derivative of the same Lewis integral, not a finite-difference derivative.
    """
    ratios=np.asarray(ratios,float)
    nodes,weights=roots_legendre(numerical['fourier_nodes'])
    u=(nodes+1)*numerical['fourier_upper']/2
    weights=weights*numerical['fourier_upper']/2
    phi=characteristic_function(u-.5j,T,model,numerical['riccati_steps'])
    phase=np.exp(-1j*np.outer(np.log(ratios),u))*phi
    I=(phase.real/(u*u+.25))@weights
    Ip=(((-1j*u)*phase).real/(u*u+.25))@weights
    root=np.sqrt(ratios)
    return 1-root/np.pi*I, -root/np.pi*(.5*I+Ip)


# Sensibilités des prix aux paramètres : exclusivement Fourier/Riccati.
def price_payoff_fourier(maturity, ratios, model, numerical, market, payoff=None):
    """Prix actualisé de call, put ou portefeuille fini de vanilles européennes.

    payoff={'type':'call'|'put','notional':1.0}, ou
    {'type':'portfolio','legs':[{'type':'call','weight':1.,'strike_multiplier':1.}, ...]}.
    Chaque ratio désigne un strike de référence K/F ; les legs utilisent K*multiplier.
    Pas de Monte Carlo, pas de substitution Black, pas de payoff arbitraire exécuté.
    """
    payoff={'type':'call','notional':1.0} if payoff is None else dict(payoff)
    ratios=np.asarray(ratios,float)
    if ratios.ndim!=1 or not len(ratios) or not np.isfinite(ratios).all() or np.any(ratios<=0):
        raise ValueError('Ratios K/F positifs finis requis.')
    if not np.isfinite(maturity) or maturity<=0:raise ValueError('Maturité positive finie requise.')
    if not np.isfinite([market['spot'],market['rate'],market['dividend']]).all() or market['spot']<=0:
        raise ValueError('Marché invalide.')
    for key in ['riccati_steps','fourier_nodes']:
        if type(numerical[key]) is not int or numerical[key]<2:raise ValueError(key+' entier >=2 requis.')
    if not np.isfinite(numerical['fourier_upper']) or numerical['fourier_upper']<=0:raise ValueError('Borne Fourier positive requise.')
    notional=payoff.get('notional',1.)
    if not np.isfinite(notional):raise ValueError('Notional fini requis.')
    kind=payoff.get('type')
    if kind in ('call','put'):
        legs=[dict(type=kind,weight=1.,strike_multiplier=1.)]
    elif kind=='portfolio':
        legs=payoff.get('legs',[])
        if not isinstance(legs,list) or not legs:raise ValueError('Portefeuille non vide requis.')
    else:raise ValueError('Payoff supporté : call, put, portfolio de vanilles européennes.')
    forward=market['spot']*np.exp((market['rate']-market['dividend'])*maturity)
    discount=np.exp(-market['rate']*maturity)
    all_ratios=[]
    for leg in legs:
        if leg.get('type') not in ('call','put'):raise ValueError('Leg call ou put requis.')
        mult=leg.get('strike_multiplier',1.);weight=leg.get('weight',1.)
        if not np.isfinite([mult,weight]).all() or mult<=0:raise ValueError('Leg invalide.')
        all_ratios.append(ratios*mult)
    unique=np.unique(np.concatenate(all_ratios))
    calls=forward*normalized_calls(maturity,unique,model,numerical)
    result=np.zeros(len(ratios))
    for leg,leg_ratios in zip(legs,all_ratios):
        value=calls[np.searchsorted(unique,leg_ratios)]
        if leg['type']=='put':value=value-forward+forward*leg_ratios
        result+=leg.get('weight',1.)*value
    result*=discount*notional
    if not np.isfinite(result).all():raise FloatingPointError('Prix de payoff non finis.')
    return result


def parameter_greeks_fourier(model,numerical,market,grid,controls=None,payoff=None):
    """dP/dp, p in H,xi,kappa,theta, à S0,r,q,T,K et autres paramètres fixes.

    Différence centrée ordre 2 à h et h/2 ; au bord, différence unilatérale
    ordre 2 avec stencil déclaré. greek est l'estimation h/2, sans extrapolation.
    fd_error_estimate=|D(h/2)-D(h)|/3 : diagnostic asymptotique, non borne garantie.
    Pas absolus explicites. Valeurs par unité absolue de paramètre et par bump.
    """
    controls={} if controls is None else dict(controls)
    steps=controls.get('steps',{'H':.001,'xi':.001,'kappa':.005,'theta':.0001})
    atol=controls.get('fd_abs_tolerance',1e-4);rtol=controls.get('fd_rel_tolerance',.01)
    if not np.isfinite([atol,rtol]).all() or min(atol,rtol)<0:raise ValueError('Tolérances FD invalides.')
    ts=np.asarray(grid['maturities'],float);ratios=np.asarray(grid.get('moneyness',grid.get('strikes')),float)
    for values in [ts,ratios]:
        if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or np.any(values<=0) or np.any(np.diff(values)<=0):
            raise ValueError('Grilles positives strictement croissantes requises.')
    parameters=('H','xi','kappa','theta');bounds={'H':(0.,.5),'xi':(0.,np.inf),'kappa':(0.,np.inf),'theta':(0.,np.inf)}
    for param in parameters:
        if param not in steps or not np.isfinite(steps[param]) or steps[param]<=0:raise ValueError('Pas absolu positif requis : '+param)
    rows=[]
    for T in ts:
        cache={}
        def price(param=None,offset=0.):
            key=(param,float(offset))
            if key not in cache:
                shifted=dict(model)
                if param is not None:shifted[param]+=offset
                cache[key]=price_payoff_fourier(float(T),ratios,shifted,numerical,market,payoff)
            return cache[key]
        base=price();F=market['spot']*np.exp((market['rate']-market['dividend'])*T)
        for param in parameters:
            h=float(steps[param]);x=model[param];lower,upper=bounds[param]
            admissible=lambda v: v>=lower if param=='xi' else v>lower
            if admissible(x-h) and x+h<=upper:scheme='central_2';direction=0
            elif x+2*h<=upper:scheme='forward_2';direction=1
            elif admissible(x-2*h):scheme='backward_2';direction=-1
            else:raise ValueError('Pas trop large pour le domaine : '+param)
            def derivative(step):
                if direction==0:return (price(param,step)-price(param,-step))/(2*step)
                return direction*(-3*base+4*price(param,direction*step)-price(param,2*direction*step))/(2*step)
            coarse=derivative(h);fine=derivative(h/2);error=abs(fine-coarse)/3
            tolerance=atol+rtol*abs(fine)
            for j,ratio in enumerate(ratios):
                rows.append(dict(T=float(T),moneyness=float(ratio),K=float(F*ratio),F=float(F),parameter=param,
                    parameter_value=float(x),discounted_price=float(base[j]),greek=float(fine[j]),
                    greek_coarse=float(coarse[j]),bump=h,bump_fine=h/2,
                    linear_pnl_for_bump=float(fine[j]*h),fd_error_estimate=float(error[j]),
                    fd_tolerance=float(tolerance[j]),fd_stable=bool(error[j]<=tolerance[j]),
                    stencil=scheme,pricing_method='Fourier_Lewis_Riccati',
                    units='discounted_price_per_absolute_parameter_unit',
                    payoff_type=('call' if payoff is None else payoff['type'])))
    return pd.DataFrame(rows)
