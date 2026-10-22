"""Offline interactive surfaces, ATM skew, finite-difference checks and BIC fits."""
from pathlib import Path
from fractions import Fraction
from functools import reduce
import math, json, copy, hashlib, platform, importlib.metadata, shutil, tempfile, os
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import rough_heston as rh
import mpce


def json_safe(x):
    if isinstance(x,dict):return {str(k):json_safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [json_safe(v) for v in x]
    if isinstance(x,np.generic):return json_safe(x.item())
    if isinstance(x,float) and not np.isfinite(x):return 'NaN' if np.isnan(x) else ('+Infinity' if x>0 else '-Infinity')
    return x


def write_json(path,value):
    path=Path(path)
    if path.parent.name!='json':path=path.parent/'json'/path.name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(json_safe(value),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def prepare_configuration(config):
    c=copy.deepcopy(config)
    ts,ks=rh.validate_market(c['market'],c['grid'])
    if c['skew']['coordinate']!='log_moneyness' or c['skew']['primary']!='digital_implicit':
        raise ValueError('Seules log_moneyness et digital_implicit sont implementees.')
    if type(c['skew']['bic_min_points_per_segment']) is not int or c['skew']['bic_min_points_per_segment']<3:
        raise ValueError('Au moins trois points par segment BIC requis.')
    if not np.isfinite(c['skew']['confidence_z']) or c['skew']['confidence_z']<=0:
        raise ValueError('Quantile de confiance positif fini requis.')
    ds=np.asarray(c['skew']['distances'],float)
    if ds.ndim!=1 or not len(ds) or not np.isfinite(ds).all() or np.any(ds<=0) or np.any(np.diff(ds)<=0):
        raise ValueError('Distances de skew positives et strictement croissantes requises.')
    horizon=max(float(c['simulation']['horizon']),float(ts[-1]))
    den=[]
    for t in ts:
        ratio=float(t/horizon); f=Fraction(ratio).limit_denominator(10000)
        if abs(float(f)-ratio)>c['simulation']['alignment_tolerance']/horizon:
            raise ValueError('Maturite non representable sur la grille uniforme : modifier la grille.')
        den.append(f.denominator)
    multiple=reduce(math.lcm,den,1)
    original_requested=c['simulation']['n_steps']
    requested=original_requested
    min_steps=c['simulation'].get('min_steps_shortest_maturity',1)
    if type(min_steps) is not int or min_steps<1:raise ValueError('Nombre minimum de pas invalide.')
    requested=max(requested,math.ceil(min_steps*horizon/float(ts[0])))
    steps=math.ceil(requested/multiple)*multiple
    if steps>c['simulation'].get('maximum_aligned_steps',2000):
        raise ValueError(f'Alignement exige {steps} pas ; limite depassee, modifier la grille.')
    c['simulation'].update(horizon=horizon,n_steps=steps)
    c['simulation']['integral_check_indices']=sorted(set([1,steps]+np.rint(ts/horizon*steps).astype(int).tolist()))
    c['alignment']=dict(requested_steps=original_requested,effective_steps=steps,dt=horizon/steps,steps_shortest_maturity=int(round(float(ts[0])*steps/horizon)),time_convergence_status='NOT_ESTABLISHED_BY_ALIGNMENT')
    return c


def invert(price,F,K,T,controls):
    return rh.implied_vol(float(price),float(F),float(K),float(T),'black',controls)


def black_vega(F,K,T,sigma):
    if not np.isfinite(sigma) or sigma<=0:return np.nan
    a=sigma*np.sqrt(T); d1=np.log(F/K)/a+.5*a
    return F*np.exp(-.5*d1*d1)/np.sqrt(2*np.pi)*np.sqrt(T)


def skew_from_price_derivative(price,derivative,F,K,T,controls):
    sigma,status,res=invert(price,F,K,T,controls)
    vega=black_vega(F,K,T,sigma)
    if status!='ok' or not np.isfinite(vega) or vega<=0:return sigma,np.nan,vega,status
    d2=np.log(F/K)/(sigma*np.sqrt(T))-.5*sigma*np.sqrt(T)
    return sigma,(derivative+K*ndtr(d2))/vega,vega,status


def skew_influence(estimate,F,T,controls):
    price=estimate['price'];derivative=estimate['derivative']
    sigma,skew,vega,status=skew_from_price_derivative(price,derivative,F,F,T,controls)
    if not np.isfinite(skew):return sigma,skew,np.nan,None,status
    def g(vol):
        return (derivative+F*ndtr(-.5*vol*np.sqrt(T)))/black_vega(F,F,T,vol)
    eps=max(sigma*1e-5,1e-7)
    dg=(g(sigma+eps)-g(sigma-eps))/(2*eps)
    influence=dg*(estimate['values']-price)/vega+(estimate['derivatives']-derivative)/vega
    se=mpce.conditional_standard_error(influence,estimate.get('uncertainty_context'))
    return sigma,skew,se,influence,status


def build_outputs(c,samples,conditional=None):
    market=c['market'];controls=c['pricing'];ts=c['grid']['maturities'];ds=c['skew']['distances']
    ratios=np.unique(np.r_[c['grid']['moneyness'],1.,np.exp(-np.asarray(ds)),np.exp(ds)])
    surfaces=[];skews=[];checks=[]
    for j,T in enumerate(ts):
        F=market['spot']*np.exp((market['rate']-market['dividend'])*T)
        normalized,derivatives=rh.price_and_logstrike_derivative(T,ratios,c['rough_model'],c['rough_numerical'])
        for model in ['rough_heston','mpce']:
            vols={};influences={};statuses={};estimates={}
            for i,ratio in enumerate(ratios):
                K=F*ratio
                if model=='rough_heston':
                    price=float(F*normalized[i]);se=np.nan;count=None;raw_price=price;raw_se=np.nan;method='Lewis_integral'
                else:
                    estimate=mpce.estimate_point(samples,conditional,j,F,K,T,c['mpce_model']['initial_log_variance'])
                    estimates[ratio]=estimate
                    price=estimate['price'];se=estimate['price_se'];count=estimate['exercise_count']
                    raw_price=estimate['raw_price'];raw_se=estimate['raw_price_se'];method=estimate['method']
                vol,status,res=invert(price,F,K,T,controls)
                raw_vol,raw_status,_=invert(raw_price,F,K,T,controls)
                vega=black_vega(F,K,T,vol)
                vols[ratio]=vol;statuses[ratio]=status
                if model=='mpce' and np.isfinite(vega) and vega>0:
                    influences[ratio]=(estimate['values']-price)/vega
                sampled=model=='rough_heston' or conditional is not None or count>=30
                difference_se=np.nan;difference_z=np.nan
                if model=='mpce':
                    diff=estimate['raw_values']-estimate['values']
                    difference_se=mpce.conditional_standard_error(diff,conditional)
                    if difference_se>0:difference_z=float(diff.mean()/difference_se)
                surfaces.append(dict(model=model,T=T,moneyness=ratio,log_moneyness=np.log(ratio),K=K,F=F,
                    undiscounted_call=price,discounted_call=price*np.exp(-market['rate']*T),black_iv=vol,
                    black_status=status,repricing_residual=res,price_se=se,exercise_count=count,
                    raw_call=raw_price,raw_price_se=raw_se,raw_black_iv=raw_vol,raw_black_status=raw_status,
                    raw_minus_estimator_se=difference_se,raw_minus_estimator_z=difference_z,estimator=method,
                    display_black_iv=vol if sampled and status=='ok' else np.nan,
                    sampling_status='not_applicable' if model=='rough_heston' else ('conditional_estimator' if conditional is not None else ('sampled' if sampled else 'insufficient_sampling')),
                    requested_surface_point=bool(np.any(np.isclose(ratio,c['grid']['moneyness'],rtol=0,atol=1e-14)))))
            if model=='rough_heston':
                idx=int(np.flatnonzero(ratios==1.)[0])
                sigma,skew,_,status=skew_from_price_derivative(F*normalized[idx],F*derivatives[idx],F,F,T,controls)
                se_skew=np.nan;raw_skew=skew;raw_skew_se=np.nan;influence=None
            else:
                sigma,skew,se_skew,influence,status=skew_influence(estimates[1.],F,T,controls)
                raw=dict(estimates[1.])
                raw.update(price=raw['raw_price'],values=raw['raw_values'],derivatives=raw['raw_derivatives'],derivative=float(raw['raw_derivatives'].mean()))
                _,raw_skew,raw_skew_se,_,_=skew_influence(raw,F,T,controls)
            valid=np.isfinite(skew) and status=='ok'
            numerical_zero=bool(valid and abs(skew)<=c['skew'].get('zero_skew_tolerance',1e-10))
            significant=bool(valid and not numerical_zero and (model=='rough_heston' or abs(skew)>c['skew']['confidence_z']*se_skew))
            skews.append(dict(model=model,T=T,atm_iv=sigma,skew=skew,skew_se=se_skew,
                raw_skew=raw_skew,raw_skew_se=raw_skew_se,
                skew_ci_low=skew-c['skew']['confidence_z']*se_skew,
                skew_ci_high=skew+c['skew']['confidence_z']*se_skew,
                status=status,significant=significant,numerical_zero=numerical_zero,
                steps_to_maturity=int(round(T/c['alignment']['dt'])),confidence_z=c['skew']['confidence_z'],
                loglog_eligible=bool(valid and not numerical_zero and (significant or not c['skew']['bic_require_significance'])),
                derivative_method='Lewis_integral_derivative' if model=='rough_heston' else ('conditional_analytic_digital_with_proxy' if conditional is not None else 'empirical_digital_implicit_derivative')))
            for distance in ds:
                lo=float(np.exp(-distance));hi=float(np.exp(distance));slope=(vols[hi]-vols[lo])/(2*distance)
                fdse=np.nan;diffse=np.nan
                if model=='mpce' and lo in influences and hi in influences:
                    fd_influence=(influences[hi]-influences[lo])/(2*distance)
                    fdse=mpce.conditional_standard_error(fd_influence,conditional)
                    if influence is not None:diffse=mpce.conditional_standard_error(fd_influence-influence,conditional)
                error=slope-skew
                tol=c['skew']['distance_abs_tolerance']+c['skew']['distance_rel_tolerance']*abs(skew)
                if np.isfinite(diffse):tol+=c['skew']['confidence_z']*diffse
                checks.append(dict(model=model,T=T,distance=distance,K_minus=F*lo,K_plus=F*hi,
                    iv_minus=vols[lo],iv_plus=vols[hi],status_minus=statuses[lo],status_plus=statuses[hi],
                    centered_slope=slope,centered_slope_se=fdse,digital_skew=skew,difference=error,
                    difference_se=diffse,tolerance=tol,consistent=bool(np.isfinite(error) and abs(error)<=tol)))
    return pd.DataFrame(surfaces),pd.DataFrame(skews),pd.DataFrame(checks)


def fit_bic(T,skew,min_side=3):
    """Exploratory Gaussian OLS BIC. Continuous one-knot fit, exhaustive intervals.
    k=3 (intercept,slope,variance) versus k=5 (+slope change,knot).
    MC maturity errors are correlated: this is not inferential evidence of regimes.
    """
    T=np.asarray(T,float); skew=np.asarray(skew,float)
    if len(T)!=len(skew) or not np.isfinite(T).all() or not np.isfinite(skew).all() or np.any(T<=0) or np.any(skew==0):
        raise ValueError('BIC requires finite nonzero skew and positive maturities.')
    order=np.argsort(T); T=T[order]; skew=skew[order]
    if np.any(np.diff(T)<=0):raise ValueError('Unique maturities required.')
    x=np.log(T); y=np.log(np.abs(skew)); n=len(x)
    if n<3:return dict(status='insufficient_points',n=n),pd.DataFrame()
    A=np.c_[np.ones(n),x]; coef=np.linalg.lstsq(A,y,rcond=None)[0]
    pred=A@coef; rss=float(np.sum((y-pred)**2))
    exact_tol=1e-24*max(1.,float(np.sum(y*y)))
    if rss<=exact_tol:
        return dict(status='exact_single_line_BIC_undefined',selected='single',n=n,
                    slope_left=float(coef[1]),slope_right=None,break_T=None,intercept=float(coef[0]),slope_change=0.,rss_single=rss,
                    bic_single=None,bic_segmented=None),pd.DataFrame(dict(T=T,log_T=x,log_abs_skew=y,fitted_log_skew=pred,fitted_abs_skew=np.exp(pred)))
    bic1=n*np.log(rss/n)+3*np.log(n)
    best=None
    def at_knot(knot):
        design=np.c_[np.ones(n),x,np.maximum(x-knot,0.)]
        beta=np.linalg.lstsq(design,y,rcond=None)[0]
        fitted=design@beta
        return float(np.sum((y-fitted)**2)),beta,fitted
    if n>=2*min_side:
        for split in range(min_side,n-min_side+1):
            lo=x[split-1]; hi=x[split]
            result=minimize_scalar(lambda knot:at_knot(knot)[0],bounds=(lo,hi),method='bounded',options={'xatol':1e-12})
            if not result.success:raise ArithmeticError('Knot optimization failed')
            left_beta=np.linalg.lstsq(np.c_[np.ones(split),x[:split]],y[:split],rcond=None)[0]
            right_beta=np.linalg.lstsq(np.c_[np.ones(n-split),x[split:]],y[split:],rcond=None)[0]
            candidates=[lo,hi,float(result.x)]
            if left_beta[1]!=right_beta[1]:
                intersection=(right_beta[0]-left_beta[0])/(left_beta[1]-right_beta[1])
                if lo<=intersection<=hi:candidates.append(float(intersection))
            for knot in candidates:
                if np.count_nonzero(x<=knot)<min_side or np.count_nonzero(x>knot)<min_side:continue
                ss,beta,fitted=at_knot(knot)
                if best is None or ss<best[0]:best=(ss,knot,beta,fitted)
    bic2=None if best is None or best[0]<=exact_tol else n*np.log(best[0]/n)+5*np.log(n)
    selected='segmented' if best is not None and (best[0]<=exact_tol or bic2<bic1) else 'single'
    if selected=='segmented':
        ss,knot,beta,pred=best; left=float(beta[1]); right=float(beta[1]+beta[2]); breakT=float(np.exp(knot))
    else:left=float(coef[1]);right=None;breakT=None
    report=dict(status='exploratory' if best is None or best[0]>exact_tol else 'exact_segmented_BIC_undefined',
        selected=selected,n=n,bic_single=float(bic1),bic_segmented=None if bic2 is None else float(bic2),
        delta_bic=None if bic2 is None else float(bic1-bic2),rss_single=rss,
        rss_segmented=None if best is None else best[0],break_T=breakT,slope_left=left,slope_right=right,
        parameter_count_single=3,parameter_count_segmented=5,
        intercept=float(beta[0]) if selected=='segmented' else float(coef[0]),
        slope_change=float(beta[2]) if selected=='segmented' else 0.,
        sign_changes=int(np.count_nonzero(np.diff(np.sign(skew)))),
        interpretation='Descriptive in-sample fit; no structural regime or asymptotic Hurst inference.')
    return report,pd.DataFrame(dict(T=T,log_T=x,log_abs_skew=y,fitted_log_skew=pred,fitted_abs_skew=np.exp(pred)))


def create_figures(surface,skews,checks,regressions,folder):
    figures={}
    surf=make_subplots(rows=1,cols=2,specs=[[{'type':'surface'},{'type':'surface'}]],subplot_titles=['Rough Heston','MPCE'])
    for col,model in enumerate(['rough_heston','mpce'],1):
        p=surface[(surface.model==model)&surface.requested_surface_point].pivot(index='T',columns='moneyness',values='display_black_iv').sort_index()
        surf.add_trace(go.Surface(x=p.columns.to_numpy(),y=p.index.to_numpy(),z=100*p.to_numpy(),name=model,
            colorbar=dict(title='Vol (%)',x=.45 if col==1 else 1.),showscale=True,
            hovertemplate='K/F=%{x:.4f}<br>T=%{y:.4f} an<br>Vol=%{z:.4f}%<extra>'+model+'</extra>'),row=1,col=col)
    surf.update_layout(title='Surfaces de volatilité : Rough Heston et MPCE',
        scene=dict(xaxis_title='K/F',yaxis_title='Maturite (annees)',zaxis_title='Volatilite (%)'),
        scene2=dict(xaxis_title='K/F',yaxis_title='Maturite (annees)',zaxis_title='Volatilite (%)'))
    common_camera=dict(eye=dict(x=1.4,y=1.4,z=1.1))
    finite=100*surface.loc[surface.requested_surface_point,'display_black_iv'].dropna().to_numpy()
    low=float(finite.min());high=float(finite.max())
    surf.update_layout(scene_camera=common_camera,scene2_camera=common_camera,updatemenus=[dict(type='buttons',buttons=[
        dict(label='Echelle commune',method='relayout',args=[{'scene.zaxis.range':[low,high],'scene2.zaxis.range':[low,high]}]),
        dict(label='Echelles individuelles',method='relayout',args=[{'scene.zaxis.autorange':True,'scene2.zaxis.autorange':True}])])])
    figures['surfaces_interactives']=surf
    linear=go.Figure();logfig=go.Figure()
    fig,ax=plt.subplots();figlog,axlog=plt.subplots()
    for model in ['rough_heston','mpce']:
        p=skews[skews.model==model].sort_values('T')
        linear.add_trace(go.Scatter(x=p['T'],y=p['skew'],mode='lines+markers',name=model,
            error_y=dict(type='data',array=p['confidence_z']*p['skew_se'].fillna(0).to_numpy(),visible=model=='mpce')))
        ax.errorbar(p['T'],p['skew'],yerr=p['confidence_z']*p['skew_se'].fillna(0),label=model,marker='o')
        for distance,q in checks[checks.model==model].groupby('distance'):
            linear.add_trace(go.Scatter(x=q['T'],y=q.centered_slope,mode='lines',name=f'{model} pente h={distance}',visible='legendonly'))
        eligible=p[p.loglog_eligible]
        logfig.add_trace(go.Scatter(x=eligible['T'],y=np.abs(eligible['skew']),mode='markers',name=model+' |skew|',
            customdata=np.c_[eligible['skew'],eligible['skew_se']],hovertemplate='T=%{x}<br>|skew|=%{y}<br>skew signe=%{customdata[0]}<br>SE=%{customdata[1]}<extra></extra>'))
        axlog.loglog(eligible['T'],np.abs(eligible['skew']),'o',label=model+' |skew|')
        fit=regressions[model]['curve'];r=regressions[model]['report']
        if len(fit):
            if model=='mpce' and not bool(p.significant.all()):
                logfig.add_annotation(text='MPCE : incertitude du skew sur certains points',x=.5,y=1.12,xref='paper',yref='paper',showarrow=False)
            label=f"{model} {r.get('selected')} : pente={r.get('slope_left',np.nan):.3f}"
            if r.get('slope_right') is not None:label+=f" / {r['slope_right']:.3f}"
            logfig.add_trace(go.Scatter(x=fit['T'],y=fit.fitted_abs_skew,mode='lines',name=label))
            axlog.loglog(fit['T'],fit.fitted_abs_skew,'-',label=label)
            if r.get('break_T') is not None:
                logfig.add_vline(x=r['break_T'],line_dash='dash')
                axlog.axvline(r['break_T'],linestyle='--')
    linear.update_layout(title='Structure à terme du skew : Rough Heston et MPCE',xaxis_title='Maturite (annees)',yaxis_title='d sigma / d log(K/F)')
    logfig.update_layout(title='Amplitude du skew en fonction de la maturité',xaxis_type='log',yaxis_type='log',xaxis_title='Maturite (annees)',yaxis_title='|d sigma / d log(K/F)|')
    figures['skew_atm']=linear;figures['skew_atm_loglog']=logfig
    ax.set_xlabel('Maturite (annees)');ax.set_ylabel('Skew ATM signe');ax.legend();fig.tight_layout();fig.savefig(folder/'skew_atm.png');plt.close(fig)
    axlog.set_xlabel('Maturite (annees)');axlog.set_ylabel('|Skew ATM|');axlog.legend(fontsize=8);figlog.tight_layout();figlog.savefig(folder/'skew_atm_loglog.png');plt.close(figlog)
    if 'raw_black_iv' in surface.columns:
        create_estimator_audit_figures(surface,skews,folder)
    for name,f in figures.items():
        f.write_html(folder/(name+'.html'),include_plotlyjs=True,full_html=True,auto_open=False)
        (folder/'json').mkdir(exist_ok=True)
        f.write_json(folder/'json'/(name+'.json'))
    return figures


def run_project(configuration,project_root):
    root=Path(project_root).resolve();c=prepare_configuration(configuration)
    name=c['output']['folder']
    if Path(name).name!=name or name in ('','.','..'):raise ValueError('Dossier enfant direct requis.')
    output=root/name
    if output.is_symlink():raise ValueError('Dossier symbolique interdit.')
    if output.exists():
        if not c['output']['overwrite']:raise FileExistsError(output)
        manifest_path=output/'json'/'manifest.json'
        if not manifest_path.exists():raise FileExistsError('Dossier non gere : choisir un nouveau dossier de resultats.')
        old=json.loads(manifest_path.read_text())
        actual={str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}
        unknown=actual-set(old)-{'json/manifest.json'}
        if unknown:raise FileExistsError(f'Fichiers non geres : {unknown}')
    lock=root/'.publication.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    stage=Path(tempfile.mkdtemp(prefix='.stage_',dir=root));(stage/'json').mkdir()
    try:
        samples,diagnostics,metadata,history=mpce.simulate(c['mpce_model'],c['simulation'],c['market'],c['grid'])
        conditional=metadata.pop('conditional_mc',None)
        metadata.pop('observation_weights',None)
        metadata.pop('observation_weights',None)
        surface,skews,checks=build_outputs(c,samples,conditional)
        greeks=None
        if c.get('rough_greeks',{}).get('enabled',False):
            controls=c['rough_greeks']
            greeks=rh.parameter_greeks_fourier(c['rough_model'],c['rough_numerical'],c['market'],c['grid'],controls,controls.get('payoff'))
            greeks.to_csv(stage/'greeks_rough_heston.csv',index=False)
        regressions={}
        for model in ['rough_heston','mpce']:
            p=skews[(skews.model==model)&skews.loglog_eligible]
            report,fit=fit_bic(p['T'],p['skew'],c['skew']['bic_min_points_per_segment'])
            curve=regression_curve(report,fit)
            regressions[model]=dict(report=report,data=fit,curve=curve)
            fit.to_csv(stage/('regression_'+model+'.csv'),index=False)
            curve.to_csv(stage/('courbe_regression_'+model+'.csv'),index=False)
        verification={}
        if c['verification']['enabled']:
            report,points=mpce.verify_reference(c['mpce_model'],c['simulation'],c['market'],c['grid'],c['verification'])
            points.to_csv(stage/'verification_reference_mpce.csv',index=False)
            verification['mpce']=report
            if report['status'] in ['failed','numerical_failure'] or not report['kernel_passed'] or not report['batch_passed']:
                write_json(root/'verification_echec.json',dict(report=report,points=points.to_dict('records')))
                raise ArithmeticError('Verification MPCE echouee : voir json/verification_echec.json')
        verification.update(status='PARTIALLY_VERIFIED',skew_consistency_failures=int((~checks.consistent).sum()),
            invalid_surface_points=int((~np.isfinite(surface.black_iv)).sum()),
            limitations=['Scenarios non calibres','Conditionnement : voir simulation_metadata.json ; erreurs IID non valides entre historiques',
                          'Convergence temporelle non etablie','Ensemble historique : erreurs IID de trajectoires descriptives seulement','BIC descriptif, erreurs correlees',
                          'Coefficient corrige : anciens chiffres de diffusion invalides'])
        for name,df in [('surfaces',surface),('skew_atm',skews),('verification_pentes_atm',checks),('diagnostics_mpce',diagnostics),('historique_mpce',history)]:
            df.to_csv(stage/(name+'.csv'),index=False)
        if c['output']['save_samples']:
            arrays=dict(prices=samples,maturities=c['grid']['maturities'])
            if conditional is not None:arrays.update({'conditional_'+key:value for key,value in conditional.items()})
            np.savez_compressed(stage/'echantillons_mpce.npz',**arrays)
        write_json(stage/'configuration_effective.json',c);write_json(stage/'verification.json',verification)
        write_json(stage/'regressions_bic.json',{m:v['report'] for m,v in regressions.items()})
        write_json(stage/'simulation_metadata.json',metadata)
        env={n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','numba','matplotlib','plotly']}
        write_json(stage/'environnement.json',dict(python=platform.python_version(),packages=env))
        figures=create_figures(surface,skews,checks,regressions,stage)
        if greeks is not None:figures['rough_greeks']=create_rough_greek_figures(greeks,stage)
        create_comparison_figures(surface,diagnostics,stage)
        create_diagnostic_decomposition(diagnostics,stage)
        for p in stage.rglob('*.json'):json.loads(p.read_text())
        for name,df in [('surfaces',surface),('skew_atm',skews),('verification_pentes_atm',checks)]:
            read=pd.read_csv(stage/(name+'.csv'));assert len(read)==len(df)
            np.testing.assert_allclose(read.select_dtypes('number'),df.select_dtypes('number'),rtol=1e-10,atol=1e-12,equal_nan=True)
        if greeks is not None:
            exported=pd.read_csv(stage/'greeks_rough_heston.csv')
            np.testing.assert_allclose(exported.select_dtypes('number'),greeks.select_dtypes('number'),rtol=1e-10,atol=1e-12)
        write_json(stage/'manifest.json',{str(p.relative_to(stage)):hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.rglob('*') if p.is_file()})
        if output.exists():
            backup=root/(c['output']['folder']+'.backup')
            if backup.exists():raise FileExistsError(backup)
            output.rename(backup)
            try:stage.rename(output)
            except Exception:backup.rename(output);raise
            shutil.rmtree(backup)
        else:stage.rename(output)
        return dict(surface=surface,skews=skews,checks=checks,regressions=regressions,rough_greeks=greeks,
                    verification=verification,figures=figures,configuration=c)
    finally:
        if stage.exists():shutil.rmtree(stage)
        lock.unlink()


def create_comparison_figures(surface,diagnostics,folder):
    """Keep baseline comparison charts in addition to the requested new charts."""
    selected=surface[surface.requested_surface_point].copy()
    left=selected[selected.model=='rough_heston'];right=selected[selected.model=='mpce']
    comparison=left.merge(right,on=['T','moneyness','K','F'],suffixes=('_rh','_mpce'),validate='one_to_one')
    comparison['black_difference_bp']=(comparison.black_iv_mpce-comparison.black_iv_rh)*1e4
    comparison.to_csv(folder/'comparaison_surfaces.csv',index=False)
    for convention,factor in [('black',100.)]:
        column=convention+'_iv'
        fig=plt.figure(figsize=(12,5))
        for j,model in enumerate(['rough_heston','mpce'],1):
            p=selected[selected.model==model].pivot(index='T',columns='moneyness',values=column).sort_index()
            X,Y=np.meshgrid(p.columns,p.index);ax=fig.add_subplot(1,2,j,projection='3d')
            ax.plot_surface(X,Y,p.to_numpy()*factor)
            ax.set_xlabel('K/F');ax.set_ylabel('Maturite (annees)');ax.set_zlabel('Vol (%)');ax.set_title(model)
        fig.tight_layout();fig.savefig(folder/('surfaces_'+convention+'.png'));plt.close(fig)
        diff='black_difference_bp'
        p=comparison.pivot(index='T',columns='moneyness',values=diff).sort_index()
        fig,ax=plt.subplots();im=ax.pcolormesh(p.columns,p.index,p.to_numpy(),shading='nearest')
        fig.colorbar(im,ax=ax,label='MPCE - rough Heston (bp de vol)')
        ax.set_xlabel('K/F');ax.set_ylabel('Maturite (annees)');fig.tight_layout();fig.savefig(folder/('ecarts_'+convention+'.png'));plt.close(fig)
        create_black_cuts(selected,folder)
        fig,ax=plt.subplots()
        for model in ['rough_heston','mpce']:
            p=selected[(selected.model==model)&(selected.moneyness==1.)].sort_values('T')
            ax.plot(p['T'],p[column]*factor,label=model)
        ax.set_xlabel('Maturite (annees)');ax.set_ylabel('Vol ATM (%)');ax.legend();fig.tight_layout();fig.savefig(folder/('atm_'+convention+'.png'));plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(diagnostics['T'],diagnostics.mean_X,label='Moyenne MPCE')
    axes[0].plot(diagnostics['T'],diagnostics.theoretical_F,label='Forward theorique');axes[0].legend()
    axes[1].plot(diagnostics['T'],diagnostics.mean_variance);axes[1].set_ylabel('Moyenne de la variance')
    for ax in axes:ax.set_xlabel('Maturite (annees)')
    fig.tight_layout();fig.savefig(folder/'diagnostics_mpce.png');plt.close(fig)


def regression_curve(report,points):
    """Render the exact fitted kink, not a chord between observed maturities."""
    if not len(points):return pd.DataFrame(columns=['T','fitted_log_skew','fitted_abs_skew'])
    x=np.linspace(float(points.log_T.min()),float(points.log_T.max()),200)
    if report.get('break_T') is not None:x=np.unique(np.r_[x,np.log(report['break_T'])])
    y=report['intercept']+report['slope_left']*x
    if report.get('break_T') is not None:y+=report['slope_change']*np.maximum(x-np.log(report['break_T']),0.)
    return pd.DataFrame(dict(T=np.exp(x),fitted_log_skew=y,fitted_abs_skew=np.exp(y)))


def create_estimator_audit_figures(surface,skews,folder):
    selected=surface[(surface.model=='mpce')&surface.requested_surface_point]
    times=sorted(selected['T'].unique())[:3]
    fig,axes=plt.subplots(1,len(times),figsize=(5*len(times),4),squeeze=False)
    for ax,T in zip(axes[0],times):
        p=selected[selected['T']==T].sort_values('moneyness')
        ax.plot(p.moneyness,100*p.raw_black_iv,'o-',label='Payoffs bruts')
        ax.plot(p.moneyness,100*p.black_iv,'o-',label='Conditionnel + controle')
        ax.set_title(f'T={T:g} an');ax.set_xlabel('K/F');ax.set_ylabel('Volatilité (%)');ax.legend()
    fig.tight_layout();fig.savefig(folder/'audit_smiles_brut_conditionnel.png');plt.close(fig)
    p=skews[skews.model=='mpce'].sort_values('T')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].errorbar(p['T'],p.raw_skew,yerr=p.confidence_z*p.raw_skew_se,label='Payoffs bruts',marker='o')
    axes[0].errorbar(p['T'],p['skew'],yerr=p.confidence_z*p.skew_se,label='Conditionnel + controle',marker='o')
    axes[1].errorbar(p['T'],p['skew'],yerr=p.confidence_z*p.skew_se,label='Conditionnel + controle',marker='o')
    for ax in axes:ax.set_xlabel('Maturite (annees)');ax.set_ylabel('Skew ATM MPCE');ax.legend()
    axes[1].set_title('Structure à terme du skew : MPCE')
    fig.tight_layout();fig.savefig(folder/'audit_skew_mpce.png');plt.close(fig)


def create_diagnostic_decomposition(diagnostics,folder):
    if 'history_log_contribution' not in diagnostics:return
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for key,label in [('history_log_contribution','Historique'),('mean_drift_log_contribution','Derive future'),('mean_stochastic_log_contribution','Bruit futur moyen')]:
        axes[0].plot(diagnostics['T'],diagnostics[key],label=label)
    axes[0].set_ylabel('Contribution a la log-variance');axes[0].legend()
    axes[1].plot(diagnostics['T'],100*diagnostics.rms_volatility,label='Racine de E[variance]')
    axes[1].set_ylabel('Volatilite RMS (%)');axes[1].legend()
    for ax in axes:ax.set_xlabel('Maturite (annees)')
    fig.tight_layout();fig.savefig(folder/'decomposition_log_variance.png');plt.close(fig)


# Public configuration and notebook entry points.
def load_config(path=None):
    path=Path(path) if path else Path(__file__).resolve().parent/'config.json'
    return json.loads(path.read_text(encoding='utf-8'))

def to_engine_config(config):
    """Adaptation explicite des noms et de la grille, sans changement de noyau."""
    c=copy.deepcopy(config)
    grid=c['grid'];ts=np.asarray(grid['maturities'],float);ks=np.asarray(grid['strikes'],float)
    if grid['strike_coordinate'] not in ('K/F','K'):
        raise ValueError('strike_coordinate doit être K/F ou K.')
    if grid['strike_coordinate']=='K':
        raise ValueError('Le rendu historique requiert K/F. Utiliser K/F ; aucun strike absolu ne sera converti silencieusement.')
    rh.validate_market(c['market'],dict(maturities=ts,moneyness=ks))
    sim=c['simulation'];horizon=float(sim['horizon']);dt=horizon/sim['n_steps']
    indices=np.rint(ts/dt).astype(int)
    if np.any(indices<1) or np.any(indices>sim['n_steps']):raise ValueError('Maturités hors horizon ou inférieures à un pas.')
    effective=np.unique(indices*dt)
    if len(effective)!=len(ts):raise ValueError('Grille trop fine : plusieurs maturités sur le même pas.')
    if c.get('grid_alignment','nearest_step')=='strict' and not np.allclose(ts,effective,rtol=0,atol=1e-12):
        raise ValueError('Maturités non alignées : modifier n_steps ou activer nearest_step.')
    length=sim['history_length']
    if not np.isfinite(length) or length<=0:raise ValueError('history_length doit être positif, en années.')
    if not np.isclose(length/dt,round(length/dt),atol=1e-9,rtol=0):raise ValueError('La longueur historique doit être un multiple du pas dt.')
    c['grid']=dict(maturities=effective.tolist(),moneyness=ks.tolist())
    c['requested_grid']=grid
    c['public_config']=copy.deepcopy(config)
    c['simulation']['history_multiple']=1
    c['simulation']['min_steps_shortest_maturity']=1
    if not np.isfinite(c['mpce_model']['V0']) or c['mpce_model']['V0']<=0: raise ValueError('V0 positif fini requis.')
    c['mpce_model']['initial_log_variance']=float(np.log(c['mpce_model'].pop('V0')))
    rho=c['mpce_model'].pop('correlations')
    corr=np.array([[1,rho['rho_H_y'],rho['rho_H_S']],[rho['rho_H_y'],1,rho['rho_y_S']],[rho['rho_H_S'],rho['rho_y_S'],1]],float)
    if not np.isfinite(corr).all() or np.any(abs(corr)>1):raise ValueError('Corrélations invalides.')
    np.linalg.cholesky(corr)
    c['mpce_model']['correlation']=corr.tolist()
    return c

def correlation_matrix(config):
    rho=config['mpce_model']['correlations']
    return pd.DataFrame([[1,rho['rho_H_y'],rho['rho_H_S']],
                         [rho['rho_H_y'],1,rho['rho_y_S']],
                         [rho['rho_H_S'],rho['rho_y_S'],1]],index=['H','y','S'],columns=['H','y','S'])

def run(config=None, project_root=None):
    config=load_config() if config is None else config
    root=Path(project_root) if project_root else Path(__file__).resolve().parent
    result=run_project(to_engine_config(config),root)
    # The public config is embedded in the effective config, before publication.
    return result



def create_black_cuts(surface,folder):
    """Axes indépendants : gauche rough Heston, droite MPCE ; aucune normalisation."""
    selected=surface[surface.requested_surface_point] if 'requested_surface_point' in surface else surface
    times=sorted(selected['T'].unique());columns=min(4,len(times));rows=math.ceil(len(times)/columns)
    fig,axes=plt.subplots(rows,columns,figsize=(5*columns,3.7*rows),squeeze=False)
    pairs=[]
    for ax,T in zip(axes.ravel(),times):
        right=ax.twinx();pairs.append((ax,right))
        for model,target,color in [('rough_heston',ax,'tab:blue'),('mpce',right,'tab:orange')]:
            points=selected[(selected.model==model)&(selected['T']==T)].sort_values('moneyness')
            target.plot(points.moneyness,100*points.black_iv,color=color,label=model)
        ax.set_ylabel('Volatilité Rough Heston (%)',color='tab:blue');ax.tick_params(axis='y',labelcolor='tab:blue')
        right.set_ylabel('Volatilité MPCE (%)',color='tab:orange');right.tick_params(axis='y',labelcolor='tab:orange')
        ax.set_title(f'T={T:g} an');ax.set_xlabel('K/F')
        ax.legend(ax.lines+right.lines,[line.get_label() for line in ax.lines+right.lines],fontsize=7)
    for ax in axes.ravel()[len(times):]:ax.set_visible(False)
    fig.suptitle('Coupes de volatilité : Rough Heston et MPCE')
    fig.tight_layout();fig.savefig(Path(folder)/'coupes_black.png');plt.close(fig)
    return pairs


def create_rough_greek_figures(greeks,folder):
    figures={}
    for parameter,points in greeks.groupby('parameter',sort=False):
        figure=go.Figure()
        for T,cut in points.groupby('T',sort=True):
            cut=cut.sort_values('moneyness')
            figure.add_trace(go.Scatter(x=cut.moneyness,y=cut.greek,mode='lines',name=f'T={T:g}',
                customdata=np.c_[cut.K,cut.fd_error_estimate,cut.fd_stable],
                hovertemplate='K/F=%{x}<br>Grecque=%{y}<br>K=%{customdata[0]}<br>Erreur FD estimée=%{customdata[1]}<br>Stable FD=%{customdata[2]}<extra></extra>'))
        figure.update_layout(title=f'Sensibilité du prix à {parameter} : Rough Heston',xaxis_title='K/F',yaxis_title=f'dP/d{parameter} (par unité absolue)')
        figure.write_html(str(Path(folder)/f'greek_rough_{parameter}.html'),include_plotlyjs=True)
        figures[parameter]=figure
    return figures
