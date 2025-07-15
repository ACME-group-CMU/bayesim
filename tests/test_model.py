import pytest
import pandas as pd
import os
import numpy as np
from bayesim.params import Param_list
from bayesim.model import Model

# def make_simple_pl():
#     pl = Param_list()
#     pl.add_fit_param(name='Nt_i', vals=[1e10,1e11], spacing='log')
#     pl.add_fit_param(name='Nt_Sb2Se3', vals=[1e14,1e15], spacing='log')
#     pl.add_fit_param(name='mu_n_Sb2Se3', vals=[0.1,0.2], spacing='linear')
#     pl.add_fit_param(name='EA_CdS', vals=[3.5,3.6], spacing='linear')
#     for name in ('Voltage', 'Temperature', 'Illumination'):
#         pl.add_ec(name=name)
#     pl.add_output(name='Current')
#     return pl

# def test_attach_and_run(tmp_path, dummy_model_h5, dummy_obs_h5):
#     pl = make_simple_pl()
#     m = Model(params=pl)
#     m.attach_observations(
#         obs_data_path=dummy_obs_h5,
#         ec_x_var='Voltage',
#         fixed_unc=0.5,
#         keep_all=True
#     )
#     m.attach_model(
#         mode='file',
#         model_data_path=dummy_model_h5
#     )
#     m.run(th_pm=0.5)
#     top = m.top_probs(1)
#     assert 'Nt_i' in top.columns
#     assert 'prob' in top.columns
#     out_png = tmp_path / "out.png"
#     m.visualize_probs(fpath=str(out_png))
#     assert out_png.exists() and out_png.stat().st_size > 0

def make_kinematics_pl():
    pl = Param_list()
    pl.add_fit_param(name='v0', val_range=(0,20), length=10)
    pl.add_fit_param(name='g', val_range=(5,15), length=10)
    pl.add_fit_param(name='y0', vals=(1,10), length=10)
    pl.add_ec(name='time')
    pl.add_output(name='y')
    return pl

def freefall_model(ecs: dict, params: dict) -> float:
    t  = ecs['time']
    v0 = params['v0']
    g  = params['g']
    y0 = params['y0']
    return y0 + v0*t - 0.5*g*t**2

def test_attach_and_run(tmp_path, kinematics_obs_h5):
    pl = make_kinematics_pl()
    m = Model(params=pl, output_var='y')

    m.attach_observations(
        obs_data_path=kinematics_obs_h5,
        ec_x_var='time',
        fixed_unc=0.3,
        keep_all=True
    )

    m.attach_model(
    mode='function',
    model_data_func=freefall_model,
    calc_model_unc=True
    )

    m.run()
    top = m.top_probs(1)
    assert 'v0' in top.columns
    assert 'prob' in top.columns
    out_png = tmp_path / "out.png"
    m.visualize_probs(fpath=str(out_png))
    assert out_png.exists() and out_png.stat().st_size > 0