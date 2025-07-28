import numpy as np
import pandas as pd
import pytest
import itertools

@pytest.fixture
def dummy_obs_h5(tmp_path):
    df = pd.DataFrame({
        'Voltage':      [0, 0, 0],
        'Temperature':  [300, 300, 300],
        'Illumination': [1, 1, 1],
        'Current':      [0.14, 0.15, 0.16]
    })
    path = tmp_path / "obs.h5"
    df.to_hdf(path, key='dummykey', mode='w')
    return str(path)

@pytest.fixture
def dummy_model_h5(tmp_path):
    Nt_i_vals       = [1e10, 1e11]
    Nt_Sb2Se3_vals  = [1e14, 1e15]
    mu_vals         = [0.1, 0.2]
    EA_vals         = [3.5, 3.6]

    rows = []
    for Nt_i, Nt_Sb, mu, EA in itertools.product(
            Nt_i_vals, Nt_Sb2Se3_vals, mu_vals, EA_vals):
        rows.append({
            'Nt_i':          Nt_i,
            'Nt_Sb2Se3':     Nt_Sb,
            'mu_n_Sb2Se3':   mu,
            'EA_CdS':        EA,
            'Voltage':       0,
            'Temperature':   300,
            'Illumination':  1,
            'Current':       0.15,
        })
    df = pd.DataFrame(rows)
    df['uncertainty'] = 0.1
    path = tmp_path / "model.h5"
    df.to_hdf(path, key='dummykey', mode='w')
    return str(path)

@pytest.fixture
def kinematics_obs_h5(tmp_path):
    df = pd.DataFrame({
        'time':[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
        'y': [4.106762790545101, 4.665020657297301, 5.666324131998576, 6.135059566734121,
              5.560421103204023, 5.284494987088342, 4.172131801460504]
    })
    path = tmp_path / "obs.h5"
    df.to_hdf(path, key='dummykey', mode='w')
    return str(path)