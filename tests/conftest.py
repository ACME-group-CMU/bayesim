import numpy as np
import pandas as pd
import pytest
import itertools

# @pytest.fixture
# def dummy_model_h5(tmp_path):
#     df = pd.DataFrame({
#         'Nt_i': np.logspace(10, 12, num=3),
#         'Nt_Sb2Se3': np.logspace(14, 16, num=3),
#         'mu_n_Sb2Se3': np.linspace(0.1, 1.0, num=3),
#         'EA_CdS': np.linspace(3.5, 4.0, num=3),
#         'Voltage': [0, 0.5, 1.0],
#         'Temperature': [300, 300, 300],
#         'Illumination': [1, 1, 1],
#         'Current': [0.1, 0.2, 0.15],
#     })
#     path = tmp_path / "model.h5"
#     df.to_hdf(path, key='dummykey', mode='w')
#     return str(path)

# @pytest.fixture
# def dummy_obs_h5(tmp_path):
#     df = pd.DataFrame({
#         'Voltage': [0, 0.5, 1.0],
#         'Temperature': [300, 300, 300],
#         'Illumination': [1, 1, 1],
#         'Current': [0.11, 0.19, 0.16],
#     })
#     path = tmp_path / "obs.h5"
#     df.to_hdf(path, key='dummykey', mode='w')
#     return str(path)

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