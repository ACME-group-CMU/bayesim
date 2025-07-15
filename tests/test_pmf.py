import numpy as np
import pandas as pd
import pytest
from bayesim.params import Param_list
from bayesim.pmf import Pmf

@pytest.fixture
def simple_pl():
    pl = Param_list()
    pl.add_fit_param(name='x', vals=[1, 2], spacing='linear')
    pl.add_fit_param(name='y', vals=[10, 20], spacing='linear')
    return pl

def test_make_points_list(simple_pl):
    pmf = Pmf(params=simple_pl.fit_params)
    df = pmf.make_points_list(simple_pl.fit_params, total_prob=2.0)

    # 2×2 grid -> 4 rows
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 4

    # check for required columns
    for col in ['x','x_min','x_max','y','y_min','y_max','prob']:
        assert col in df.columns

    # total probability matches
    assert pytest.approx(df['prob'].sum(), rel=1e-9) == 2.0

def test_init_and_as_dict(simple_pl):
    # empty‐init branch
    pmf0 = Pmf()
    assert pmf0.is_empty
    d0 = pmf0.as_dict()
    pmf0_roundtrip = Pmf(prob_dict=d0)
    assert pmf0_roundtrip.is_empty

    # params‐init branch
    pmf1 = Pmf(params=simple_pl.fit_params)
    assert not pmf1.is_empty
    d1 = pmf1.as_dict()
    pmf1_roundtrip = Pmf(prob_dict=d1)
    assert pmf1_roundtrip.param_names() == pmf1.param_names()
    assert pmf1_roundtrip.points.shape == pmf1.points.shape

def test_normalize_and_uniformize(simple_pl):
    pmf = Pmf(params=simple_pl.fit_params)
    # corrupt probabilities
    pmf.points['prob'] = [1, 2, 3, 4]
    pmf.normalize()
    # should sum to 1
    assert pytest.approx(pmf.points['prob'].sum(), rel=1e-9) == 1.0

    # uniformize -> all equal
    pmf.uniformize()
    assert all(np.isclose(pmf.points['prob'], 0.25))

def test_all_current_values(simple_pl):
    pmf = Pmf(params=simple_pl.fit_params)
    assert pmf.all_current_values('x') == [1, 2]
    assert pmf.all_current_values('y') == [10, 20]

def test_multiply_and_assertions(simple_pl):
    pmf1 = Pmf(params=simple_pl.fit_params)
    pmf2 = Pmf(params=simple_pl.fit_params)
    pmf1.uniformize()
    pmf2.uniformize()
    # give pmf2 a non‐uniform pattern
    pmf2.points['prob'] = np.array([1, 2, 3, 4], float)

    # multiplying and normalizing
    pmf1.multiply(pmf2)
    assert pytest.approx(pmf1.points['prob'].sum(), rel=1e-9) == 1.0

    # mismatched‐size multiplication raises
    with pytest.raises(AssertionError):
        pmf1.multiply(Pmf())
    
def test_param_names_and_most_probable(simple_pl):
    pmf = Pmf(params=simple_pl.fit_params)
    pmf.uniformize()
    assert pmf.param_names() == ['x', 'y']
    top2 = pmf.most_probable(2)
    # must include the two params + prob column
    for col in ['x','y','prob']:
        assert col in top2.columns
    assert len(top2) == 2

def test_project_1D(simple_pl):
    pmf = Pmf(params=simple_pl.fit_params)
    pmf.uniformize()
    # project along the first parameter
    first = pmf.params[0]
    bins, probs = pmf.project_1D(first)

    # bins should be one longer than probs
    assert len(bins) == len(probs) + 1
    # sum of probs = 1
    assert pytest.approx(sum(probs), rel=1e-9) == 1.0