import pytest
from bayesim.params import Param_list

def test_add_fit_param_list():
    pl = Param_list()
    # initially no fit params
    assert pl.fit_params == []

    # add a fit param
    pl.add_fit_param(name='Nt_i', vals=[1,10,100], spacing='log')
    assert len(pl.fit_params) == 1
    p = pl.fit_params[0]
    assert p.name == 'Nt_i'
    assert p.spacing == 'log'
    assert p.vals == [1,10,100]

def test_add_ec_and_output():
    pl = Param_list()
    assert pl.ecs == []
    assert pl.output == []
    pl.add_ec(name='Voltage')
    pl.add_ec(name='Temperature')
    pl.add_output(name='Current')
    assert [ec.name for ec in pl.ecs] == ['Voltage', 'Temperature']
    assert [out.name for out in pl.output] == ['Current']