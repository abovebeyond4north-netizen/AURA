import pytest
from experiments.paired_budget_audit import audit

def row(name, success, n):
    return dict(seed=1,size=9,column=4,before=[0,1,2,3],after=[0,1,2,3],noise=0,when=0,reset_start=0,controller=name,success=success,actions=[0]*n)

def test_early_failure_not_rewarded():
    result=audit([row('Budget6',False,4),row('EntropyProbeCamera',False,12)])['overall']
    assert result['Budget6']['failure_adjusted']==result['EntropyProbeCamera']['failure_adjusted']==65

def test_success_and_unpaired():
    a=row('Budget6',True,4); b=row('EntropyProbeCamera',False,12)
    result=audit([a,b])['overall']; assert result['discordant']==1 and result['Budget6']['failure_adjusted']==4
    for rows in ([a],[a,a,b],[a,row('EntropyProbeCamera',False,66)]):
        with pytest.raises(ValueError): audit(rows)
