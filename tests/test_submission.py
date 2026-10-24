import polars as pl
import pytest
from src.submission import check_submission

def test_uint64_ids_exact_and_invalid_lists():
    t=pl.DataFrame({'user_id':['18443832178892490752'],'tv_show_id':['0 0 0 0 0']})
    out=t.with_columns(pl.lit('1 2 3 4 5').alias('tv_show_id'))
    assert check_submission(t,out,[1,2,3,4,5])['status']=='PASS'
    for value in ['1 1 2 3 4','0 1 2 3 4','1 2 3 4','1 2 3 4 6','1  2 3 4 5']:
        with pytest.raises(ValueError): check_submission(t,t.with_columns(pl.lit(value).alias('tv_show_id')),[1,2,3,4,5])
