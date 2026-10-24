import polars as pl
from src.validation import period,labels

def test_cutoff_excludes_crossing_and_late_records():
    f=pl.DataFrame({'start_time':[0,80,100,100], 'end_time':[80,120,110,150]})
    assert period(f,0,100).height==1
    assert period(f,100,140).height==1

def test_no_invented_five_item_ground_truth():
    f=pl.DataFrame({'user_idx':[0]*6+[1]*2,'tv_show_id':[1,2,3,4,5,1,8,9]})
    eligible,active=labels(f,[0,1,2])
    assert eligible['user_idx'].to_list()==[0]
    assert eligible['top5'].to_list()==[[1,2,3,4,5]]
    assert active.height==2
