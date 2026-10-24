import numpy as np
import polars as pl
from .data_loading import ROOT,save_json,read_template

def check_submission(template,output,candidates):
    if output.columns!=template.columns: raise ValueError('Column mismatch')
    if output['user_id'].to_list()!=template['user_id'].to_list(): raise ValueError('User order/coverage mismatch')
    if output['user_id'].n_unique()!=output.height: raise ValueError('Duplicate users')
    allowed=set(map(int,candidates)); invalid=0
    for value in output['tv_show_id']:
        tokens=value.split(' ')
        if len(tokens)!=5 or any(not t.isdecimal() for t in tokens): raise ValueError('Expected five integer tokens')
        ids=list(map(int,tokens))
        if len(set(ids))!=5 or 0 in ids: raise ValueError('Duplicate/zero prediction')
        invalid+=sum(x not in allowed for x in ids)
    if invalid: raise ValueError('Out-of-schedule prediction')
    return {'rows':output.height,'expected_rows':template.height,'unique_users':output['user_id'].n_unique(),'exact_user_order':True,'columns':output.columns,'predictions_per_user':5,'zero_ids':0,'within_row_duplicate_ids':0,'out_of_future_schedule_ids':invalid,'encoding':'UTF-8','delimiter':',','status':'PASS'}

def generate(folder,users,engine,spec):
    template=read_template(folder)
    # engine rows must follow the official template, not a numeric sort.
    indices=template.join(users,on='user_id',how='left',maintain_order='left')['user_idx'].to_numpy()
    if not np.array_equal(indices,engine.users): raise AssertionError('Prediction user order')
    pred=engine.recommend(spec)
    output=template.with_columns(pl.Series('tv_show_id',[' '.join(map(str,row)) for row in pred]))
    summary=check_submission(template,output,engine.items[engine.candidates])
    output.write_csv(ROOT/'outputs/submission.csv')
    reread=pl.read_csv(ROOT/'outputs/submission.csv',schema_overrides={'user_id':pl.String,'tv_show_id':pl.String})
    check_submission(template,reread,engine.items[engine.candidates])
    save_json(ROOT/'outputs/submission_verification.json',summary)
    return pred
