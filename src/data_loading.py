"""Discover exact-schema CSV inputs without rounding uint64 user identifiers."""
from pathlib import Path
import hashlib
import json
import zipfile
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

def locate_data(directory=None):
    candidates = [Path(directory)] if directory else [ROOT/'data', ROOT.parent/'data/sweettv-tv-program-recommender']
    for folder in candidates:
        logs = list(folder.glob('dataset*.csv'))
        schedules = sorted(folder.glob('export_arh*.csv'))
        if logs and len(schedules) == 3 and (folder/'submission.csv').exists():
            return folder, logs[0], schedules
    archives = list(ROOT.parent.glob('**/sweettv-tv-program-recommender.zip'))
    if archives:
        target = ROOT/'data/raw'
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archives[0]) as archive:
            if not all((target/n).resolve().is_relative_to(target.resolve()) for n in archive.namelist()):
                raise ValueError('Unsafe archive path')
            archive.extractall(target)
        return locate_data(target)
    raise FileNotFoundError('Use --data-dir with the five extracted Kaggle CSV files.')

def read_template(folder):
    return pl.read_csv(Path(folder)/'submission.csv', schema_overrides={'user_id':pl.String,'tv_show_id':pl.String})

def provenance(paths):
    result=[]
    for p in paths:
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(2**20),b''): h.update(block)
        result.append({'name':p.name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
    return result

def save_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding='utf-8')

def inspect_inputs(folder):
    """Full raw schema, missingness, duplicate and date audit."""
    result={}
    for path in sorted(Path(folder).glob('*.csv')):
        frame=pl.read_csv(path,schema_overrides={'user_id':pl.String},infer_schema_length=10000)
        info={'rows':frame.height,'columns':frame.columns,'schema':{k:str(v) for k,v in frame.schema.items()},'missing':frame.null_count().row(0,named=True),'duplicates':frame.height-frame.unique().height,'replacement_characters':path.read_text(encoding='utf-8-sig').count('\ufffd')}
        for c in ['user_id','vsetv_id','channel_id','tv_show_id']:
            if c in frame.columns: info[c+'_unique']=frame[c].n_unique()
        if 'start_time' in frame.columns:
            fmt='%Y-%m-%d %H:%M:%S' if 'user_id' in frame.columns else '%d.%m.%Y %H:%M:%S'
            t=frame['start_time'].str.to_datetime(fmt,strict=False)
            info.update(start_min=str(t.min()),start_max=str(t.max()),invalid_start=t.null_count())
        if 'tv_show_title' in frame.columns: info['title_samples']=frame['tv_show_title'].head(5).to_list()
        result[path.name]=info
    save_json(ROOT/'outputs/data_inspection.json',result)
    return result
