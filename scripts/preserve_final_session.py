"""Copy the native JSONL unchanged; finish when the live turn completes.

This optional local helper watches only the explicitly selected source file.
It never uploads the session and never edits its contents.
"""
from pathlib import Path
import argparse,json,time,hashlib
from datetime import datetime,timezone

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--destination',required=True);args=p.parse_args()
    source=Path(args.source);dest=Path(args.destination);dest.mkdir(parents=True,exist_ok=True)
    last_signature=None;last_change=time.monotonic();started=last_change
    while time.monotonic()-started<21600:
        stat=source.stat();signature=(stat.st_size,stat.st_mtime_ns)
        if signature!=last_signature:
            data=source.read_bytes();after=source.stat()
            if (after.st_size,after.st_mtime_ns)!=signature:
                time.sleep(2);continue
            (dest/source.name).write_bytes(data)
            manifest={'source':str(source),'file':source.name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'copied_utc':datetime.now(timezone.utc).isoformat(),'transformation':'none; exact original native Codex JSONL','finalized_after_chat':False}
            (dest/'integrity_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            last_signature=signature;last_change=time.monotonic()
        if time.monotonic()-last_change>=90:
            # A stable snapshot is preserved even if this host's log format does
            # not expose a task_complete event. The manifest says what was seen.
            rows=[]
            for line in data.splitlines()[-100:]:
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass
            events=[r.get('payload',{}).get('type') for r in rows if r.get('type')=='event_msg']
            last_completed=max((i for i,e in enumerate(events) if e=='task_complete'),default=-1)
            last_started=max((i for i,e in enumerate(events) if e=='task_started'),default=-1)
            complete=last_completed>last_started and last_completed>=0
            if complete:
                manifest['finalized_after_chat']=True
                manifest['completion_evidence']='Native event_msg task_complete and at least 90 seconds of source stability'
                (dest/'integrity_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
                return
        time.sleep(5)

if __name__=='__main__':main()
