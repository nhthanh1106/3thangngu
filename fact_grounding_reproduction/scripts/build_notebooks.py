import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def md(s):
    return {"cell_type": "markdown", "id": hashlib.sha1(s.encode()).hexdigest()[:12], "metadata": {}, "source": s.splitlines(True)}

def code(s):
    return {"cell_type": "code", "id": hashlib.sha1(s.encode()).hexdigest()[:12], "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)}

def save(name, cells):
    doc = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (ROOT / name).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

worker_source = (ROOT / "scripts/kaggle_train_worker.py").read_text(encoding="utf-8")
profile_code = """import sys, subprocess, json, math, re, os, queue, signal, threading, time
from pathlib import Path
WORKER_SOURCE = WORKER_SOURCE_PLACEHOLDER
worker_path=Path('/kaggle/working/kaggle_train_worker.py')
worker_path.write_text(WORKER_SOURCE,encoding='utf-8')

def run_streaming(cmd,label,log_path,heartbeat_seconds=30,timeout_seconds=None):
    env=os.environ.copy(); env['PYTHONUNBUFFERED']='1'; env['TOKENIZERS_PARALLELISM']='false'
    started=time.perf_counter(); lines=[]
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] [{label}] PROCESS_START cmd={cmd}',flush=True)
    process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,encoding='utf-8',errors='replace',env=env,start_new_session=True)
    messages=queue.Queue(); finished=object()
    def pump_output():
        try:
            for line in process.stdout: messages.put(line)
        finally:
            messages.put(finished)
    reader=threading.Thread(target=pump_output,daemon=True); reader.start()
    log_path=Path(log_path); done=False
    with log_path.open('w',encoding='utf-8') as sink:
        while not done:
            elapsed=time.perf_counter()-started
            if timeout_seconds is not None and elapsed>timeout_seconds:
                print(f'[{label}] TIMEOUT after {elapsed/60:.1f} minutes; terminating process group',flush=True)
                try: os.killpg(process.pid,signal.SIGTERM)
                except OSError: process.terminate()
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try: os.killpg(process.pid,signal.SIGKILL)
                    except OSError: process.kill()
                    process.wait()
                raise TimeoutError(f'{label} exceeded {timeout_seconds} seconds; inspect {log_path}')
            try:
                item=messages.get(timeout=heartbeat_seconds)
            except queue.Empty:
                print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] [{label}] HEARTBEAT elapsed={elapsed/60:.1f}m child_alive={process.poll() is None}',flush=True)
                continue
            if item is finished:
                done=True
            else:
                lines.append(item); sink.write(item); sink.flush(); print(item,end='',flush=True)
    returncode=process.wait(); elapsed=time.perf_counter()-started
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] [{label}] PROCESS_END returncode={returncode} elapsed={elapsed/60:.1f}m log={log_path}',flush=True)
    return returncode,''.join(lines)

train_file=Path(find_file('deberta_fg_train.csv')); dev_file=Path(find_file('deberta_fg_dev.csv'))
assert train_file.parent==dev_file.parent
DATA_ROOT=train_file.parent
TARGET_BATCH=16; ngpu=torch.cuda.device_count()
if ngpu<1: raise RuntimeError('No Kaggle GPU detected for DDP training.')
profiles=[]; prof=None
profile_batches=list(dict.fromkeys([min(8,max(1,TARGET_BATCH//ngpu)),min(4,max(1,TARGET_BATCH//ngpu)),min(2,max(1,TARGET_BATCH//ngpu)),1]))
for batch in profile_batches:
    accumulation=max(1,math.ceil(TARGET_BATCH/(batch*ngpu)))
    profile_path=Path(f'/kaggle/working/profile_batch_{batch}.json')
    cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone',f'--nproc_per_node={ngpu}',str(worker_path),'--mode','profile','--data-root',str(DATA_ROOT),'--batch',str(batch),'--accum',str(accumulation),'--profile-out',str(profile_path)]
    print(f'PROFILE_CHECKPOINT_BEGIN batch={batch} accumulation={accumulation} effective_batch={batch*ngpu*accumulation}',flush=True)
    returncode,combined=run_streaming(cmd,f'profile-batch-{batch}',f'/kaggle/working/profile_batch_{batch}.log',heartbeat_seconds=30,timeout_seconds=1800)
    print(f'PROFILE_CHECKPOINT_END batch={batch} accumulation={accumulation} returncode={returncode}',flush=True)
    if returncode==0 and profile_path.exists():
        prof=json.loads(profile_path.read_text(encoding='utf-8')); print(json.dumps(prof,indent=2)); micro,accum=batch,accumulation; break
    if re.search(r'out of memory|cuda out of memory|outofmemoryerror',combined,re.I):
        print(f'OOM at per-device batch {batch}; retrying smaller batch.')
        continue
    raise RuntimeError('DDP profile failed for a non-OOM reason. See the captured profile log: '+combined[-5000:])
if prof is None: raise RuntimeError(f'DDP GPU profiling failed at all per-device batches {profile_batches}; no full training was started.')
steps_epoch=math.ceil(len(train)/TARGET_BATCH); estimate=prof['seconds_per_step']*steps_epoch*3*1.25/3600
prof.update({'steps_per_epoch':steps_epoch,'estimated_steps_through_epoch_3':steps_epoch*3,'estimated_train_hours_with_25pct_overhead':estimate})
Path('/kaggle/working/profile_summary.json').write_text(json.dumps(prof,indent=2),encoding='utf-8')
print(json.dumps(prof,indent=2))
if estimate>7: raise RuntimeError('DDP profile projects >7 hours through epoch 3 including 25% overhead; stop before full training under the observed Kaggle 8-hour limit.')
""".replace("WORKER_SOURCE_PLACEHOLDER", repr(worker_source))

full_code = """import sys, subprocess, json
from pathlib import Path
profile=json.loads(Path('/kaggle/working/profile_summary.json').read_text(encoding='utf-8'))
worker_path=Path('/kaggle/working/kaggle_train_worker.py')
train_file=Path(find_file('deberta_fg_train.csv')); dev_file=Path(find_file('deberta_fg_dev.csv'))
assert train_file.parent==dev_file.parent
micro=int(profile['micro_batch_per_gpu']); accum=int(profile['gradient_accumulation']); ngpu=int(profile['gpu_count'])
cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone',f'--nproc_per_node={ngpu}',str(worker_path),'--mode','full','--data-root',str(train_file.parent),'--batch',str(micro),'--accum',str(accum)]
training_log=Path('/kaggle/working/ddp_training_output.log')
print('CHECKPOINT full_training_begin',json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'micro_batch_per_gpu':micro,'gradient_accumulation':accum,'gpu_count':ngpu,'effective_batch':micro*accum*ngpu}),flush=True)
returncode,full_output=run_streaming(cmd,'full-training',training_log,heartbeat_seconds=30,timeout_seconds=None)
if returncode: raise RuntimeError('DDP full training failed; inspect /kaggle/working/ddp_training_output.log\\n'+full_output[-5000:])
print('CHECKPOINT full_training_complete',flush=True)
out=Path('/kaggle/working/deberta_fact_grounding_checkpoint')
print('CHECKPOINT export_and_fresh_reload_begin',flush=True)
meta=json.loads((out/'training_metadata.json').read_text(encoding='utf-8'))
meta['data_audit']=audit_data; meta['runtime']=runtime; meta['profile']=profile
(out/'training_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(json.dumps(meta,indent=2))
print('CHECKPOINT export_and_fresh_reload_complete',flush=True)
"""

verify_code = """import json
from pathlib import Path
OUT=Path('/kaggle/working/deberta_fact_grounding_checkpoint')
metrics=json.loads(Path('/kaggle/working/in_domain_metrics.json').read_text(encoding='utf-8'))
summary=json.loads(Path('/kaggle/working/reproduction_summary.json').read_text(encoding='utf-8'))
reload_check=summary['reload_check']; comparison=summary['comparison']
assert reload_check['pass'],reload_check
assert len(pd.read_csv('/kaggle/working/in_domain_predictions.csv'))==6404
assert (OUT/'training_metadata.json').exists() and (OUT/'checkpoint_reload.json').exists() and (OUT/'config.json').exists()
print(json.dumps({'metrics':metrics,'paper_comparison':comparison,'checkpoint_reload':reload_check,'checkpoint_files':sorted(p.name for p in OUT.iterdir()),'checkpoint_export_seconds':summary.get('checkpoint_export_seconds'),'checkpoint_reload_seconds':summary.get('checkpoint_reload_seconds')},indent=2))
"""

train = [
md("""# Fact-Grounding Classifier: Live Training Reproduction

Trains the paper's `microsoft/deberta-v3-large` classifier on the official MuSiQue CSV split. Each notebook cell prints a timestamped checkpoint; long steps report progress while running. Training logs appear every 10 optimizer steps, validation prediction logs every 250 batches, and recoverable model checkpoints are saved each epoch with the best dev-F1 checkpoint retained. Kaggle Internet must be enabled for package, data, and base-model downloads.
"""),
code(r'''import sys, subprocess, json, platform, time, math, random, gc
from pathlib import Path
def checkpoint(name, status, **details):
    print(json.dumps({"time":time.strftime("%Y-%m-%d %H:%M:%S"),"checkpoint":name,"status":status,**details},default=str),flush=True)
checkpoint("00_environment_and_dependencies","begin")
pip_started=time.perf_counter()
subprocess.check_call([sys.executable,"-m","pip","install","-q","transformers==4.57.1","sentencepiece","protobuf","scikit-learn","accelerate"])
import numpy as np, pandas as pd, torch, transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
assert torch.cuda.is_available(), "Select a Kaggle GPU accelerator."
gpus=[]
for i in range(torch.cuda.device_count()):
    free,total=torch.cuda.mem_get_info(i); p=torch.cuda.get_device_properties(i)
    gpus.append({"name":p.name,"total_gib":round(total/2**30,2),"free_at_start_gib":round(free/2**30,2)})
runtime={"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,"transformers":transformers.__version__,"cuda":torch.version.cuda,"gpus":gpus,"seed":SEED}
checkpoint("00_environment_and_dependencies","complete",pip_seconds=round(time.perf_counter()-pip_started,2),gpu_count=len(gpus))
print(json.dumps(runtime,indent=2))'''),
md("""## Checkpoint 01 — Load and verify official splits

The files have `text,label` columns and no IDs. Kaggle input files are preferred. If the private input is not attached to this account, the notebook downloads the same files from the pinned official repository commit and verifies SHA256 before reading. Download progress is printed every 10 MiB. Exact duplicates are counted and retained. We check cross-split text leakage and conflicting labels. The formatter caps passage content at 1,500 characters; the full string also includes the question prefix.
"""),
code(r'''import hashlib, urllib.request
checkpoint("01_official_data","begin")
DATA_SPECS={
    "deberta_fg_train.csv":("https://raw.githubusercontent.com/RichardZhu123/fact-grounding-gap/3df77c2c94863ff35b398e14b37bd1e130e9d6de/deberta_fg_train.csv","7832387E08D082E4E2CA8B4161AEFEA2CC56450E003E14E312C3BC5F31C0557D"),
    "deberta_fg_dev.csv":("https://raw.githubusercontent.com/RichardZhu123/fact-grounding-gap/3df77c2c94863ff35b398e14b37bd1e130e9d6de/deberta_fg_dev.csv","F7EC0B2308EBDC811BFC11828A076F498FD2CB542C503274DAD5CBA852EBD1BC"),
}
DATA_ROOT=Path('/kaggle/working/official_training_data'); DATA_ROOT.mkdir(parents=True,exist_ok=True)
def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest().upper()
def find_file(name):
    hits=list(Path('/kaggle/input').rglob(name))
    url,expected=DATA_SPECS[name]
    if hits:
        path=hits[0]
    else:
        path=DATA_ROOT/name
        if not path.exists() or sha256_file(path)!=expected:
            checkpoint("01_official_data_download","begin",file=name,url=url)
            request=urllib.request.Request(url,headers={"User-Agent":"fact-grounding-reproduction/1.0"})
            with urllib.request.urlopen(request,timeout=60) as response, open(path,'wb') as dest:
                total=int(response.headers.get('Content-Length',0)); received=0; next_report=10*1024*1024
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk: break
                    dest.write(chunk); received+=len(chunk)
                    if received>=next_report:
                        checkpoint("01_official_data_download","progress",file=name,downloaded_mib=round(received/2**20,1),total_mib=round(total/2**20,1) if total else None)
                        next_report+=10*1024*1024
            checkpoint("01_official_data_download","complete",file=name,bytes=path.stat().st_size)
    actual=sha256_file(path)
    if actual!=expected: raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected}")
    checkpoint("01_official_data_file_verified","complete",file=name,path=str(path),sha256=actual)
    return path
train_path=find_file('deberta_fg_train.csv'); dev_path=find_file('deberta_fg_dev.csv')
checkpoint("01_official_data","reading_csv")
train=pd.read_csv(train_path,dtype={'text':str,'label':int})
dev=pd.read_csv(dev_path,dtype={'text':str,'label':int})
assert list(train.columns)==['text','label'] and list(dev.columns)==['text','label']
assert len(train)==46610 and len(dev)==6404
def audit(frame):
    g=frame.groupby('text').label.agg(['count','nunique'])
    passage=frame.text.str.split('\\n\\nPassages:\\n',n=1).str[-1]
    return {'rows':len(frame),'unique_texts':int(frame.text.nunique()),'duplicate_rows':int((g['count']-1).clip(lower=0).sum()),'conflicting_duplicate_texts':int((g['nunique']>1).sum()),'labels':{str(k):int(v) for k,v in frame.label.value_counts().sort_index().items()},'max_full_text_chars':int(frame.text.str.len().max()),'max_passage_chars':int(passage.str.len().max())}
audit_data={'train':audit(train),'dev':audit(dev),'exact_train_dev_overlap':len(set(train.text)&set(dev.text)),'id_column_present':False}
print(json.dumps(audit_data,indent=2))
assert audit_data['exact_train_dev_overlap']==0
assert audit_data['train']['max_passage_chars']<=1500 and audit_data['dev']['max_passage_chars']<=1500
checkpoint("02_dataset_audit","complete",train_rows=len(train),dev_rows=len(dev),train_dev_overlap=audit_data['exact_train_dev_overlap'])
'''),
md("""## Checkpoint 03 — Model, tokenizer, and 100-step GPU profile

The effective batch stays at 16. Kaggle reported two 14.56-GiB Tesla T4 GPUs. Earlier DataParallel profiles projected 11.39 hours through epoch 3 after dynamic padding, non-reentrant gradient checkpointing, and length-grouped batches. This run explicitly launches two `torchrun` processes so each T4 owns a distributed-training worker; it records world size, rank, effective batch, and per-GPU memory. Dynamic padding, length grouping, and the 512-token cap remain. PyTorch recommends the non-reentrant checkpoint implementation, which avoids limitations of the reentrant variant. Kaggle's observed notebook limit is 8 hours, so the notebook stops before full fine-tuning if its three-epoch projection plus 25% overhead exceeds 7 hours, leaving time for validation and checkpoint export. The paper reports convergence by epoch 3; if early stopping requires more epochs, the run may still exceed the platform limit.
"""),
code(profile_code),
md("""## Checkpoint 04 — Full training and complete dev evaluation

The official code uses an epoch cap of 10, 2e-5 learning rate, effective batch 16, 512 tokens, 10% linear warmup, and dev F1 with patience 3. The paper reports convergence at epoch 3. No examples are removed and no epoch cap is reduced. The external worker uses `torchrun` DistributedDataParallel to use both Kaggle T4 GPUs explicitly.
"""),
code(full_code),
md("""## Checkpoint 05 — Export, fresh reload, and paper comparison

The threshold stays at 0.5. A fresh model reload is checked on 32 evenly spaced dev examples. Predictions must match exactly and maximum absolute logit difference must be at most 1e-3. The report and all metrics are saved before the check can fail, so diagnostics remain available. All key outputs are written under `/kaggle/working` so Kaggle keeps them with the notebook version.
"""),
code(verify_code),

]

eval_cells = [
md("""# Cross-Domain Evaluation: Fact-Grounding Classifier

This notebook independently loads a persisted Kaggle checkpoint Dataset and prepared evaluation Dataset. It contains no training or threshold tuning. The fixed answerable threshold is 0.5. HotpotQA labels are constructed from its annotated supporting facts using paired evidence sets; SciFact verdict labels and FiQA relevance judgments are not treated as answerability labels. Unsupported datasets are explicitly reported as not evaluated.
"""),
code(r'''import sys,subprocess,json,time,random
from pathlib import Path
subprocess.check_call([sys.executable,'-m','pip','install','-q','transformers==4.57.1','sentencepiece','protobuf','scikit-learn'])
import numpy as np,pandas as pd,torch,transformers
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,balanced_accuracy_score,roc_auc_score,confusion_matrix
SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
assert torch.cuda.is_available()
print({'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})'''),
code(r'''def input_dir(fragment):
    hits=[p for p in Path('/kaggle/input').iterdir() if fragment in p.name.lower()]
    if not hits: raise FileNotFoundError(f'Attach Kaggle input dataset containing {fragment!r}')
    return hits[0]
ckpt_root=input_dir('fact-grounding-checkpoint'); eval_root=input_dir('fact-grounding-eval-data')
configs=list(ckpt_root.rglob('config.json')); ckpt=next((p.parent for p in configs if (p.parent/'model.safetensors').exists() or (p.parent/'pytorch_model.bin').exists()),None)
if ckpt is None: raise FileNotFoundError('Model checkpoint weights not found')
tokenizer=AutoTokenizer.from_pretrained(ckpt); model=AutoModelForSequenceClassification.from_pretrained(ckpt).cuda().eval()
assert model.config.label2id.get('ANSWERABLE')==1 and model.config.label2id.get('NOT-ANSWERABLE')==0
examples_file=next(eval_root.rglob('prepared_eval_examples.jsonl')); manifest_file=next(eval_root.rglob('dataset_manifest.json'))
examples=[json.loads(s) for s in examples_file.read_text(encoding='utf-8').splitlines() if s.strip()]
manifest=json.loads(manifest_file.read_text(encoding='utf-8'))
need={'id','dataset','question','passages','label','label_source','construction_method','original_document_ids'}
assert all(need<=set(e) and e['label'] in (0,1) for e in examples)
df=pd.DataFrame(examples); print(df.groupby(['dataset','label']).size().rename('n').to_frame()); print(json.dumps(manifest,indent=2))'''),
md("""## Fixed-threshold evaluation

Report positive-class precision, recall, F1, macro-F1, balanced accuracy, AUROC, 15-bin ECE, Brier score, confusion matrix, and inference latency. AUROC is omitted when only one class is present. Errors are saved per example.
"""),
code(r'''def ece(y,p,bins=15):
    edges=np.linspace(0,1,bins+1); out=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
        if m.any(): out+=m.mean()*abs(float(y[m].mean())-float(p[m].mean()))
    return float(out)
rows=[]; pred_rows=[]
for name,g in df.groupby('dataset',sort=True):
    texts=[]
    for r in g.itertuples():
        passage_text='\\n'.join(r.passages)[:1500]
        texts.append(f"Sub-question: {r.question}\\n\\nPassages:\\n{passage_text}")
    y=g.label.to_numpy(dtype=int); p=[]; t=time.perf_counter()
    for i in range(0,len(texts),16):
        x=tokenizer(texts[i:i+16],max_length=512,padding='max_length',truncation=True,return_tensors='pt').to('cuda')
        with torch.no_grad(): p.extend(torch.softmax(model(**x).logits.float(),dim=-1)[:,1].cpu().numpy().tolist())
    elapsed=time.perf_counter()-t; p=np.asarray(p); yh=(p>=.5).astype(int)
    rows.append({'dataset':name,'n':len(y),'n_positive':int(y.sum()),'n_negative':int((1-y).sum()),'threshold':.5,'accuracy':accuracy_score(y,yh),'precision':precision_score(y,yh,zero_division=0),'recall':recall_score(y,yh,zero_division=0),'f1':f1_score(y,yh,zero_division=0),'macro_f1':f1_score(y,yh,average='macro',zero_division=0),'balanced_accuracy':balanced_accuracy_score(y,yh),'auroc':roc_auc_score(y,p) if len(np.unique(y))==2 else None,'ece_15':ece(y,p),'brier_score':float(np.mean((p-y)**2)),'confusion_matrix_0_1':confusion_matrix(y,yh,labels=[0,1]).tolist(),'inference_seconds':elapsed,'latency_ms_per_example':elapsed*1000/len(y),'status':'evaluated'})
    for e,prob,hat in zip(g.to_dict('records'),p,yh): pred_rows.append({**e,'prob_answerable':float(prob),'prediction':int(hat),'correct':int(hat==e['label'])})
for name in ['MuSiQue','HotpotQA','SciFact','FiQA-2018']:
    if not any(r['dataset']==name for r in rows): rows.append({'dataset':name,'n':0,'status':'not_evaluated','reason':manifest.get('evaluation_data',{}).get('not_evaluated',{}).get(name,'No compatible labeled examples attached.'),'threshold':.5})
out=Path('/kaggle/working/results'); out.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(out/'cross_domain_metrics.csv',index=False); pd.DataFrame(pred_rows).to_csv(out/'predictions.csv',index=False)
'''),
md("""## Question-only ablation and failure cases

Question-only is an intentionally out-of-distribution diagnostic. It does not change weights or threshold. False positives and false negatives are exported for review.
"""),
code(r'''abl=[]
for name,g in df.groupby('dataset'):
    texts=[f'Sub-question: {q}\\n\\nPassages:\\n' for q in g.question]
    y=g.label.to_numpy(dtype=int); p=[]
    for i in range(0,len(texts),16):
        x=tokenizer(texts[i:i+16],max_length=512,padding='max_length',truncation=True,return_tensors='pt').to('cuda')
        with torch.no_grad(): p.extend(torch.softmax(model(**x).logits.float(),dim=-1)[:,1].cpu().numpy().tolist())
    yh=(np.asarray(p)>=.5).astype(int)
    abl.append({'dataset':name,'condition':'question_only_ood','n':len(y),'accuracy':accuracy_score(y,yh),'f1':f1_score(y,yh,zero_division=0),'macro_f1':f1_score(y,yh,average='macro',zero_division=0)})
pd.DataFrame(abl).to_csv(out/'question_only_ablation.csv',index=False)
pred=pd.DataFrame(pred_rows)
if len(pred):
    fail=pred[pred.correct==0].copy(); fail['error_type']=np.where((fail.label==0)&(fail.prediction==1),'false_positive_missing_evidence','false_negative_sufficient_evidence')
else: fail=pred
fail.to_csv(out/'failure_cases.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False)); print('failure_cases=',len(fail))'''),
]

save('01_train_fact_grounding_live.ipynb', train)
save('02_cross_domain_eval.ipynb', eval_cells)

verify_cells = [
md("""# Verify Persisted Fact-Grounding Checkpoint

This notebook does not train. It recomputes predictions for the official MuSiQue dev split from the persisted checkpoint in original row order, reports whether the old prediction file matches, then reloads the checkpoint and compares a fixed sample. It logs every ten inference batches and saves diagnostics before checking the reload.
"""),
code(r'''import sys,subprocess,json,time,os
from pathlib import Path
def log(stage,status,**details):
    print(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'checkpoint':stage,'status':status,**details},default=str),flush=True)
log('00_environment','begin')
subprocess.check_call([sys.executable,'-m','pip','install','-q','transformers==4.57.1','sentencepiece','protobuf','scikit-learn'])
import numpy as np,pandas as pd,torch,transformers
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score,confusion_matrix,precision_recall_fscore_support
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type=='cpu': torch.set_num_threads(min(4,os.cpu_count() or 1))
log('00_environment','complete',torch=torch.__version__,transformers=transformers.__version__,device=str(device),gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'''),
code(r'''log('01_find_persisted_checkpoint','begin')
matches=[]
for candidate in Path('/kaggle/input').iterdir():
    configs=list(candidate.rglob('config.json'))
    checkpoints=[p.parent for p in configs if (p.parent/'model.safetensors').exists() or (p.parent/'pytorch_model.bin').exists()]
    preferred=next((p for p in checkpoints if p.name=='deberta_fact_grounding_checkpoint'),None)
    checkpoint=preferred or (checkpoints[0] if checkpoints else None)
    pred_files=list(candidate.rglob('in_domain_predictions.csv'))
    metric_files=list(candidate.rglob('in_domain_metrics.json'))
    if checkpoint is not None and pred_files and metric_files: matches.append((candidate,checkpoint,pred_files[0],metric_files[0]))
if len(matches)!=1: raise FileNotFoundError(f'Expected one attached training-output/checkpoint input with model weights and in-domain metrics, found {[m[0].name for m in matches]}')
root,checkpoint,pred_file,metric_file=matches[0]
pred=pd.read_csv(pred_file); metrics=json.loads(metric_file.read_text(encoding='utf-8'))
required={'text','label','prob_answerable','prediction'}
if not required<=set(pred.columns): raise ValueError(f'Missing prediction columns: {required-set(pred.columns)}')
assert len(pred)==6404, f'Expected 6404 saved dev predictions, got {len(pred)}'
dev_files=list(root.rglob('deberta_fg_dev.csv'))
if len(dev_files)!=1: raise FileNotFoundError(f'Expected one official dev split in {root}, found {dev_files}')
dev=pd.read_csv(dev_files[0],dtype={'text':str,'label':int})
if not {'text','label'}<=set(dev.columns): raise ValueError(f'Official dev split is missing text/label: {dev.columns.tolist()}')
if len(dev)!=6404: raise ValueError(f'Expected 6404 official dev rows, found {len(dev)}')
same_source_order=(len(pred)==len(dev) and pred.text.fillna('').equals(dev.text.fillna('')) and np.array_equal(pred.label.to_numpy(dtype=int),dev.label.to_numpy(dtype=int)))
log('01_find_persisted_checkpoint','complete',input=root.name,checkpoint=str(checkpoint),checkpoint_files=sorted(p.name for p in checkpoint.iterdir()),saved_predictions=len(pred),official_dev_rows=len(dev),saved_rows_match_dev_order=same_source_order,saved_metrics=metrics)'''),
code(r'''log('02_load_checkpoint','begin',checkpoint=str(checkpoint),device=str(device))
stage=time.perf_counter(); tokenizer=AutoTokenizer.from_pretrained(checkpoint)
log('02a_load_tokenizer','complete',seconds=round(time.perf_counter()-stage,2))
stage=time.perf_counter(); model=AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device).eval()
if model.config.label2id.get('ANSWERABLE')!=1 or model.config.label2id.get('NOT-ANSWERABLE')!=0: raise ValueError(f'Unexpected label mapping: {model.config.label2id}')
log('02b_load_model','complete',seconds=round(time.perf_counter()-stage,2),parameters=sum(p.numel() for p in model.parameters()))
batch_size=8 if device.type=='cuda' else 4
dev_text=dev.text.fillna('').astype(str).tolist(); probabilities=[]; started=time.perf_counter()
total_batches=(len(dev_text)+batch_size-1)//batch_size
for batch_no,start in enumerate(range(0,len(dev_text),batch_size),1):
    batch_texts=dev_text[start:start+batch_size]
    inputs=tokenizer(batch_texts,max_length=512,padding=True,pad_to_multiple_of=8 if device.type=='cuda' else None,truncation=True,return_tensors='pt').to(device)
    with torch.inference_mode(): probs=torch.softmax(model(**inputs).logits.float(),dim=-1)[:,1].cpu().numpy()
    probabilities.extend(probs.tolist())
    if batch_no%10==0 or batch_no==total_batches: log('03_full_dev_inference','progress',completed_rows=min(start+len(batch_texts),len(dev)),total_rows=len(dev),batch=batch_no,total_batches=total_batches,elapsed_seconds=round(time.perf_counter()-started,1))
fresh=np.asarray(probabilities,dtype=float); fresh_labels=(fresh>=0.5).astype(int); gold=dev.label.to_numpy(dtype=int)
precision,recall,f1,_=precision_recall_fscore_support(gold,fresh_labels,average='binary',pos_label=1,zero_division=0)
fresh_metrics={'dataset':'MuSiQue dev','n':len(dev),'n_positive':int(gold.sum()),'n_negative':int((gold==0).sum()),'threshold':0.5,'accuracy':float(accuracy_score(gold,fresh_labels)),'precision':float(precision),'recall':float(recall),'f1':float(f1),'confusion_matrix_labels_0_1':confusion_matrix(gold,fresh_labels,labels=[0,1]).tolist()}
fresh_predictions=dev[['text','label']].copy(); fresh_predictions['prob_answerable']=fresh; fresh_predictions['prediction']=fresh_labels
fresh_predictions.to_csv('/kaggle/working/recomputed_in_domain_predictions.csv',index=False)
Path('/kaggle/working/recomputed_in_domain_metrics.json').write_text(json.dumps(fresh_metrics,indent=2),encoding='utf-8')
saved_text_match=(len(pred)==len(dev) and pred.text.fillna('').equals(dev.text.fillna('')))
saved_label_match=(len(pred)==len(dev) and np.array_equal(pred.label.to_numpy(dtype=int),gold))
saved_prob=pred.prob_answerable.to_numpy(dtype=float) if saved_text_match and saved_label_match else np.full(len(dev),np.nan)
saved_diff=np.abs(fresh-saved_prob) if np.isfinite(saved_prob).all() else np.array([],dtype=float)
indices=np.linspace(0,len(dev)-1,num=min(32,len(dev)),dtype=int).tolist()
pd.DataFrame({'sample_index':indices,'label':gold[indices],'saved_prediction':pred.prediction.to_numpy(dtype=int)[indices] if saved_label_match else np.full(len(indices),-1),'fresh_prediction':fresh_labels[indices],'saved_probability':saved_prob[indices] if len(saved_prob) else np.full(len(indices),np.nan),'fresh_probability':fresh[indices],'abs_probability_diff':saved_diff[indices] if len(saved_diff) else np.full(len(indices),np.nan)}).to_csv('/kaggle/working/checkpoint_reload_samples.csv',index=False)
saved_metrics_delta={key:float(fresh_metrics[key]-metrics[key]) for key in ['accuracy','precision','recall','f1'] if key in metrics}
reload_indices=np.array(sorted(set([0,len(dev)//2,len(dev)-1])),dtype=int); reference=fresh[reload_indices].copy(); del model
if device.type=='cuda': torch.cuda.empty_cache()
log('04_reload_checkpoint','begin',sample_indices=reload_indices.tolist())
stage=time.perf_counter(); reload_model=AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device).eval(); reload_probs=[]
with torch.inference_mode():
    for idx in reload_indices:
        inputs=tokenizer(dev_text[idx],max_length=512,padding='max_length',truncation=True,return_tensors='pt').to(device)
        reload_probs.append(float(torch.softmax(reload_model(**inputs).logits.float(),dim=-1)[0,1].cpu()))
reload_diff=float(np.max(np.abs(reference-np.asarray(reload_probs))))
report={'checkpoint_dir':str(checkpoint),'device':str(device),'sample_count':len(indices),'full_dev_inference_seconds':round(time.perf_counter()-started,2),'fresh_dev_metrics':fresh_metrics,'saved_dev_metrics':metrics,'fresh_minus_saved_metric_delta':saved_metrics_delta,'saved_text_order_matches_dev':saved_text_match,'saved_labels_match_dev':saved_label_match,'saved_predictions_match_fresh':bool(saved_text_match and saved_label_match and np.array_equal(pred.prediction.to_numpy(dtype=int),fresh_labels)),'max_abs_saved_probability_diff':float(np.max(saved_diff)) if len(saved_diff) else None,'reload_sample_indices':reload_indices.tolist(),'reload_max_abs_probability_diff':reload_diff,'reload_tolerance':0.001,'checkpoint_reload_pass':bool(reload_diff<=0.001),'recomputed_predictions_path':'/kaggle/working/recomputed_in_domain_predictions.csv','recomputed_metrics_path':'/kaggle/working/recomputed_in_domain_metrics.json','sample_diagnostics_path':'/kaggle/working/checkpoint_reload_samples.csv'}
report_path=Path('/kaggle/working/checkpoint_reload_report.json'); report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
log('05_report_saved','complete',report_path=str(report_path),metrics_path='/kaggle/working/recomputed_in_domain_metrics.json',predictions_path='/kaggle/working/recomputed_in_domain_predictions.csv',checkpoint_reload_pass=report['checkpoint_reload_pass'],fresh_metrics=fresh_metrics,saved_predictions_match_fresh=report['saved_predictions_match_fresh'],max_abs_saved_probability_diff=report['max_abs_saved_probability_diff'],reload_max_abs_probability_diff=reload_diff)
if not report['checkpoint_reload_pass']: raise RuntimeError(f'Fresh checkpoint reload changed sampled probabilities; report saved at {report_path}: {report}')
log('06_run_complete','complete',report_path=str(report_path))'''),
]
save('03_verify_persisted_checkpoint.ipynb', verify_cells)
print('Wrote notebooks to', ROOT)
