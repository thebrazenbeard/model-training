"""Bounded Vera R11 development-only heldout generation probe.

Single GPU owner, strictly no optimizer or weights save. 3 original heldout
voice prompts excluded from corpus, no system-message coaching. Results
are exploratory text samples and do not qualify either model.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(r"D:\VERA")
SOURCE=Path(__file__).resolve().parents[2]
MODEL=ROOT/"models"/"latest-trained"/"base"
VOICE=ROOT/"portal-task-state"/"vera-model-glm24b-lappy-20261012"/"evaluation"/"cases"/"vera_voice_heldout_v1.jsonl"
R11=ROOT/"models"/"adapters"/"v10r3r11-vera-voice-qwen35-r10-resume-20261015-pagefile-retry1"/"adapter"
DIR=ROOT/"portal-task-state"
PREFLIGHT=DIR/"VERA_R11_VOICE_GENERATION_PREFLIGHT_20261015_V1.json"
OUT=DIR/"VERA_R11_VOICE_GENERATION_PROBE_20261015_V1.json"
STATUS=DIR/"VERA_R11_VOICE_GENERATION_PARTIAL_20261015_V1.jsonl"
CASES=("VERA_VOICE_HELDOUT_001","VERA_VOICE_HELDOUT_007","VERA_VOICE_HELDOUT_011")
SAMPLE_SHA="bd2b5375e2b7da78f42ed3ac1387f73ddec4807bc5e446b228bdbb7db3caf25f"
R11_SHA="45f056297a55e19b22918d5298bcd5a9808fb0d8b8f5bf4210ebac91e4b88e10"

def digest(path:Path):
 h=hashlib.sha256()
 with path.open("rb") as fd:
  for chunk in iter(lambda:fd.read(4*1024*1024),b""):
   h.update(chunk)
 return h.hexdigest()

def stat_resources():
 cmd=("$o=Get-CimInstance Win32_OperatingSystem;"
      "[pscustomobject]@{ram=[double]$o.FreePhysicalMemory/1048576;"
      "virtual=[double]$o.FreeVirtualMemory/1048576}|ConvertTo-Json -Compress")
 p=subprocess.run(["powershell.exe","-NoProfile","-Command",cmd],
    capture_output=True,text=True,timeout=20,check=True)
 result=json.loads(p.stdout)
 gpu=subprocess.run(["nvidia-smi","--query-gpu=memory.free",
                      "--format=csv,noheader,nounits"],
                      capture_output=True,text=True,timeout=15,check=True)
 lines=[s.strip() for s in gpu.stdout.splitlines() if s.strip()]
 if len(lines)!=1 or not lines[0].isdigit():
  raise RuntimeError("Ambiguous or unavailable GPU")
 return {"ram_free_gib":float(result["ram"]),
         "virtual_free_gib":float(result["virtual"]),
         "gpu_free_mib":int(lines[0])}

def ensure_preflight():
 if OUT.exists() or STATUS.exists() or PREFLIGHT.exists():
  raise RuntimeError("Probe already attempted; no overwrite")
 if not sys.executable.lower().startswith(r"d:\python\envs\vera-training-20261014"):
  raise RuntimeError("Wrong environment: expected D-verified Python")
 if not SOURCE.is_dir() or not MODEL.is_dir():
  raise RuntimeError("Frozen Qwen model source missing")
 if json.loads((MODEL/"config.json").read_text(encoding="utf8")).get("model_type")!="qwen3_5":
  raise RuntimeError("Wrong base model: Qwen3.5 required")
 for label,p,expected in (
   ("voice",VOICE,SAMPLE_SHA),
   ("r11",R11/"adapter_model.safetensors",R11_SHA),
 ):
  if digest(p)!=expected:raise RuntimeError("Wrong "+label+" input hash")
 rows=[json.loads(line) for line in VOICE.read_text(encoding="utf8").splitlines() if line.strip()]
 if len(rows)!=25 or any(r["training_use"]!="PROHIBITED" for r in rows):
  raise RuntimeError("Wrong holdout panel")
 chosen=[next((r for r in rows if r["id"]==name),None) for name in CASES]
 if any(r is None for r in chosen):
  raise RuntimeError("One of frozen case identifiers is missing")
 for r in chosen:
  if r["mode"]!="NO_SYSTEM_FIRST" or not r["user"]:
   raise RuntimeError("Probe violates no-system challenge contract")
 resources=stat_resources()
 if resources["gpu_free_mib"] < 3900 or resources["ram_free_gib"] < 7 or resources["virtual_free_gib"] < 6:
  raise RuntimeError("Unsafe resource headroom for local model load: "+str(resources))
 result={
  "schema":"VERA_R11_HELDOUT_GENERATION_PREFLIGHT_V1",
  "status":"INPUTS_AND_LOCAL_RESOURCE_HEADROOM_VERIFIED",
  "model_type":"qwen3_5","base_path":str(MODEL),
  "candidates":[{"name":"r11","adapter_dir":str(R11),"sha256":R11_SHA}],
  "heldout_source_sha256":SAMPLE_SHA,
  "selected_case_ids":CASES,
  "selected_prompt_count":len(CASES),
  "no_system_prompt":True,
  "generation":{"max_new_tokens":72,"do_sample":False,"max_time_seconds":90},
  "resources":resources,
  "optimizer_step_count":0,
  "classification":"DEVELOPMENT_TEXT_PROBE_ONLY_NOT_INDEPENDENT_QUALIFICATION"
 }
 with PREFLIGHT.open("x",encoding="utf8") as out:
  json.dump(result,out,indent=2);out.write("\n")
 print("PREFLIGHT_PASS",resources,flush=True)
 return chosen,result

def load_stack():
 sys.path.insert(0,str(SOURCE))
 from successor.experiments.train_v10_qwen35_authorized import load_training_stack
 return load_training_stack()

def run_candidate(name,adapter_path,selected,stack,tokenizer):
 torch=stack["torch"]
 peft=stack["peft"]
 quant=stack["BitsAndBytesConfig"](
  load_in_4bit=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_use_double_quant=True,
  bnb_4bit_compute_dtype=torch.bfloat16,
 )
 print("LOADING",name,flush=True)
 start=time.perf_counter()
 model=stack["Qwen3_5ForCausalLM"].from_pretrained(
  MODEL,quantization_config=quant,device_map={"":0},dtype=torch.bfloat16
 )
 model=peft.PeftModel.from_pretrained(
  model,adapter_path,is_trainable=False,autocast_adapter_dtype=False)
 model.eval()
 print("LOADED",name,"seconds",round(time.perf_counter()-start,1),flush=True)
 cases=[]
 with torch.inference_mode():
  for r in selected:
   torch.cuda.synchronize()
   begin=time.perf_counter()
   encoded=tokenizer.apply_chat_template(
    [{"role":"user","content":r["user"]}],
    tokenize=True,add_generation_prompt=True,enable_thinking=False,
    return_tensors="pt",
   )
   if hasattr(encoded, 'keys'):
    encoded=encoded['input_ids']
   if encoded.ndim!=2 or encoded.size(1)>512:
    raise RuntimeError("Unexpected prompt length for "+r["id"])
   token_count=encoded.shape[1]
   encoded=encoded.to("cuda")
   generated=model.generate(
    input_ids=encoded,
    attention_mask=torch.ones_like(encoded),
    do_sample=False,
    max_new_tokens=72,
    max_time=90,
    pad_token_id=tokenizer.eos_token_id,
   )
   new_ids=generated[0,token_count:]
   response=tokenizer.decode(new_ids,skip_special_tokens=True).strip()
   record={"candidate":name,"case_id":r["id"],"axis":r["axis"],
           "prompt":r["user"],"generated_text":response,
           "prompt_token_count":int(token_count),
           "new_token_count":int(new_ids.numel()),
           "elapsed_seconds":round(time.perf_counter()-begin,2),
           "does_not_establish_scoring":True}
   cases.append(record)
   with STATUS.open("a",encoding="utf8") as output:
    output.write(json.dumps(record,ensure_ascii=False)+"\n")
    output.flush();os.fsync(output.fileno())
   print("GENERATED",name,r["id"],"TOKENS",record["new_token_count"],
         "SECS",record["elapsed_seconds"],flush=True)
   del encoded,generated,new_ids
 del model
 gc.collect()
 torch.cuda.empty_cache()
 torch.cuda.synchronize()
 print("UNLOADED",name,flush=True)
 return cases

def main():
 selected,preflight=ensure_preflight()
 stack=load_stack()
 torch=stack["torch"]
 tokenizer=stack["AutoTokenizer"].from_pretrained(MODEL)
 if tokenizer.pad_token_id is None:
  tokenizer.pad_token=tokenizer.eos_token
 all_cases=[]
 for name,adapter in (("r11",R11),):
  all_cases+=run_candidate(name,adapter,selected,stack,tokenizer)
 expected={(name,case) for name in ("r11",) for case in CASES}
 observed={(r["candidate"],r["case_id"]) for r in all_cases}
 if observed!=expected:raise RuntimeError("Incomplete paired candidate sample")
 result={
  "schema":"VERA_R11_HELDOUT_TEXT_DIAGNOSTIC_V1",
  "status":"R11_GENERATION_SAMPLES_COMPLETE_NOT_SEMANTICALLY_SCORED",
  "claim_ceiling":"EXPLORATORY_HELDOUT_TEXT_COMPARISON_NOT_MODEL_QUALIFICATION",
  "model_type":"qwen3_5","base_path":str(MODEL),
  "preflight_receipt_sha256":digest(PREFLIGHT),
  "heldout_sha256":SAMPLE_SHA,
  "candidate_r11_sha256":R11_SHA,
  "generation":preflight["generation"],
  "results":all_cases,
  "optimizer_steps_performed":0,
  "model_files_mutated":False,
  "score_requires_independent_reviewer":True
 }
 with OUT.open("x",encoding="utf8") as output:
  json.dump(result,output,indent=2,sort_keys=True,ensure_ascii=False)
  output.write("\n")
 print("R11_DIAGNOSTIC_COMPLETE",OUT,flush=True)

if __name__=="__main__":
 main()
