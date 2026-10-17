"""Bounded READ-ONLY GGUF metadata/weight-layer feasibility inspection.

No imports from model repos, inference, mmap of full weights, model loading,
GPU allocation or provider selection. Budget is illustrative, NOT evidence
of successful backend layer offload or an admitted model process.
"""
from __future__ import annotations
import argparse
import json
import re
import struct
from pathlib import Path

MODELS={
 "qwen3_4b_ollama": "ollama/blobs/sha256-85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9",
 "qwen35_4b_q5ks": "gguf/model_q5_k_s.gguf",
 "glm24b_iq2xs": "gguf/vera-glm24b-iq2-xs/Dolphin-Mistral-GLM-4.7-Flash-24B-Venice-Edition-Thinking-Uncensored.i1-IQ2_XS.gguf",
 "glm24b_q4km": "gguf/vera-glm24b-q4-k-m/Dolphin-Mistral-GLM-4.7-Flash-24B-Venice-Edition-Thinking-Uncensored.i1-Q4_K_M.gguf",
}
FLOAT_TYPES={0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}
def inspect(path:Path, *, vram_free_mib:int=3954, reserve_mib:int=768):
    def exact(f,n):
        value=f.read(n)
        if len(value)!=n:raise ValueError("truncated GGUF metadata")
        return value
    with path.open("rb") as f:
        if exact(f,4)!=b"GGUF":raise ValueError("not GGUF")
        version=struct.unpack("<I",exact(f,4))[0]
        tensor_count,kv_count=struct.unpack("<QQ",exact(f,16))
        if version not in (2,3) or tensor_count>100000 or kv_count>20000:
            raise ValueError("implausible header")
        def number(fmt):
            return struct.unpack("<"+fmt,exact(f,struct.calcsize(fmt)))[0]
        def string():
            n=number("Q")
            if n>4_000_000:raise ValueError("unbounded metadata string")
            return exact(f,n).decode("utf8","replace")
        def value(typ):
            if typ==8:return string()
            if typ in FLOAT_TYPES:return number(FLOAT_TYPES[typ])
            if typ==9:
                member=number("I");n=number("Q")
                if n>1000000:raise ValueError("unbounded metadata array")
                if member in FLOAT_TYPES:
                    f.seek(n*struct.calcsize(FLOAT_TYPES[member]),1)
                elif member==8:
                    for _ in range(n):string()
                else:raise ValueError("unknown array member type")
                return f"array[{n}]"
            raise ValueError("unknown metadata type")
        md={}
        for _ in range(kv_count):
            key=string()
            val=value(number("I"))
            if isinstance(val,(int,float,str)) and len(str(val))<300:
                md[key]=val
        tensor_infos=[]
        for _ in range(tensor_count):
            name=string(); dims=number("I")
            if dims>8:raise ValueError("unbounded dimensions")
            for _ in range(dims):number("Q")
            quant=number("I");off=number("Q")
            tensor_infos.append((name,off,quant))
        alignment=int(md.get("general.alignment",32))
        if alignment<=0 or alignment>4096:raise ValueError("invalid alignment")
        at=f.tell();data_start=(at+alignment-1)//alignment*alignment
    nbytes=path.stat().st_size
    ordered=sorted(tensor_infos,key=lambda x:x[1])
    layers={}
    nonlayer=0
    for i,(name,off,quant) in enumerate(ordered):
        size=(ordered[i+1][1] if i+1<len(ordered) else nbytes-data_start)-off
        if size<0 or size>nbytes:raise ValueError("invalid tensor offset")
        match=re.search(r"(?:^|\.)blk\.(\d+)\.",name)
        if match:
            k=int(match.group(1))
            layers[k]=layers.get(k,0)+size
        else:nonlayer+=size
    arch=md.get("general.architecture")
    prefix=str(arch)
    block=int(md.get(prefix+".block_count",0))
    kv_heads=int(md.get(prefix+".attention.head_count_kv",0))
    keys=int(md.get(prefix+".attention.key_length",0))
    vals=int(md.get(prefix+".attention.value_length",0))
    mib=1024*1024
    kv_token=block*kv_heads*(keys+vals)*2 # illustrative FP16 K+V, NOT runtime-measured
    budget=max(0,vram_free_mib-reserve_mib)
    weight_mib=round(nbytes/mib,2)
    def scenario(context):
        kv_mib=kv_token*context/mib
        layer_budget_mib=max(0,budget-kv_mib)
        layers_sorted=sorted(layers.items(),key=lambda x:x[0])
        # Layer-only optimistic ceiling (excludes non-layer tensors, graph and
        # actual allocator). Reserve assumed separate.
        fit=0
        remaining=layer_budget_mib*mib
        for _,size in layers_sorted:
            if remaining<size:break
            remaining-=size;fit+=1
        return {"context_tokens":context,"fp16_kv_mib":round(kv_mib,1),
                "weight_budget_after_reserve_and_kv_mib":round(layer_budget_mib,1),
                "optimistic_layer_only_fit_of":fit,
                "total_layers":block,
                "full_weight_with_reserved_vram_budget":weight_mib<=layer_budget_mib}
    return {"path":str(path),"file_bytes":nbytes,
            "weight_gib":round(nbytes/1024**3,3),
            "gguf_architecture":arch,"gguf_type":md.get("general.type"),
            "gguf_file_type":md.get("general.file_type"),
            "gguf_name":md.get("general.name"),"gguf_version":version,
            "tensor_count":tensor_count,"declared_layers":block,
            "parsed_layer_count":len(layers),"unlayered_disk_mib":round(nonlayer/mib,1),
            "gpu_free_assumption_mib":vram_free_mib,
            "runtime_reserve_assumption_mib":reserve_mib,
            "fp16_kv_estimate_only":True,
            "scenarios":[scenario(t) for t in (2048,4096,8192)],
            "observed_backend_offload_layers":None,
            "provider_admitted":False}
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--models-root",default=r"D:\VERA\models")
    parser.add_argument("--free-vram-mib",type=int,default=3954)
    parser.add_argument("--reserve-mib",type=int,default=768)
    args=parser.parse_args()
    result={}
    for key,relative in MODELS.items():
        path=Path(args.models_root)/relative
        result[key]=inspect(path,vram_free_mib=args.free_vram_mib,
                            reserve_mib=args.reserve_mib) if path.is_file() else {"missing":relative}
    print(json.dumps({"schema":"VERA_TRAINING_GGUF_PLACEMENT_METADATA_V1",
            "models":result,"source_only":True,"inference_executed":False,
            "gpu_allocated":False,"full_gguf_weight_hash_recomputed":False,
            "limits":"tensor disk sizes are not live allocator weights; optimistic layer bound is not measured offload; KV FP16 no runtime allocations; CPU RAM availability varies"},indent=2,sort_keys=True))
if __name__=="__main__":main()
