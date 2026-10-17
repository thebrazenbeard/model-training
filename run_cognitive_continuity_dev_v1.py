"""Run a frozen synthetic development evaluation, without loading a model.

python -B run_cognitive_continuity_dev_v1.py \\
    --mono-root <exact-source-worktree> --fixtures fixtures/cognitive_continuity_devset_v1.json

Candidate structured predictions live separately from evaluator-only expected
outcomes in the in-memory scoring call; the entire developer fixture file is
public, NOT held-out model qualification.
"""
from __future__ import annotations
import argparse
from dataclasses import fields,replace
import json
from pathlib import Path
import sys
from cognitive_continuity_eval_v1 import (
    CognitiveSource,CognitiveObservation,CognitiveExpectation,
    CognitivePrediction,CognitiveEvaluationError,
    run_cognitive_evaluation,guard_prediction,
)

def load_development_fixture(path: Path):
    if not isinstance(path,Path) or not path.is_file() or path.stat().st_size>160000:
        raise CognitiveEvaluationError("missing or overlarge development fixtures")
    data=json.loads(path.read_text("utf-8"))
    if type(data) is not dict or set(data)!={"schema","purpose","limitations","cases"} or (
       data["schema"]!="VERAOS_COGNITIVE_CONTINUITY_SYNTHETIC_DEVSET_V1" or
       type(data["cases"]) is not list or not 1<=len(data["cases"])<=64):
        raise CognitiveEvaluationError("unknown frozen development schema")
    public=[]; hidden=[]; candidate={}
    for item in data["cases"]:
        if type(item) is not dict:
            raise CognitiveEvaluationError("invalid case")
        i=dict(item)
        source=i.pop("sources")
        expected=i.pop("expected")
        prediction=i.pop("prediction")
        if set(i)!={x.name for x in fields(CognitiveObservation)}-{"sources"}:
            raise CognitiveEvaluationError("invalid case fields")
        if type(source) is not list or len(source)>20:
            raise CognitiveEvaluationError("invalid bounded event history")
        observation=CognitiveObservation(sources=tuple(CognitiveSource(**x) for x in source),**i)
        want=CognitiveExpectation(**expected)
        output=CognitivePrediction(task_id=observation.task_id,
            request_id=observation.request_id,**prediction)
        if observation.case_id in candidate:
            raise CognitiveEvaluationError("duplicate candidate case")
        public.append(observation);hidden.append(want);candidate[observation.case_id]=output
    return tuple(public),tuple(hidden),candidate

def load_external_predictions(path: Path, observations: tuple[CognitiveObservation,...]):
    if not isinstance(path,Path) or not path.is_file() or path.stat().st_size>80000:
        raise CognitiveEvaluationError("external predictions missing/overlarge")
    value=json.loads(path.read_text("utf-8"))
    if (type(value) is not dict or set(value)!={"schema","predictions"} or
        value["schema"]!="VERAOS_COGNITIVE_STRUCTURED_PREDICTIONS_V1" or
        type(value["predictions"]) is not list):
        raise CognitiveEvaluationError("invalid external prediction schema")
    cases={obs.case_id:obs for obs in observations}
    if len(value["predictions"])!=len(cases):
        raise CognitiveEvaluationError("prediction cardinality mismatch")
    result={}
    for element in value["predictions"]:
        if (type(element) is not dict or
            set(element)!={"case_id","status","task_id","request_id",
                           "referent","scope","claimed_value","source_event_id",
                           "effect_authority","independent_model_claim"}):
            raise CognitiveEvaluationError("invalid external prediction fields")
        item=dict(element)
        case_id=item.pop("case_id")
        if type(case_id) is not str or case_id not in cases or case_id in result:
            raise CognitiveEvaluationError("external case identity mismatch")
        # Do not repair forged task binding here: allow guard/score to catch it.
        result[case_id]=CognitivePrediction(**item)
    return result

def evaluate(path:Path,mono_root:Path,*,predictions:Path|None=None)->dict:
    paths=tuple(sorted((mono_root/"packages").glob("*/src")))
    if not paths or not (mono_root/"packages"/"vera_core"/"src"/"vera_core"/"prequential_evaluation.py").is_file():
        raise CognitiveEvaluationError("exact local Mono source root required")
    for item in paths:sys.path.insert(0,str(item))
    from vera_core.interaction_semantics import interpret_utterance
    public,hidden,candidate=load_development_fixture(path)
    if predictions is not None:
        candidate=load_external_predictions(predictions,public)
    def predict(obs):
        return candidate[obs.case_id]
    def governed(obs,result):
        return guard_prediction(obs,result,interpret=lambda text,prev:
            interpret_utterance(text,previous_interaction_id=prev))
    baseline=run_cognitive_evaluation(public,hidden,predict)
    guarded=run_cognitive_evaluation(public,hidden,predict,guard=governed)
    return {
        "schema":"VERAOS_COGNITIVE_CONTINUITY_DEV_COMPARISON_V1",
        "prediction_origin":"STRUCTURED_EXTERNAL_FILE" if predictions is not None else "SCRIPTED_DEV_FIXTURE",
        "evaluator_source":"vera_core.prequential_evaluation",
        "interpretation_source":"vera_core.interaction_semantics",
        "baseline":baseline,
        "source_guard":guarded,
        "delta_passes":guarded["passes"]-baseline["passes"],
        "caveats":["frozen synthetic fixtures","scripted predictions not live model outputs",
                   "guard withholds unjustified claims, does not produce new answers",
                   "not evidence of model learning, consciousness or real OS consumption"],
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mono-root",type=Path,required=True)
    parser.add_argument("--fixtures",type=Path,
        default=Path(__file__).resolve().parent/"fixtures"/"cognitive_continuity_devset_v1.json")
    parser.add_argument("--predictions",type=Path,default=None,
        help="Optional precomputed structured outputs; does not load or run a model")
    args=parser.parse_args()
    result=evaluate(args.fixtures,args.mono_root,predictions=args.predictions)
    print(json.dumps(result,sort_keys=True,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
