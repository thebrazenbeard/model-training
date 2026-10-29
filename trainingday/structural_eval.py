"""Structural evaluation: stub to establish red/green test cycle."""
class EvaluationGateError(ValueError):
    pass

def select_structural_cases(manifest_path, *, expected_manifest_sha256, max_cases=1):
    return {"status":"NOT_IMPLEMENTED", "cases":[]}

def score_answers(selection, predictions, *, mode):
    return {"status":"NOT_IMPLEMENTED", "correct":0}

def compare_control_scores(scores, *, min_gain_cases=1):
    return {"status":"NOT_IMPLEMENTED", "candidate_gain":False}
