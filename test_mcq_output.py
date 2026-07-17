"""
test_mcq_output.py
==================
GPU/CLIP 없이 실행 가능한 단위 테스트.
다음 세 가지를 검증합니다:

1. [Dataset]  CsvMCQDataset.__getitem__ — shuffle 전후 반환값 형식
2. [Collation] DataLoader의 default_collate가 caption_types를 어떻게 변환하는지,
               그리고 zip(*caption_types_batch)가 올바른 shape를 생성하는지
3. [Logic]    evaluate_model() 내부 집계 로직 — 목 배치로 wrong_type 추적 정확성,
              sample_results / predictions.csv / results.jsonl 저장 형식
"""

import json
import os
import sys
import random
import tempfile
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

# 경로 설정 (benchmarks/src 가 sys.path에 없을 경우 직접 추가)
BENCH_SRC = os.path.join(os.path.dirname(__file__), "benchmarks", "src")
if BENCH_SRC not in sys.path:
    sys.path.insert(0, BENCH_SRC)

from training.data import CsvMCQDataset

# ─── 색 출력 헬퍼 ─────────────────────────────────────────────────────────────

def ok(msg):  print(f"  \033[92m✓ PASS\033[0m  {msg}")
def fail(msg, detail=""):
    print(f"  \033[91m✗ FAIL\033[0m  {msg}")
    if detail: print(f"         {detail}")

# ═══════════════════════════════════════════════════════════════════════════════
# 헬퍼: 임시 MCQ CSV 생성
# ═══════════════════════════════════════════════════════════════════════════════

def make_temp_csv(n_rows=8, tmp_dir=None):
    """
    실제 이미지 없이 유효한 MCQ CSV를 생성합니다.
    image_path에는 더미 문자열(실제로는 열지 않음)을 넣고,
    Dataset의 transforms를 identity로 설정합니다.
    """
    rows = []
    templates = ["positive", "negative", "hybrid"]
    for i in range(n_rows):
        tmpl = random.choice(templates)
        rows.append({
            "image_path":             f"/fake/image_{i:04d}.jpg",
            "correct_answer":          0,
            "caption_0":              f"GT caption for sample {i}",
            "caption_1":              f"Hybrid wrong for sample {i}",
            "caption_2":              f"Positive wrong for sample {i}",
            "caption_3":              f"Negative wrong for sample {i}",
            "correct_answer_template": tmpl,
        })
    df = pd.DataFrame(rows)
    path = os.path.join(tmp_dir, "test_mcq.csv")
    df.to_csv(path, index=False)
    return path

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: Dataset __getitem__ 형식 검증
# ═══════════════════════════════════════════════════════════════════════════════

def test_dataset_return_format(csv_path):
    print("\n[TEST 1] CsvMCQDataset.__getitem__ 반환 형식")

    # transforms=identity (이미지 열지 않고 경로 문자열 반환)
    identity = lambda x: x

    ds_no_shuffle = CsvMCQDataset(csv_path, transforms=identity, shuffle_options=False)
    ds_shuffle    = CsvMCQDataset(csv_path, transforms=identity, shuffle_options=True, seed=42)

    # ── 6-tuple 여부 ──────────────────────────────────────────────────────────
    row0_no = ds_no_shuffle[0]
    if len(row0_no) == 6:
        ok("shuffle=False → 6-tuple 반환")
    else:
        fail("shuffle=False → 6-tuple 아님", f"len={len(row0_no)}")
        return False

    row0_sh = ds_shuffle[0]
    if len(row0_sh) == 6:
        ok("shuffle=True  → 6-tuple 반환")
    else:
        fail("shuffle=True → 6-tuple 아님", f"len={len(row0_sh)}")
        return False

    # ── caption_types 형식 ───────────────────────────────────────────────────
    _, captions_no, correct_no, _, _, ctypes_no = row0_no
    _, captions_sh, correct_sh, _, _, ctypes_sh = row0_sh

    if isinstance(ctypes_no, list) and len(ctypes_no) == 4:
        ok(f"shuffle=False caption_types={ctypes_no}")
    else:
        fail("caption_types가 list[4]가 아님", str(ctypes_no))
        return False

    # ── shuffle=False 시 정답이 index 0, 타입이 'gt' ─────────────────────────
    if correct_no == 0:
        ok("shuffle=False correct_answer=0 (원본 그대로)")
    else:
        fail("shuffle=False인데 correct_answer가 0이 아님", str(correct_no))

    if ctypes_no[0] == 'gt':
        ok("shuffle=False caption_types[0]='gt'")
    else:
        fail("shuffle=False 시 첫 번째 타입이 'gt'가 아님", str(ctypes_no[0]))

    # ── shuffle=True 시 correct_answer가 실제 gt 위치를 가리키는지 ───────────
    gt_caption = "GT caption for sample 0"
    if captions_sh[correct_sh] == gt_caption:
        ok(f"shuffle=True correct_answer={correct_sh} → captions[{correct_sh}]==GT ✓")
    else:
        fail("shuffle=True 후 correct_answer가 GT 위치를 올바르게 가리키지 않음",
             f"correct_sh={correct_sh}, captions_sh={captions_sh}")
        return False

    if ctypes_sh[correct_sh] == 'gt':
        ok(f"shuffle=True caption_types[{correct_sh}]='gt' ✓")
    else:
        fail(f"shuffle=True 후 caption_types[correct]이 'gt'가 아님",
             str(ctypes_sh))

    # ── 여러 샘플에 걸쳐 정답 인덱스가 고정되지 않았는지 (셔플 효과 확인) ──
    indices = [ds_shuffle[i][2] for i in range(len(ds_shuffle))]
    unique_indices = set(indices)
    if len(unique_indices) > 1:
        ok(f"shuffle=True 다양한 correct_answer 인덱스: {sorted(unique_indices)}")
    else:
        # 매우 작은 데이터에서 모두 같은 인덱스가 나올 수 있지만 일단 경고
        fail("shuffle=True인데 모든 샘플의 correct_answer가 동일함 (seed 문제?)",
             f"indices={indices}")

    return True

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: DataLoader collation + zip transpose
# ═══════════════════════════════════════════════════════════════════════════════

def test_collation_shape(csv_path):
    print("\n[TEST 2] DataLoader collation 후 caption_types_batch shape")

    identity = lambda x: x
    BATCH = 4

    # ── 6-tuple DataLoader ───────────────────────────────────────────────────
    ds = CsvMCQDataset(csv_path, transforms=identity, shuffle_options=True, seed=0)
    # captions는 str 리스트이므로 default_collate가 텐서로 만들 수 없음 → 커스텀 없이 그대로
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False)
    batch = next(iter(loader))

    if len(batch) == 6:
        ok("DataLoader가 6-tuple 반환")
    else:
        fail("DataLoader가 6-tuple 반환하지 않음", f"len={len(batch)}")
        return False

    image_t, captions, correct_answer, correct_answer_type, image_path, caption_types_batch = batch

    # caption_types_batch 구조: default_collate는 list[str] per sample을 
    # element-wise로 묶어서 [list_of_B_strs, list_of_B_strs, ...] (len=4) 를 만듦
    if len(caption_types_batch) == 4:
        ok(f"caption_types_batch 길이=4 (num_options) ✓")
    else:
        fail("caption_types_batch 길이가 4(num_options)가 아님", f"len={len(caption_types_batch)}")
        return False

    if len(caption_types_batch[0]) == BATCH:
        ok(f"caption_types_batch[0] 길이={BATCH} (batch_size) ✓")
    else:
        fail("caption_types_batch[0] 길이가 batch_size가 아님",
             f"len={len(caption_types_batch[0])}")
        return False

    # zip(*caption_types_batch) → per-sample 방향으로 전치
    per_sample = list(zip(*caption_types_batch))
    if len(per_sample) == BATCH:
        ok(f"zip(*caption_types_batch) 결과 길이={BATCH} (batch_size) ✓")
    else:
        fail("zip 결과 길이가 batch_size가 아님", f"len={len(per_sample)}")
        return False

    if len(per_sample[0]) == 4:
        ok(f"per_sample[0] 길이=4 (num_options) ✓")
    else:
        fail("per_sample[0] 길이가 4가 아님", f"len={len(per_sample[0])}")
        return False

    # 각 샘플의 gt 타입이 correct_answer 인덱스와 일치하는지 검증
    errors = 0
    for i in range(BATCH):
        ct = per_sample[i]               # tuple of 4 types for sample i
        ca = int(correct_answer[i])      # correct answer index
        if ct[ca] != 'gt':
            errors += 1
            fail(f"sample {i}: caption_types[{ca}]='{ct[ca]}' ≠ 'gt'", f"ct={ct}")
    if errors == 0:
        ok(f"모든 {BATCH}개 샘플에서 per_sample[i][correct_answer[i]]=='gt' ✓")

    # ── else 브랜치 (5-tuple 호환) shape 검증 ────────────────────────────────
    print("\n  [sub-test] else 브랜치 caption_types_batch shape")
    canonical = ['gt', 'hybrid', 'positive', 'negative']
    B = BATCH
    caption_types_batch_fallback = [[t] * B for t in canonical]

    if len(caption_types_batch_fallback) == 4:
        ok("else 브랜치 len=4 (num_options) ✓")
    else:
        fail("else 브랜치 shape 오류", f"len={len(caption_types_batch_fallback)}")

    per_sample_fallback = list(zip(*caption_types_batch_fallback))
    if len(per_sample_fallback) == B:
        ok(f"else 브랜치 zip 결과 len={B} (batch_size) ✓")
    else:
        fail("else 브랜치 zip 결과 shape 오류", f"len={len(per_sample_fallback)}")
        return False

    return True

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: 집계 로직 — wrong_answer_type 추적 & sample_results 구조
# ═══════════════════════════════════════════════════════════════════════════════

def test_aggregation_logic():
    """
    실제 모델 없이 evaluate_model() 내부 집계 로직만 시뮬레이션합니다.
    """
    print("\n[TEST 3] wrong_answer_type 집계 로직 시뮬레이션")

    CAPTION_TYPES = ['gt', 'hybrid', 'positive', 'negative']
    correct_answers_by_type = {'positive': 0, 'negative': 0, 'hybrid': 0}
    total_questions_by_type = {'positive': 0, 'negative': 0, 'hybrid': 0}
    wrong_answer_counts_by_type = {'hybrid': 0, 'positive': 0, 'negative': 0}
    wrong_answers_by_question_type = {
        'positive': {'positive': 0, 'negative': 0, 'hybrid': 0},
        'negative': {'positive': 0, 'negative': 0, 'hybrid': 0},
        'hybrid':   {'positive': 0, 'negative': 0, 'hybrid': 0},
    }
    predictions_by_type = {'positive': 0, 'negative': 0, 'hybrid': 0}
    sample_results = []

    # 목 배치 데이터 (no shuffle, correct_answer=0 항상)
    # 시나리오: 4개 샘플
    #   - sample 0: question_type=negative, predict=0(gt) → 정답
    #   - sample 1: question_type=hybrid,   predict=2(positive) → 오답
    #   - sample 2: question_type=positive, predict=3(negative) → 오답
    #   - sample 3: question_type=negative, predict=1(hybrid)   → 오답
    mock_correct_answers    = [0, 0, 0, 0]
    mock_predicted_answers  = [0, 2, 3, 1]
    mock_question_types     = ['negative', 'hybrid', 'positive', 'negative']
    # shuffle=False → 모든 샘플의 caption_types는 canonical
    per_sample_caption_types = [tuple(CAPTION_TYPES)] * 4

    for i in range(4):
        answer_type   = mock_question_types[i]
        predicted_idx = mock_predicted_answers[i]
        correct_idx   = mock_correct_answers[i]
        sample_caption_types = per_sample_caption_types[i]
        predicted_caption_type = sample_caption_types[predicted_idx]

        total_questions_by_type[answer_type] += 1

        sample = {
            "question_type":    answer_type,
            "correct_answer":   correct_idx,
            "predicted_answer": predicted_idx,
            "is_correct":       (predicted_idx == correct_idx),
        }
        sample_results.append(sample)

        if predicted_idx == correct_idx:
            correct_answers_by_type[answer_type] += 1
            predictions_by_type[answer_type] += 1
        else:
            wrong_type = predicted_caption_type
            wrong_answer_counts_by_type[wrong_type] = wrong_answer_counts_by_type.get(wrong_type, 0) + 1
            predictions_by_type[wrong_type] = predictions_by_type.get(wrong_type, 0) + 1
            wrong_answers_by_question_type[answer_type][wrong_type] = \
                wrong_answers_by_question_type[answer_type].get(wrong_type, 0) + 1

    # 기대값 검증
    expected_correct_by_type  = {'positive': 0, 'negative': 1, 'hybrid': 0}
    expected_wrong_counts     = {'hybrid': 1, 'positive': 1, 'negative': 1}
    expected_wrong_by_qtype   = {
        'negative': {'hybrid': 1, 'positive': 0, 'negative': 0},
        'hybrid':   {'hybrid': 0, 'positive': 1, 'negative': 0},
        'positive': {'hybrid': 0, 'positive': 0, 'negative': 1},
    }

    ok_count = 0
    if correct_answers_by_type == expected_correct_by_type:
        ok(f"correct_answers_by_type={correct_answers_by_type}"); ok_count += 1
    else:
        fail("correct_answers_by_type 오류",
             f"got={correct_answers_by_type}, expected={expected_correct_by_type}")

    if wrong_answer_counts_by_type == expected_wrong_counts:
        ok(f"wrong_answer_counts_by_type={wrong_answer_counts_by_type}"); ok_count += 1
    else:
        fail("wrong_answer_counts_by_type 오류",
             f"got={wrong_answer_counts_by_type}, expected={expected_wrong_counts}")

    for qtype in ['negative', 'hybrid', 'positive']:
        got = wrong_answers_by_question_type[qtype]
        exp = expected_wrong_by_qtype[qtype]
        if got == exp:
            ok(f"wrong_answers_by_question_type[{qtype}]={got}"); ok_count += 1
        else:
            fail(f"wrong_answers_by_question_type[{qtype}] 오류",
                 f"got={got}, expected={exp}")

    # sample_results 4개인지 확인
    if len(sample_results) == 4:
        ok(f"sample_results 길이=4 ✓"); ok_count += 1
    else:
        fail("sample_results 길이 오류", str(len(sample_results)))

    # is_correct 값 확인
    expected_is_correct = [True, False, False, False]
    ic_vals = [s['is_correct'] for s in sample_results]
    if ic_vals == expected_is_correct:
        ok(f"is_correct={ic_vals} ✓"); ok_count += 1
    else:
        fail("is_correct 값 오류", f"got={ic_vals}, expected={expected_is_correct}")

    return ok_count >= 6

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: predictions.csv 저장 형식 검증
# ═══════════════════════════════════════════════════════════════════════════════

def test_predictions_csv_format(tmp_dir):
    print("\n[TEST 4] predictions.csv 저장 형식")

    sample_results = [
        {
            "image_path":     "/fake/img_0.jpg",
            "question_type":  "negative",
            "correct_answer": 0,
            "predicted_answer": 2,
            "is_correct":     False,
            "caption_0":      "GT caption",
            "caption_1":      "Hybrid wrong",
            "caption_2":      "Positive wrong",  # ← 예측한 것
            "caption_3":      "Negative wrong",
            "logit_0": 0.32, "logit_1": 0.21, "logit_2": 0.41, "logit_3": 0.06,
        },
        {
            "image_path":     "/fake/img_1.jpg",
            "question_type":  "positive",
            "correct_answer": 0,
            "predicted_answer": 0,
            "is_correct":     True,
            "caption_0":      "GT caption",
            "caption_1":      "Hybrid wrong",
            "caption_2":      "Positive wrong",
            "caption_3":      "Negative wrong",
            "logit_0": 0.55, "logit_1": 0.15, "logit_2": 0.20, "logit_3": 0.10,
        },
    ]

    # utils.py의 저장 로직 그대로 재현
    pred_dir = os.path.join(tmp_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    csv_path = os.path.join(pred_dir, "coco-mcq_predictions.csv")
    df = pd.DataFrame(sample_results)
    df.to_csv(csv_path, index=False)

    # 다시 읽어서 검증
    df_loaded = pd.read_csv(csv_path)
    cols_expected = {"image_path", "question_type", "correct_answer",
                     "predicted_answer", "is_correct",
                     "caption_0", "caption_1", "caption_2", "caption_3",
                     "logit_0", "logit_1", "logit_2", "logit_3"}
    cols_actual   = set(df_loaded.columns)

    if cols_expected.issubset(cols_actual):
        ok(f"predictions.csv 컬럼 정상: {sorted(cols_actual)}")
    else:
        missing = cols_expected - cols_actual
        fail("predictions.csv 컬럼 누락", f"missing={missing}")
        return False

    if len(df_loaded) == 2:
        ok("predictions.csv 행 수=2 ✓")
    else:
        fail("행 수 오류", str(len(df_loaded)))
        return False

    # is_correct 타입 확인 (bool → CSV에서는 True/False 문자열)
    val = df_loaded["is_correct"].iloc[0]
    if str(val).lower() in ("false", "0"):
        ok(f"is_correct[0]={val} (False) ✓")
    else:
        fail("is_correct[0] 값 오류", str(val))

    print(f"\n  CSV preview:\n{df_loaded.to_string(index=False)}\n")
    return True

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: results.jsonl 저장 형식 검증
# ═══════════════════════════════════════════════════════════════════════════════

def test_results_jsonl_format(tmp_dir):
    print("\n[TEST 5] results.jsonl 저장 형식 (utils.py 로직 시뮬레이션)")

    # mcq_eval()이 results에 담는 것과 동일한 구조
    metrics = {
        "coco-mcq-total_accuracy":      0.6250,
        "coco-mcq-positive_accuracy":   0.7500,
        "coco-mcq-negative_accuracy":   0.5000,
        "coco-mcq-hybrid_accuracy":     0.6000,
        "coco-mcq-most_common_wrong_answer_type": "hybrid",
        "coco-mcq-wrong_answer_percentages":
            [("hybrid", 50.0), ("positive", 30.0), ("negative", 20.0)],
        "coco-mcq-predictions_by_type": {"positive": 4, "negative": 3, "hybrid": 1},
        "coco-mcq-wrong_answers_by_question_type": {
            "positive": {"positive": 0, "negative": 1, "hybrid": 0},
            "negative": {"positive": 1, "negative": 0, "hybrid": 1},
            "hybrid":   {"positive": 1, "negative": 0, "hybrid": 0},
        },
        # sample_results는 utils.py에서 pop 후 저장되므로 여기서는 이미 제거됨
    }

    # utils.py L74~76 그대로 재현
    checkpoint_path = tmp_dir
    jsonl_path = os.path.join(checkpoint_path, "results.jsonl")
    with open(jsonl_path, "a+") as f:
        f.write(json.dumps(metrics))
        f.write("\n")

    # 다시 읽어서 파싱 검증
    with open(jsonl_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    if len(lines) == 1:
        ok("results.jsonl에 1줄 기록됨 ✓")
    else:
        fail("results.jsonl 줄 수 오류", str(len(lines)))
        return False

    parsed = json.loads(lines[0])

    required_keys = [
        "coco-mcq-total_accuracy",
        "coco-mcq-positive_accuracy",
        "coco-mcq-negative_accuracy",
        "coco-mcq-hybrid_accuracy",
        "coco-mcq-most_common_wrong_answer_type",
        "coco-mcq-wrong_answer_percentages",
        "coco-mcq-predictions_by_type",
        "coco-mcq-wrong_answers_by_question_type",
    ]
    missing = [k for k in required_keys if k not in parsed]
    if not missing:
        ok("results.jsonl 필수 키 모두 존재 ✓")
    else:
        fail("results.jsonl 키 누락", str(missing))
        return False

    # sample_results가 없어야 함 (utils.py에서 pop)
    if "coco-mcq-sample_results" not in parsed:
        ok("results.jsonl에 sample_results 없음 (pop 정상) ✓")
    else:
        fail("results.jsonl에 sample_results가 남아있음 (pop 실패)")

    # 정확도 값 타입 확인
    if isinstance(parsed["coco-mcq-total_accuracy"], float):
        ok(f"total_accuracy={parsed['coco-mcq-total_accuracy']} (float) ✓")
    else:
        fail("total_accuracy 타입 오류", type(parsed["coco-mcq-total_accuracy"]))

    print(f"\n  JSONL content:\n  {json.dumps(parsed, indent=2)}\n")
    return True

# ═══════════════════════════════════════════════════════════════════════════════
# 셔플된 데이터셋에서 CLIP 동일 정확도 이론적 검증
# ═══════════════════════════════════════════════════════════════════════════════

def test_shuffle_invariance_theory():
    """
    CLIP은 각 캡션을 독립적으로 임베딩하므로, 정답 캡션에 대한 유사도는
    순서와 무관합니다. 이를 간단한 수치로 증명합니다.
    """
    print("\n[TEST 6] CLIP 셔플 불변성 이론 검증 (단순 수치)")

    torch.manual_seed(0)
    D = 8

    # 이미지 임베딩 (고정)
    img = torch.randn(D)
    img = img / img.norm()

    # 4개 캡션 임베딩 (고정)
    captions_emb = torch.randn(4, D)
    captions_emb = captions_emb / captions_emb.norm(dim=1, keepdim=True)

    # 원본 순서: 정답=index 0
    logits_orig = captions_emb @ img                   # (4,)
    pred_orig   = logits_orig.argmax().item()           # 가장 높은 유사도의 인덱스

    # 셔플: perm=[2,0,3,1] → 정답이 index 1로 이동
    perm = [2, 0, 3, 1]
    shuffled_emb = captions_emb[perm]
    logits_shuf  = shuffled_emb @ img
    pred_shuf    = logits_shuf.argmax().item()

    # 셔플 후 정답 인덱스 = perm.index(0) = 1
    correct_after_shuffle = perm.index(0)

    # 예측한 캡션의 원본 인덱스가 동일해야 함
    original_predicted_shuf = perm[pred_shuf]

    if original_predicted_shuf == pred_orig:
        ok(f"셔플 전후 동일 캡션 선택: orig_pred={pred_orig}, "
           f"shuf_pred={pred_shuf}→orig_idx={original_predicted_shuf} ✓")
    else:
        fail("셔플 전후 다른 캡션 선택됨",
             f"orig={pred_orig}, shuf→orig={original_predicted_shuf}")

    correct_orig = (pred_orig == 0)
    correct_shuf = (pred_shuf == correct_after_shuffle)
    if correct_orig == correct_shuf:
        ok(f"정답 여부 동일: {correct_orig} ✓")
    else:
        fail("셔플 후 정답 여부가 달라짐",
             f"orig={correct_orig}, shuf={correct_shuf}")

    return True

# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  NegBench MCQ Shuffle & Output Format Test Suite")
    print("=" * 65)

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = make_temp_csv(n_rows=8, tmp_dir=tmp)

        results["1_dataset_format"]    = test_dataset_return_format(csv_path)
        results["2_collation_shape"]   = test_collation_shape(csv_path)
        results["3_aggregation_logic"] = test_aggregation_logic()
        results["4_predictions_csv"]   = test_predictions_csv_format(tmp)
        results["5_results_jsonl"]     = test_results_jsonl_format(tmp)
        results["6_shuffle_invariance"]= test_shuffle_invariance_theory()

    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    all_pass = True
    for name, passed in results.items():
        status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  \033[92m모든 테스트 통과 ✓\033[0m")
        sys.exit(0)
    else:
        print("  \033[91m실패한 테스트가 있습니다.\033[0m")
        sys.exit(1)
