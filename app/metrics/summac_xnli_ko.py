#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
summac_xnli_ko.py
- 한국어 텍스트를 다국어 NLI(XNLI: xlm-roberta-large-xnli)로 직접 평가 (SummaC-Conv 유사)
- granularity 옵션으로 평가 단위를 선택:
    • sentence  : 문장 단위(기본, 정밀)
    • paragraph : 빈 줄 기준 단락 단위
    • whole     : 통으로 한 쌍 비교(전역 한 방, ZS에 가까움)
- 최종 점수: mean(max entail per cand unit) - alpha * mean(max contrad per cand unit)
"""

import os
import argparse
from typing import List, Dict, Any, Union
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU 고정

# ------------------ 문장/단락 분할 ------------------
def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # 1) kss가 있으면 한국어 품질 우수
    try:
        import kss
        sents = [s.strip() for s in kss.split_sentences(text) if s.strip()]
        if sents:
            return sents
    except Exception:
        pass
    # 2) 백업: look-behind 없이 간단 분해
    import re
    pattern = r'(?:다\.)|[\.?!…]'               # '다.' 또는 일반 종결부호
    text_marked = re.sub(pattern, lambda m: m.group(0) + '\n', text)
    sents = [s.strip() for s in text_marked.splitlines() if s.strip()]
    return sents

def split_by_paragraph(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # 빈 줄(두 줄 이상의 개행) 기준으로 단락 분할
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return parts

def prepare_units(text: str, granularity: str) -> List[str]:
    if granularity == "whole":
        return [text.strip()] if text and text.strip() else []
    elif granularity == "paragraph":
        return split_by_paragraph(text)
    else:  # "sentence"
        return split_sentences(text)

# ------------------ SummaC-Conv 유사 집계 ------------------
def summac_like_score(ref_text: str, cand_text: str, pipe: Any, granularity: str = "sentence", batch_size: int = 16, alpha: float = 0.0) -> Dict[str, Any]:
    """
    최종 점수 = mean(max entail per cand_unit) - alpha * mean(max contrad per cand_unit)
    """
    import numpy as np

    ref_units  = prepare_units(ref_text, granularity)
    cand_units = prepare_units(cand_text, granularity)

    if not ref_units or not cand_units:
        return {
            "score": 0.0, "entail_mean": 0.0, "contrad_mean": 0.0,
            "per_cand_max_entail": [], "per_cand_max_contrad": [],
            "n_ref": len(ref_units), "n_cand": len(cand_units),
            "granularity": granularity,
        }

    per_cand_max_entail = []
    per_cand_max_contrad = []

    # cand_unit 하나씩 순회하며 ref_units 전체와 NLI
    for i in range(0, len(cand_units), batch_size):
        batch_cands = cand_units[i:i+batch_size]
        for cand in batch_cands:
            entail_list, contrad_list = [], []
            for ref in ref_units:
                scores = nli_pair_scores(pipe, ref, cand)
                entail_list.append(extract_prob(scores, "entail"))
                contrad_list.append(extract_prob(scores, "contrad"))
            per_cand_max_entail.append(max(entail_list) if entail_list else 0.0)
            per_cand_max_contrad.append(max(contrad_list) if contrad_list else 0.0)

    entail_mean  = float(np.mean(per_cand_max_entail)) if per_cand_max_entail else 0.0
    contrad_mean = float(np.mean(per_cand_max_contrad)) if per_cand_max_contrad else 0.0
    final_score  = entail_mean - alpha * contrad_mean

    return {
        "score": final_score,
        "entail_mean": entail_mean,
        "contrad_mean": contrad_mean,
        "per_cand_max_entail": per_cand_max_entail,
        "per_cand_max_contrad": per_cand_max_contrad,
        "n_ref": len(ref_units),
        "n_cand": len(cand_units),
        "granularity": granularity,
    }

# ------------------ NLI 파이프라인 ------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def load_nli(model_name: str = "joeddav/xlm-roberta-large-xnli"):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    # top_k=None: 모든 라벨 점수 반환 (return_all_scores deprecated 대체)
    pipe = pipeline(
        task="text-classification",
        model=mdl,
        tokenizer=tok,
        device=-1,           # CPU
        truncation=True,
        padding=True,
        max_length=512,
        top_k=None,
    )
    return pipe

def nli_pair_scores(pipe: Any, premise: str, hypothesis: str) -> List[Dict[str, Union[str, float]]]:
    return pipe({"text": premise, "text_pair": hypothesis})

def extract_prob(label_scores: List[Dict[str, Union[str, float]]], target_prefix: str) -> float:
    target = target_prefix.lower()  # 'entail' / 'contrad' / 'neutral'
    for item in label_scores:
        if item["label"].lower().startswith(target):
            return float(item["score"])
    return 0.0
