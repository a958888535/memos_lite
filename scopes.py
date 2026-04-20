from __future__ import annotations

import re
from typing import Iterable, List


_HEALTH_TERMS = (
    "health",
    "medical",
    "blood pressure",
    "sleep",
    "diagnosis",
    "健康",
    "医疗",
    "血压",
    "睡眠",
    "诊断",
    "用药",
    "药物",
)
_QUANT_TERMS = (
    "quant",
    "finance",
    "accounting",
    "event study",
    "regression",
    "did",
    "量化",
    "金融",
    "财务",
    "会计",
    "事件研究",
    "回归",
    "双重差分",
)
_PAPER_TERMS = (
    "paper",
    "literature review",
    "manuscript",
    "theory",
    "journal",
    "论文",
    "文献综述",
    "手稿",
    "理论",
    "期刊",
)
_RESEARCH_TERMS = (
    "research",
    "hypothesis",
    "experiment",
    "identification",
    "dataset",
    "研究",
    "假设",
    "实验",
    "识别",
    "数据集",
)
_NEWS_TERMS = ("news", "headline", "breaking", "today", "新闻", "头条", "快讯", "今日")


def _has_any_term(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def infer_scope(
    *,
    content: str = "",
    scope: str | None = None,
    domain: str | None = None,
    metadata: dict | None = None,
    config: dict | None = None,
) -> str:
    if scope:
        return scope
    metadata = metadata or {}
    defaults = config or {}
    project_id = metadata.get("project_id") or metadata.get("project")
    profile_name = metadata.get("profile_name") or metadata.get("profile")
    lowered = str(content or "").lower()
    domain_label = str(domain or "").lower()
    if profile_name:
        return f"profile:{profile_name}"
    if project_id:
        return f"project:{project_id}"
    if domain_label == "health" or _has_any_term(lowered, _HEALTH_TERMS):
        return defaults.get("health_default", "domain:health")
    if domain_label == "quant" or _has_any_term(lowered, _QUANT_TERMS):
        return defaults.get("quant_default", "domain:quant")
    if domain_label == "paper" or _has_any_term(lowered, _PAPER_TERMS):
        return defaults.get("paper_default", "domain:paper")
    if domain_label == "research" or _has_any_term(lowered, _RESEARCH_TERMS):
        return "domain:research"
    if domain_label == "news" or _has_any_term(lowered, _NEWS_TERMS):
        return "domain:news"
    return defaults.get("default", "global")


def _infer_query_scopes(query: str) -> set[str]:
    lowered = str(query or "").lower()
    scopes = {"global"}
    project_match = re.search(r"\bproject[:\s]+([a-z0-9_-]+)\b", lowered)
    profile_match = re.search(r"\bprofile[:\s]+([a-z0-9_-]+)\b", lowered)
    if project_match:
        scopes.add(f"project:{project_match.group(1)}")
    if profile_match:
        scopes.add(f"profile:{profile_match.group(1)}")
    if _has_any_term(lowered, _HEALTH_TERMS):
        scopes.add("domain:health")
    if _has_any_term(lowered, _QUANT_TERMS):
        scopes.add("domain:quant")
    if _has_any_term(lowered, _PAPER_TERMS):
        scopes.add("domain:paper")
    if _has_any_term(lowered, _RESEARCH_TERMS):
        scopes.add("domain:research")
    if _has_any_term(lowered, _NEWS_TERMS):
        scopes.add("domain:news")
    return scopes


def filter_memories_for_query(
    memories: Iterable[dict],
    *,
    query: str,
    explicit_scope: str | None = None,
) -> List[dict]:
    allowed = {"global"}
    if explicit_scope:
        allowed.add(explicit_scope)
    else:
        allowed = _infer_query_scopes(query)
    output = []
    for item in memories:
        scope = item.get("scope", "global")
        if scope in allowed or scope == "global":
            output.append(item)
    return output
