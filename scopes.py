from __future__ import annotations

import re
from typing import Iterable, List


_HEALTH_TERMS = (
    "blood pressure",
    "diagnosis",
    "prescription",
    "medication",
    "医院",
    "血压",
    "睡眠",
    "诊断",
    "用药",
    "药物",
    "治疗",
    "症状",
    "体检",
    "患者",
)
_QUANT_TERMS = (
    "quantitative trading",
    "quant trading",
    "backtest",
    "alpha factor",
    "portfolio",
    "sharpe ratio",
    "drawdown",
    "量化",
    "回测",
    "因子",
    "选股",
    "夏普",
    "动量",
    "对冲",
    "期货",
    "收益率",
    "最大回撤",
    "多因子",
)
_PAPER_TERMS = (
    "literature review",
    "manuscript",
    "journal",
    "proceedings",
    "arxiv",
    "doi:",
    "citation",
    "文献综述",
    "手稿",
    "期刊",
    "引用",
    "被引",
    "综述",
)
_RESEARCH_TERMS = (
    "hypothesis",
    "experiment",
    "identification strategy",
    "causal inference",
    "difference-in-differences",
    "instrumental variable",
    "regression discontinuity",
    "RCT",
    "panel data",
    "fixed effects",
    "假设",
    "实验",
    "识别策略",
    "因果推断",
    "双重差分",
    "工具变量",
    "断点回归",
    "面板数据",
    "固定效应",
)
_NEWS_TERMS = ("breaking news", "headline news", "新闻", "头条", "快讯", "热点新闻")
_TECH_TERMS = (
    "python",
    "javascript",
    "asyncio",
    "docker",
    "kubernetes",
    "linux",
    "api",
    "database",
    "redis",
    "nginx",
    "webhook",
    "git",
    "编译",
    "部署",
    "调试",
    "服务器",
    "数据库",
    "容器",
    "接口",
    "缓存",
    "异步",
    "并发",
    "分词",
    "去重",
    "tokeniz",
)


def _has_any_term(text: str, terms: tuple[str, ...]) -> bool:
    """Check whether *text* contains any of the given *terms*.

    English terms are matched with word boundaries (``\\b``) so that
    short words like "did" don't false-positive on "didn't" or "candidate".
    CJK terms use plain substring matching since Chinese/Japanese/Korean
    don't have whitespace-delimited word boundaries.
    """
    for term in terms:
        if any("\u4e00" <= ch <= "\u9fff" for ch in term):
            # CJK term — substring match
            if term in text:
                return True
        else:
            # English / Latin term — word-boundary match
            if re.search(rf"\b{re.escape(term)}\b", text):
                return True
    return False


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
    if domain_label == "tech" or _has_any_term(lowered, _TECH_TERMS):
        return defaults.get("tech_default", "domain:tech")
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
    if _has_any_term(lowered, _TECH_TERMS):
        scopes.add("domain:tech")
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
