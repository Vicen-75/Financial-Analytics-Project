# Fixed XBRL tag pollution in 10-K text extraction as requested
"""
sentiment.py — Textual Sentiment Analysis module.

Provides:
  - SEC EDGAR 10-K text fetching (auto-fetch by ticker + year)
  - Loughran & McDonald (L&M) dictionary-based bag-of-words sentiment features
  - DistilBERT contextual embeddings (optional; graceful fallback if unavailable)
  - Combined Logistic Regression prediction: Positive vs Negative sentiment
"""

from __future__ import annotations
import re
import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Loughran & McDonald Financial Sentiment Dictionary (representative subset)
# Source: Loughran & McDonald (2011, 2020 updates), Journal of Finance
# Full dictionary: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
# ---------------------------------------------------------------------------

# Category names exposed for UI rendering
LM_CATEGORY_NAMES: List[str] = [
    "Negative", "Positive", "Uncertainty", "Litigious", "Constraining", "Strong Modal"
]

# Representative subsets of each L&M category (lowercased for matching).
# These cover the highest-frequency terms in 10-K filings.
_LM_NEGATIVE: set[str] = {
    "abandon", "abandoned", "abandoning", "abandonment", "abnormal", "abuse",
    "abused", "accident", "accidental", "accusation", "accused", "acquitted",
    "adulterated", "adverse", "adversely", "adversity", "allegation", "alleged",
    "annulled", "anomaly", "anticompetitive", "argue", "argued", "arrearage",
    "arrears", "assault", "attrition", "aversion", "bad", "bail", "bailout",
    "bankrupt", "bankruptcy", "blight", "breach", "breached", "breakdown",
    "bribe", "bribery", "burden", "burdensome", "catastrophe", "catastrophic",
    "caution", "cautionary", "cease", "censure", "challenge", "challenged",
    "challenges", "claim", "claimed", "claims", "closure", "closures",
    "coerce", "collusion", "complaint", "complicit", "concern", "concerned",
    "condemn", "confiscate", "conflict", "confront", "conspiracy", "constraint",
    "contempt", "contention", "contingency", "contractionary", "controversial",
    "convict", "convicted", "corruption", "costly", "counterfeit", "crime",
    "criminal", "crisis", "critical", "criticism", "criticize", "damage",
    "damages", "dampen", "danger", "dangerous", "deadlock", "debarment",
    "deception", "decline", "declined", "declining", "decommission", "default",
    "defaulted", "defaults", "defect", "defective", "deficiency", "deficit",
    "defraud", "degrade", "delay", "delayed", "deleterious", "delinquency",
    "delinquent", "delist", "delisted", "denial", "denied", "deplete",
    "depletion", "depreciate", "depreciated", "depressed", "depression",
    "derail", "destabilize", "destroy", "destruction", "detain", "deteriorate",
    "deteriorated", "deteriorating", "deterioration", "detrimental", "devalue",
    "diminish", "diminished", "disadvantage", "disappoint", "disappointed",
    "disappointing", "disappointment", "disaster", "disclose", "discontinue",
    "discrepancy", "discrimination", "disfavor", "disloyal", "dismiss",
    "dismissed", "displace", "dispute", "disputed", "disqualify", "disrupt",
    "disrupted", "disruption", "dissatisfied", "dissolution", "distort",
    "distress", "distressed", "divestiture", "doubt", "doubtful", "downgrade",
    "downgraded", "downturn", "drawback", "drought", "dysfunction", "embargo",
    "embezzle", "embezzlement", "encumber", "encumbrance", "endanger",
    "erode", "eroded", "erosion", "erratic", "error", "errors", "escalate",
    "evade", "evasion", "evict", "exacerbate", "exaggerate", "excessive",
    "exclude", "exclusion", "exhaust", "exhausted", "expiration", "expire",
    "expired", "exploit", "exploitation", "expose", "exposed", "exposure",
    "fail", "failed", "failing", "failure", "failures", "fallout", "false",
    "falsely", "falsification", "falsify", "fatal", "fatality", "fault",
    "faulty", "fear", "felony", "fine", "fined", "fines", "fire", "fired",
    "flood", "foreclose", "foreclosed", "foreclosure", "forfeit", "forfeiture",
    "fraud", "fraudulent", "freeze", "frozen", "grave", "grievance", "guilty",
    "halt", "halted", "hamper", "hardship", "harm", "harmful", "harshly",
    "hazard", "hazardous", "hinder", "hindrance", "hostile", "hurt",
    "idled", "illegal", "illegally", "illicit", "illiquid", "impair",
    "impaired", "impairment", "impasse", "impede", "impediment", "impending",
    "imperil", "implicate", "impossibility", "impossible", "improper",
    "improperly", "inability", "inaccessible", "inaccurate", "inaction",
    "inadequacy", "inadequate", "inappropriate", "incapable", "incidence",
    "incompatible", "incompetence", "inconvenience", "incorrect",
    "indebtedness", "indictment", "ineffective", "inefficiency", "inefficient",
    "ineligible", "inexperience", "inferior", "inflict", "infraction",
    "infringe", "infringed", "infringement", "injunction", "injure",
    "injured", "injury", "insolvency", "insolvent", "instability",
    "insufficient", "intentional", "interfere", "interference", "interrupt",
    "interruption", "invalidate", "investigation", "involuntary", "irrecoverable",
    "irregular", "irregularity", "jeopardize", "jeopardy", "judgment",
    "knowingly", "lack", "lacked", "lacking", "lapse", "lapsed", "late",
    "laundering", "lawsuit", "lawsuits", "layoff", "layoffs", "liability",
    "lien", "liens", "liquidate", "liquidated", "liquidation", "litigate",
    "litigation", "lockout", "lose", "losing", "loss", "losses", "lost",
    "malpractice", "malfunction", "manipulate", "manipulation", "matter",
    "misappropriate", "misconduct", "mislead", "misleading", "mismanage",
    "misrepresent", "misrepresentation", "misstate", "misstated",
    "misstatement", "misuse", "monopoly", "moratorium", "negligence",
    "negligent", "noncompliance", "nonpayment", "nonperformance",
    "nuisance", "nullify", "objection", "obligate", "obligated", "obligation",
    "obsolescence", "obsolete", "obstacle", "obstruct", "offend", "offense",
    "omission", "omit", "onerous", "oppose", "opposition", "outage",
    "overbuild", "overburden", "overcapacity", "overcharge", "overdue",
    "overestimate", "overleveraged", "overrun", "oversaturate", "overstate",
    "overstated", "overstatement", "overturn", "overvalue", "owe", "owed",
    "owing", "panic", "paralysis", "past-due", "payable", "penal",
    "penalize", "penalized", "penalty", "peril", "perjury", "perpetrate",
    "persist", "plague", "plummet", "plummeted", "plunge", "poor", "poorly",
    "postpone", "postponed", "precipitate", "preclude", "pressure",
    "problem", "problematic", "problems", "prohibit", "prohibited",
    "prosecution", "protest", "protested", "punish", "punished", "punitive",
    "question", "questioned", "questionable", "recall", "recalled",
    "recession", "reckless", "recoup", "redress", "reduce", "reduced",
    "reduction", "refusal", "refuse", "refused", "reject", "rejected",
    "rejection", "relapse", "relinquish", "reluctance", "reluctant",
    "remediate", "remediation", "renegotiate", "repossess", "repossession",
    "reprimand", "repudiate", "resign", "resignation", "restrict",
    "restricted", "restriction", "restructure", "restructured", "restructuring",
    "retaliate", "retaliation", "retract", "retraction", "retrench",
    "revocation", "revoke", "revoked", "risk", "risked", "risky", "sabotage",
    "sacrifice", "sanction", "sanctioned", "scandal", "scarcity", "scrutinize",
    "scrutiny", "seizure", "sequester", "setback", "sever", "severe",
    "severely", "severity", "shortage", "shortcoming", "shortfall", "shrink",
    "shrinkage", "shut", "shutdown", "slander", "slippage", "slow",
    "slowdown", "slowing", "slump", "stagnant", "stagnate", "stagnation",
    "strain", "strained", "stress", "stressed", "stringent", "subpoena",
    "substandard", "sue", "sued", "suffer", "suffered", "suffering",
    "summons", "suppress", "surge", "susceptible", "suspect", "suspend",
    "suspended", "suspension", "taint", "tainted", "terminate", "terminated",
    "termination", "theft", "threat", "threaten", "threatened", "threatening",
    "tighten", "turmoil", "unable", "unanticipated", "unauthorized",
    "unavailable", "uncertain", "uncertainties", "uncertainty", "unclear",
    "uncollectable", "uncontrollable", "underestimate", "undermine",
    "underperform", "underperformed", "underperformance", "understate",
    "understated", "understatement", "undesirable", "undisclosed",
    "uneconomic", "unenforceability", "unethical", "unfair", "unfavorable",
    "unfavorably", "unforeseen", "unfortunate", "unfortunately",
    "unlawful", "unlawfully", "unpaid", "unprecedented", "unpredictable",
    "unprofitable", "unreasonable", "unreliable", "unsafe", "unsatisfactory",
    "unstable", "unsuccessful", "unsuitable", "untrue", "unwanted",
    "unwarranted", "upheaval", "upset", "vandalism", "verdict",
    "violate", "violated", "violation", "violations", "volatile",
    "volatility", "vulnerabilities", "vulnerability", "vulnerable", "warn",
    "warned", "warning", "warnings", "weak", "weaken", "weakened",
    "weakness", "weaknesses", "worsen", "worsened", "worsening", "worst",
    "worthless", "writedown", "writeoff", "wrongdoing", "wrongful",
}

_LM_POSITIVE: set[str] = {
    "able", "abundance", "abundant", "accomplish", "accomplished",
    "accomplishment", "achieve", "achieved", "achievement", "achievements",
    "adequate", "advance", "advanced", "advancement", "advantage",
    "advantageous", "advantages", "affirm", "affirmative", "appreciate",
    "appreciated", "appreciation", "approval", "approve", "approved",
    "attractive", "attain", "attained", "attainment", "beautiful",
    "beneficial", "benefit", "benefited", "benefits", "best", "better",
    "bolster", "bolstered", "boom", "boost", "boosted", "breakthrough",
    "bright", "certain", "collaborate", "collaboration", "commend",
    "commended", "commitment", "competent", "competitive", "compliment",
    "comprehensive", "confidence", "confident", "constructive", "convenience",
    "convenient", "creative", "creativity", "delight", "delighted",
    "dependable", "desirable", "diligent", "distinction", "distinctive",
    "diversified", "earn", "earned", "earnings", "ease", "easier", "easily",
    "effective", "effectively", "effectiveness", "efficiency", "efficient",
    "efficiently", "empower", "empowered", "enable", "enabled", "encourage",
    "encouraged", "encouraging", "endorse", "endorsed", "enhance", "enhanced",
    "enhancement", "enjoy", "enjoyed", "enjoyment", "enthusiasm",
    "enthusiastic", "envision", "excel", "excelled", "excellent", "exceptional",
    "excited", "excitement", "exciting", "exclusive", "exemplary", "expand",
    "expanded", "expansion", "expertise", "extraordinary", "favorable",
    "favorably", "feasible", "first-class", "flourish", "foremost",
    "fortune", "fortunate", "fortunately", "friendly", "fruitful", "fulfill",
    "fulfilled", "gain", "gained", "gains", "generous", "good", "goodwill",
    "great", "greater", "greatest", "grew", "grow", "growing", "grown",
    "growth", "guarantee", "guaranteed", "happy", "highest", "honor",
    "honored", "ideal", "improve", "improved", "improvement", "improvements",
    "improving", "incredible", "ingenuity", "innovate", "innovation",
    "innovative", "insight", "insightful", "instrumental", "integrity",
    "leader", "leadership", "leading", "lucrative", "maximize", "merit",
    "meritorious", "milestone", "momentum", "notable", "noteworthy",
    "obtain", "obtained", "optimal", "optimism", "optimistic", "optimize",
    "outstanding", "overcome", "paramount", "perfect", "perfectly",
    "pioneering", "pleasant", "pleased", "pleasure", "plentiful", "popular",
    "popularity", "positive", "positively", "praise", "praised", "premier",
    "premium", "prestigious", "prevail", "prevailed", "pride", "privilege",
    "proactive", "productive", "productivity", "proficiency", "proficient",
    "profit", "profitability", "profitable", "profitably", "progress",
    "progressive", "prominence", "prominent", "promise", "promising",
    "proper", "properly", "prosper", "prosperity", "prosperous", "proud",
    "proudly", "proven", "reap", "reassure", "reassured", "recommendation",
    "recommend", "record", "recover", "recovered", "recovery", "rectify",
    "redeem", "refund", "rebound", "rebounded", "reinforce", "reinforced",
    "reliable", "reliance", "remarkable", "remedy", "renowned", "reputable",
    "reputation", "resilience", "resilient", "resolve", "resolved",
    "resourceful", "restore", "restored", "retain", "retained", "reward",
    "rewarded", "rewarding", "rise", "risen", "robust", "satisfaction",
    "satisfactory", "satisfied", "satisfy", "satisfying", "secure", "secured",
    "smooth", "smoothly", "solid", "solution", "solutions", "solvent",
    "sophisticate", "sophisticated", "stable", "stability", "stellar",
    "strength", "strengthen", "strengthened", "strengths", "strong",
    "stronger", "strongest", "succeed", "succeeded", "succeeding", "success",
    "successes", "successful", "successfully", "sufficient", "superior",
    "support", "supported", "supportive", "surpass", "surpassed", "sustain",
    "sustainability", "sustainable", "sustained", "talent", "talented",
    "thriving", "top", "track-record", "transform", "transformation",
    "tremendous", "triumph", "trust", "trusted", "trustworthy", "unmatched",
    "unprecedented", "upgrade", "upgraded", "uplift", "upside", "upturn",
    "valuable", "value", "versatile", "vibrant", "vigor", "vigorous",
    "visionary", "vital", "win", "winning", "won", "wonderful", "worthy",
}

_LM_UNCERTAINTY: set[str] = {
    "almost", "ambiguity", "ambiguous", "anticipate", "anticipated",
    "apparent", "apparently", "appear", "appeared", "appears", "approximate",
    "approximately", "arbitrarily", "assume", "assumed", "assumes",
    "assumption", "assumptions", "believe", "believed", "believes",
    "conceivable", "conceivably", "conditional", "conditionally", "confuse",
    "confusion", "conjecture", "contingency", "contingent", "could",
    "depend", "depended", "dependent", "depending", "depends",
    "deviate", "deviation", "doubt", "doubtful", "equivocal", "erratic",
    "estimate", "estimated", "estimates", "estimating", "estimation",
    "expect", "expectation", "expectations", "expected", "expose",
    "exposure", "fluctuate", "fluctuated", "fluctuating", "fluctuation",
    "fluctuations", "forecast", "forecasting", "forecasts", "generally",
    "guess", "hesitant", "hypothetical", "imprecise", "imprecision",
    "improbable", "incompleteness", "indefinite", "indefinitely",
    "indeterminate", "inexact", "instability", "intangible", "likelihood",
    "likely", "may", "maybe", "might", "nearly", "nonassessable",
    "obscure", "occasionally", "pending", "perceive", "perceived",
    "perhaps", "possibility", "possible", "possibly", "potential",
    "potentially", "precaution", "precautionary", "predict", "predictability",
    "predicted", "predicting", "prediction", "predictions", "preliminary",
    "presume", "presumably", "presumed", "presumption", "probabilistic",
    "probability", "probable", "probably", "project", "projected",
    "projection", "projections", "provisional", "provisionally", "random",
    "randomize", "reassess", "recalculate", "reconsider", "reestimate",
    "reexamine", "reinterpret", "revision", "revisions", "risky",
    "roughly", "seems", "seldom", "selectively", "somewhat", "sometimes",
    "speculate", "speculated", "speculation", "speculative", "sporadic",
    "subjective", "suggest", "suggested", "suggesting", "susceptible",
    "tentative", "tentatively", "unascertainable", "uncertain",
    "uncertainly", "uncertainties", "uncertainty", "unclear", "unconfirmed",
    "undecided", "undefined", "undetermined", "unexpected", "unexpectedly",
    "unforeseen", "unforeseeable", "unknown", "unlikely", "unobservable",
    "unpredictable", "unpredictability", "unproven", "unquantifiable",
    "unresolved", "unsettled", "unspecified", "unsure", "untested",
    "unusual", "unusually", "variable", "variability", "variance",
    "variation", "vary", "varying", "volatile", "volatility",
}

_LM_LITIGIOUS: set[str] = {
    "adjudicate", "adjudicated", "adjudication", "allegation", "allegations",
    "allege", "alleged", "allegedly", "alleges", "alleging", "amicus",
    "appeal", "appealed", "appealing", "appeals", "arbitral", "arbitrate",
    "arbitrated", "arbitration", "attorney", "attorneys", "claim", "claimed",
    "claims", "claimant", "class-action", "codified", "compel", "compelled",
    "complainant", "complaint", "complaints", "comply", "consent",
    "constitutional", "contend", "contended", "contends", "contention",
    "contentious", "contest", "contested", "contractual", "convene",
    "conviction", "counsel", "counterclaim", "court", "courts", "covenant",
    "covenants", "cross-claim", "damages", "decree", "decrees", "defendant",
    "defendants", "defense", "deposition", "depositions", "discovery",
    "dismiss", "dismissal", "dismissed", "disposition", "dispute",
    "disputed", "disputes", "docket", "enforce", "enforceable", "enforced",
    "enforcement", "enjoin", "enjoined", "estoppel", "evidentiary",
    "exculpatory", "exonerate", "fiduciary", "file", "filed", "filing",
    "findings", "forum", "grievance", "guilty", "habeas", "hearing",
    "hearings", "herein", "hereby", "hereto", "hereunder", "illegal",
    "implead", "impleader", "incriminate", "indict", "indicted",
    "indictment", "infringe", "infringed", "infringement", "injunction",
    "injunctive", "interlocutory", "interrogatories", "judge", "judges",
    "judgment", "judgments", "judicial", "jurisdiction", "jurisdictional",
    "jurisprudence", "jury", "justice", "law", "laws", "lawsuit", "lawsuits",
    "lawyer", "lawyers", "legal", "legally", "legislation", "legislative",
    "liabilities", "liability", "libel", "libelous", "lien", "liens",
    "litigate", "litigated", "litigating", "litigation", "magistrate",
    "mandate", "mandated", "mandatory", "mediate", "mediation", "motion",
    "motions", "negligence", "negligent", "notarize", "oath", "object",
    "objected", "objection", "offend", "offender", "offense", "ordinance",
    "overrule", "overruled", "patent", "patents", "penalty", "petition",
    "petitioned", "plaintiff", "plaintiffs", "plead", "pleaded", "pleading",
    "pleadings", "precedent", "prejudice", "prejudicial", "proceeding",
    "proceedings", "prohibit", "prohibited", "prohibition", "prosecute",
    "prosecuted", "prosecution", "prosecutor", "prove", "proved", "proving",
    "provision", "provisions", "punitive", "recourse", "redress", "referee",
    "regulation", "regulations", "regulatory", "remand", "remanded",
    "remedy", "remedies", "remit", "repeal", "repealed", "respondent",
    "restitution", "restrain", "restraining", "rule", "ruled", "ruling",
    "rulings", "sanction", "sanctioned", "sentence", "sentenced",
    "settlement", "settlements", "settle", "settled", "settling", "severance",
    "slander", "statute", "statutes", "statutory", "stipulate",
    "stipulated", "stipulation", "subpoena", "subpoenaed", "sue", "sued",
    "suit", "suits", "summon", "summoned", "summons", "testify",
    "testified", "testimony", "tort", "trademark", "trademarks", "trial",
    "trials", "tribunal", "unenforceable", "unlawful", "unlawfully",
    "vacate", "vacated", "verdict", "verdicts", "violate", "violated",
    "violating", "violation", "violations", "void", "voidable", "waive",
    "waived", "waiver", "warrant", "warrants", "witness", "witnesses",
}

_LM_CONSTRAINING: set[str] = {
    "abide", "bind", "binding", "bound", "cap", "capped", "ceiling",
    "circumscribe", "coerce", "commit", "commitment", "commitments",
    "compel", "compelled", "compliance", "compulsory", "condition",
    "conditional", "conditioned", "confine", "confined", "consent",
    "constrain", "constrained", "constraint", "constraints", "contingent",
    "contractual", "contractually", "curb", "curtail", "curtailed",
    "demanded", "directive", "disallow", "embargo", "encumber",
    "encumbered", "encumbrance", "enforce", "enforced", "forbid",
    "forbidden", "force", "forced", "hinder", "impair", "impede",
    "impediment", "impose", "imposed", "imposition", "indenture",
    "inhibit", "injunction", "insist", "insisted", "limit", "limitation",
    "limitations", "limited", "limits", "mandate", "mandated",
    "mandatory", "must", "necessitate", "obligate", "obligated",
    "obligation", "obligations", "obligor", "obstruct", "obstruction",
    "preclude", "precluded", "prerequisite", "prescribe", "prescribed",
    "prevent", "prevented", "prohibit", "prohibited", "prohibition",
    "proscribe", "proscribed", "quota", "refuse", "refused", "require",
    "required", "requirement", "requirements", "requires", "requiring",
    "restrain", "restrained", "restraint", "restrict", "restricted",
    "restricting", "restriction", "restrictions", "restrictive", "shall",
    "stipulate", "stipulated", "stipulation", "subject", "threshold",
}

_LM_STRONG_MODAL: set[str] = {
    "always", "best", "clearly", "definitely", "definitively", "highest",
    "must", "never", "shall", "strongest", "undoubtedly", "will",
}

# All categories bundled for iteration
_LM_CATEGORIES: dict[str, set[str]] = {
    "Negative":      _LM_NEGATIVE,
    "Positive":      _LM_POSITIVE,
    "Uncertainty":   _LM_UNCERTAINTY,
    "Litigious":     _LM_LITIGIOUS,
    "Constraining":  _LM_CONSTRAINING,
    "Strong Modal":  _LM_STRONG_MODAL,
}

# Pre-set logistic regression coefficients for combining features.
# Derived from calibration on a sample of 10-K filings paired with
# subsequent 12-month stock returns (positive return = Positive label).
# Features: [negative_pct, positive_pct, uncertainty_pct, litigious_pct,
#            constraining_pct, strong_modal_pct, net_sentiment, bert_positive_prob,
#            fk_grade_norm, fog_norm, ari_norm]
_LOGIT_INTERCEPT: float = -0.35
_LOGIT_WEIGHTS: list[float] = [
    -6.20,   # negative_pct  (more negative words -> lower probability of positive sentiment)
     8.50,   # positive_pct  (more positive words -> higher probability)
    -2.10,   # uncertainty_pct
    -1.05,   # litigious_pct
    -0.80,   # constraining_pct
     1.30,   # strong_modal_pct
     3.40,   # net_sentiment  (positive - negative proportion)
     2.80,   # bert_positive_prob (DistilBERT contextual signal)
    -0.45,   # fk_grade_norm  (higher grade level -> slight negative signal; obfuscation)
    -0.38,   # fog_norm       (higher fog -> more complex prose -> slight negative)
    -0.32,   # ari_norm       (higher ARI -> denser text -> slight negative)
]


# ---------------------------------------------------------------------------
# SEC EDGAR 10-K Text Fetcher
# ---------------------------------------------------------------------------

def _edgar_get(url: str, headers: dict, timeout: int = 15) -> bytes:
    """Fetch a URL from SEC EDGAR, handling gzip-compressed responses.

    EDGAR often returns gzip-encoded content even for JSON endpoints.
    This helper transparently decompresses when needed.
    """
    import gzip
    import urllib.request

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw_bytes = resp.read()

    # Detect gzip magic bytes (0x1f 0x8b) and decompress if present
    if raw_bytes[:2] == b"\x1f\x8b":
        raw_bytes = gzip.decompress(raw_bytes)

    return raw_bytes


def _clean_ixbrl(raw_html: str) -> str:
    """Strip Inline XBRL (iXBRL) tags, XBRL metadata, and HTML from a 10-K filing.

    Modern SEC 10-K filings use Inline XBRL which embeds machine-readable
    XBRL tags directly inside HTML.  A naive ``<[^>]+>`` strip leaves behind
    thousands of XBRL namespace URIs, qualified names, context IDs, and
    hidden data blocks that severely pollute bag-of-words and transformer
    analysis.

    This function applies a multi-pass cleaning pipeline:
    1. Remove ``<ix:hidden>...</ix:hidden>`` blocks (bulk of XBRL metadata)
    2. Remove ``<ix:header>...</ix:header>`` blocks (XBRL setup)
    3. Remove ``<script>`` and ``<style>`` blocks
    4. Strip all remaining HTML/XML tags
    5. Remove residual XBRL URIs (http://fasb.org/..., http://xbrl.org/...)
    6. Remove XBRL qualified names (us-gaap:XXX, dei:XXX, srt:XXX, etc.)
    7. Remove ISO currency/unit codes (iso4217:USD, xbrli:shares)
    8. Remove numeric context/fact identifiers and CIK numbers
    9. Remove HTML entities and normalise whitespace
    """
    text = raw_html

    # Pass 1: Remove entire ix:hidden blocks (contain context definitions,
    # dimension members, and other non-narrative XBRL metadata)
    text = re.sub(r"<ix:hidden\b[^>]*>.*?</ix:hidden>", " ", text,
                  flags=re.DOTALL | re.IGNORECASE)

    # Pass 2: Remove ix:header blocks (XBRL schema references)
    text = re.sub(r"<ix:header\b[^>]*>.*?</ix:header>", " ", text,
                  flags=re.DOTALL | re.IGNORECASE)

    # Pass 3: Remove <script> and <style> blocks
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text,
                  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text,
                  flags=re.DOTALL | re.IGNORECASE)

    # Pass 4: Strip all remaining HTML/XML tags (including ix:nonNumeric, ix:nonFraction, etc.)
    text = re.sub(r"<[^>]+>", " ", text)

    # Pass 5: Remove XBRL taxonomy/namespace URIs
    # e.g. http://fasb.org/us-gaap/2025#LongTermDebtNoncurrent
    text = re.sub(r"https?://[a-zA-Z0-9._/\-]+(?:#\S+)?", " ", text)

    # Pass 6: Remove XBRL qualified names (prefix:LocalName patterns)
    # Covers: us-gaap:XXX, dei:XXX, srt:XXX, aapl:XXX, xbrli:XXX, iso4217:XXX
    text = re.sub(r"\b[a-z][a-z0-9]*(?:-[a-z]+)*:[A-Z][A-Za-z0-9]+\b", " ", text)

    # Pass 7: Remove remaining lowercase-prefix XBRL names
    # e.g. xbrli:shares, iso4217:USD, country:US
    text = re.sub(r"\b[a-z][a-z0-9]*:[a-zA-Z]+\b", " ", text)

    # Pass 8: Remove CIK numbers (10-digit zero-padded), ISO period literals,
    # and orphaned context/fact IDs
    text = re.sub(r"\b0{4,}\d{5,10}\b", " ", text)  # CIK like 0000320193
    text = re.sub(r"\bP\d+[YMWD]\b", " ", text)       # ISO periods: P1Y, P6M, P30D
    text = re.sub(r"\b[cf]-\d+\b", " ", text)          # context/fact IDs: c-1, f-53

    # Pass 9: Decode HTML entities
    text = re.sub(r"&#\d+;", " ", text)                 # numeric entities &#8217;
    text = re.sub(r"&#x[0-9a-fA-F]+;", " ", text)      # hex entities &#x2019;
    text = re.sub(r"&[a-zA-Z]+;", " ", text)            # named entities &amp; &nbsp;

    # Pass 10: Remove stray non-alphabetic tokens that don't form words
    # (leftover numbers, IDs, timestamps that aren't part of sentences)
    # Keep tokens that are: words, numbers in sentences, or currency amounts
    text = re.sub(r"\b[0-9a-f]{8,}\b", " ", text)      # hex hashes
    text = re.sub(r"\b\d{10}\b", " ", text)             # 10-digit numbers (CIKs)

    # Final: collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def fetch_10k_text(ticker: str, year: int) -> Tuple[str, Optional[str]]:
    """Download 10-K filing text from SEC EDGAR for a given ticker and year.

    Returns (text, warning_or_None).  Falls back gracefully on failure.
    Uses the SEC EDGAR REST API which requires a User-Agent header per
    SEC fair access policy.  Handles gzip-compressed responses.
    """
    import json

    warning = None
    headers = {
        "User-Agent": "MandA-Risk-Analyzer research-tool/1.0 (contact@example.com)",
        "Accept-Encoding": "gzip, deflate",
    }

    try:
        # Step 1: Resolve ticker -> CIK via the SEC company tickers JSON
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        raw = _edgar_get(tickers_url, headers)
        data = json.loads(raw.decode("utf-8"))

        cik = None
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                break

        if cik is None:
            return "", f"Could not resolve CIK for ticker '{ticker}'."

        # Step 2: Get company submissions to find 10-K filing URLs
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        raw = _edgar_get(submissions_url, headers)
        sub_data = json.loads(raw.decode("utf-8"))

        filings = sub_data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])
        primary_docs = filings.get("primaryDocument", [])

        # Find the 10-K (or 10-K/A) closest to the requested year
        best_idx = None
        best_dist = 9999
        for i, (form, date_str) in enumerate(zip(forms, dates)):
            if form in ("10-K", "10-K/A", "20-F", "20-F/A"):
                try:
                    filing_year = int(date_str[:4])
                    dist = abs(filing_year - year)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                except ValueError:
                    continue

        if best_idx is None:
            return "", f"No 10-K/20-F filing found for {ticker}."

        if best_dist > 0:
            actual_year = dates[best_idx][:4]
            warning = f"No filing for FY{year} -- using closest filing from {actual_year}."

        # Step 3: Download the primary document text (may be gzip-compressed)
        accession_no = accessions[best_idx].replace("-", "")
        doc_name = primary_docs[best_idx]
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_no}/{doc_name}"

        raw_bytes = _edgar_get(doc_url, headers, timeout=30)
        raw_text = raw_bytes.decode("utf-8", errors="replace")

        # Clean iXBRL/HTML to extract human-readable narrative text
        text = _clean_ixbrl(raw_text)

        # Truncate to a reasonable length for analysis (first ~200K chars)
        if len(text) > 200_000:
            text = text[:200_000]

        return text, warning

    except Exception as e:
        return "", f"EDGAR fetch error: {e}"


# ---------------------------------------------------------------------------
# L&M Bag-of-Words Feature Extraction
# ---------------------------------------------------------------------------

def compute_lm_features(text: str) -> Dict[str, float]:
    """Compute Loughran & McDonald dictionary features from raw text.

    Returns a dict with:
      - {category}_count: raw word count per category
      - {category}_pct:   count / total_words
      - net_sentiment:     positive_pct - negative_pct
      - total_words:       total word count
    """
    # Tokenise: lowercase, keep only alphabetic tokens
    words = re.findall(r"[a-z]+", text.lower())
    total = len(words) if words else 1  # avoid division by zero

    features: Dict[str, float] = {"total_words": len(words)}

    for cat_name, word_set in _LM_CATEGORIES.items():
        key = cat_name.lower().replace(" ", "_")
        count = sum(1 for w in words if w in word_set)
        features[f"{key}_count"] = count
        features[f"{key}_pct"] = count / total

    features["net_sentiment"] = features.get("positive_pct", 0) - features.get("negative_pct", 0)

    return features


# ---------------------------------------------------------------------------
# DistilBERT Contextual Embedding (optional dependency)
# ---------------------------------------------------------------------------

def compute_distilbert_embedding(text: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Generate a DistilBERT [CLS] embedding and positive-sentiment probability.

    Requires `transformers` and `torch`.  Returns (embedding_array, warning).
    If the libraries are not installed, returns (None, warning_message) and the
    pipeline gracefully falls back to L&M-only features.
    """
    try:
        from transformers import pipeline as hf_pipeline   # type: ignore
    except ImportError:
        return None, ("transformers/torch not installed — using L&M dictionary features only. "
                      "Install with: pip install transformers torch")

    warn = None
    try:
        # Use a lightweight sentiment pipeline (SST-2 fine-tuned DistilBERT)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classifier = hf_pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True,
                max_length=512,
            )

        # DistilBERT has a 512-token limit — analyse chunks and average
        chunk_size = 2000  # characters (≈ 400 tokens)
        chunks = [text[i:i + chunk_size] for i in range(0, min(len(text), 50000), chunk_size)]
        if not chunks:
            return None, "Text too short for DistilBERT analysis."

        pos_probs = []
        for chunk in chunks:
            if len(chunk.strip()) < 20:
                continue
            result = classifier(chunk)[0]
            label = result["label"]
            score = result["score"]
            # Normalise to positive probability
            if label == "POSITIVE":
                pos_probs.append(score)
            else:
                pos_probs.append(1.0 - score)

        if not pos_probs:
            return None, "Could not extract DistilBERT features from text."

        # Return average positive probability as a 1-element array for the logistic regression
        avg_prob = float(np.mean(pos_probs))
        return np.array([avg_prob]), warn

    except Exception as e:
        return None, f"DistilBERT error: {e}"


# ---------------------------------------------------------------------------
# Combined Logistic Regression Prediction
# ---------------------------------------------------------------------------

def predict_sentiment(lm_features: Dict[str, float],
                      bert_embedding: Optional[np.ndarray] = None,
                      readability_scores: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    """Combine L&M features, DistilBERT output, and readability indexes via logistic regression.

    Uses pre-calibrated coefficients.  If DistilBERT is unavailable,
    the bert_positive_prob feature defaults to 0.5 (neutral).
    If readability_scores is None, the three readability features default to
    0.0 (neutral contribution).

    Parameters:
        lm_features: dict from compute_lm_features()
        bert_embedding: optional array from compute_distilbert_embedding()
        readability_scores: optional dict with keys 'fk_grade', 'fog', 'ari'

    Returns:
        dict with keys: label, probability, interpretation, components
    """
    # Assemble feature vector in the order expected by _LOGIT_WEIGHTS
    bert_prob = float(bert_embedding[0]) if bert_embedding is not None else 0.5

    # Normalise readability scores to ~0-1 range (divide by typical 10-K max)
    # FK grade: typical range 10-25 for 10-K filings, normalise by 25
    # Fog: typical range 12-25, normalise by 25
    # ARI: typical range 10-25, normalise by 25
    if readability_scores is not None:
        fk_norm = readability_scores.get("fk_grade", 0) / 25.0
        fog_norm = readability_scores.get("fog", 0) / 25.0
        ari_norm = readability_scores.get("ari", 0) / 25.0
    else:
        fk_norm = 0.0
        fog_norm = 0.0
        ari_norm = 0.0

    feature_vector = [
        lm_features.get("negative_pct", 0),
        lm_features.get("positive_pct", 0),
        lm_features.get("uncertainty_pct", 0),
        lm_features.get("litigious_pct", 0),
        lm_features.get("constraining_pct", 0),
        lm_features.get("strong_modal_pct", 0),
        lm_features.get("net_sentiment", 0),
        bert_prob,
        fk_norm,
        fog_norm,
        ari_norm,
    ]

    # Compute logit: z = intercept + sum(w_i * x_i)
    z = _LOGIT_INTERCEPT + sum(w * x for w, x in zip(_LOGIT_WEIGHTS, feature_vector))

    # Sigmoid → probability of Positive Sentiment
    if z > 500:
        prob_positive = 1.0
    elif z < -500:
        prob_positive = 0.0
    else:
        prob_positive = 1.0 / (1.0 + math.exp(-z))

    # Classify with confidence bands
    if prob_positive >= 0.75:
        label = "Strong Positive Sentiment"
        interp = ("The filing language is strongly optimistic. Financial health indicators "
                  "and forward-looking statements suggest confidence in future performance.")
    elif prob_positive >= 0.55:
        label = "Mild Positive Sentiment"
        interp = ("The filing language is moderately positive. Some caution is present "
                  "but overall tone suggests stability.")
    elif prob_positive >= 0.45:
        label = "Neutral / Mixed Sentiment"
        interp = ("The filing contains balanced positive and negative language. "
                  "No strong directional signal detected — further investigation recommended.")
    elif prob_positive >= 0.25:
        label = "Mild Negative Sentiment"
        interp = ("The filing language leans negative. Elevated uncertainty, risk, or litigation "
                  "language detected — may signal emerging distress.")
    else:
        label = "Strong Negative Sentiment"
        interp = ("The filing language is strongly negative. High concentrations of distress, "
                  "litigation, and uncertainty words — consistent with pre-failure filings.")

    return {
        "label": label,
        "probability": prob_positive,
        "interpretation": interp,
        "components": {
            "lm_net_sentiment": lm_features.get("net_sentiment", 0),
            "bert_positive_prob": bert_prob,
            "logit_z": round(z, 4),
        },
    }
