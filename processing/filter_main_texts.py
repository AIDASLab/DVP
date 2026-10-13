#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multilingual cleaner for EN/KR plus Arabic (ar), Dutch (nl), Spanish (es), French (fr),
Japanese (ja), Russian (ru), and Chinese (zh).

Adds language-specific:
  - Ad/spam regex patterns
  - Header/boilerplate regex patterns
  - Machine translation hints
  - Script ratio checks via Unicode ranges

USAGE EXAMPLES
--------------
# English (default)
python -m processing.filter_main_texts --dataset_name culturax_en --output_path ./out --lang en

# Korean with a minimum script ratio
python -m processing.filter_main_texts --dataset_name culturax_ko --output_path ./out --lang ko --min_script_ratio 0.3

# Spanish with LLM refinement
python -m processing.filter_main_texts --dataset_name culturax_es --output_path ./out --lang es --use_llm --llm_model path/to/chat-model

# Chinese, require at least 40% Han characters
python -m processing.filter_main_texts --dataset_name culturax_zh --output_path ./out --lang zh --min_script_ratio 0.4

NOTES
-----
- The language-specific regex sets are intentionally conservative. Tune as needed for your corpus.
- For Latin languages (en, nl, es, fr), script ratio checks are optional; for non‑Latin (ko, ja, zh, ar, ru), they can be helpful.
"""

import os
import re
import csv
import json
import argparse
from typing import List, Tuple, Dict
from collections import Counter

import pandas as pd
from tqdm import tqdm

# -------------------------------
# Dataset loading
# -------------------------------
try:
    from .utils import load_local_dataset
except Exception:
    from utils import load_local_dataset

# The inference backend is imported only when LLM refinement is requested.

# =========================================
# Unicode script ranges (rough, practical)
# =========================================
SCRIPT_RANGES = {
    # Basic Latin + Latin-1 supplement + Latin Extended blocks
    "latin": [
        (0x0041, 0x007A),  # Basic Latin A-z
        (0x00C0, 0x024F),  # Latin-1 + Extended-A/B
        (0x1E00, 0x1EFF),  # Latin Extended Additional
    ],
    "hangul": [
        (0xAC00, 0xD7A3),  # Hangul Syllables
        (0x1100, 0x11FF),  # Hangul Jamo
        (0x3130, 0x318F),  # Hangul Compatibility Jamo
    ],
    "hiragana": [(0x3040, 0x309F)],
    "katakana": [(0x30A0, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9D)],
    "han": [
        (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    ],
    "cyrillic": [(0x0400, 0x04FF), (0x0500, 0x052F)],
    "arabic": [
        (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
        (0xFB50, 0xFDFF), (0xFE70, 0xFEFF),  # Presentation forms
    ],
}

LANG_TO_SCRIPTS = {
    "en": ["latin"],
    "nl": ["latin"],
    "es": ["latin"],
    "fr": ["latin"],
    "ko": ["hangul"],
    "ja": ["hiragana", "katakana", "han"],
    "zh": ["han"],
    "ru": ["cyrillic"],
    "ar": ["arabic"],
}

URL_RE = re.compile(r"https?://|www\\.", re.IGNORECASE)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}")
ALL_CAPS_LINE_RE = re.compile(r"^[^a-z]*[A-Z][^a-z]*$")  # Latin all caps

# Some generic phones (KR kept for retro-compat), plus very loose international pattern
PHONE_RE_KR = re.compile(r"\\b01[016789]-?\\d{3,4}-?\\d{4}\\b")
PHONE_RE_INTL = re.compile(r"\\+?\\d[\\d\-()\\s]{7,}\\d")

# Platform handles (generic)
KAKAO_TELEGRAM_RE = re.compile(r"(카카오톡|카톡|텔레그램|telegram|whatsapp|wechat|微信)\\s*[:：]?\\s*[@]?[A-Za-z0-9_]+", re.IGNORECASE)

# =========================================
# Language-specific regex dictionaries
# =========================================

AD_PATTERNS: Dict[str, List[str]] = {
    "en": [
        r"\\b(casino|betting|sportsbook|slots?|roulette|blackjack|poker)\\b",
        r"\\b(bitcoin|crypto|forex|get rich quick|passive income)\\b",
        r"(subscribe\\s+and\\s+save|limited time offer|click here)",
        r"(sponsored by|paid partnership|advertisement|promoted)",
        r"(adult content|xxx|escort|viagra|cialis|porn)",
    ],
    "ko": [
        r"(유료\\s*광고|광고\\s*포함|협찬|스폰서|후원|프로모션|홍보)",
        r"(카지노|토토|바카라|홀덤|포커|슬롯머신?|맞고|룰렛|스포츠베팅)",
        r"(성인|19\\s*금|야동|에로|AV|청소년\\s*이용\\s*불가)",
        r"(비트코인|코인|가상화폐|선물거래|에어드롭|코인리딩|원금보장)",
        r"(카카오톡|카톡|오픈채팅|텔레그램|라인|위챗|wechat)\\s*ID?",
        r"(토렌트|torrent|불법\\s*다운로드|고속\\s*다운로드)",
        r"(클릭\\s*하세요|바로\\s*가기|지금\\s*구매|한정\\s*특가|특별\\s*할인)",
        r"(대출|작업대출|신용대출|저금리|상담\\s*문의)",
        r"(티케팅|티켓팅|공구|공동구매)",
        r"(원클릭\\s*수익|하루\\s*\\d+만원|수익\\s*보장|N배\\s*수익)",
    ],
    "es": [
        r"(casino|apuestas deportivas|tragamonedas|ruleta|póker|blackjack)",
        r"(bitcoin|cripto|forex|ingresos pasivos|hazte rico|enriquézete rápido)",
        r"(suscríbete y ahorra|oferta por tiempo limitado|haz clic aquí)",
        r"(patrocinado por|contenido patrocinado|publicidad|promocionado)",
        r"(contenido para adultos|xxx|escort|viagra|cialis|porno)",
    ],
    "fr": [
        r"(casino|paris sportifs|machines? à sous|roulette|blackjack|poker)",
        r"(bitcoin|crypto|forex|revenus passifs|devenir riche)",
        r"(abonnez-vous et économisez|offre limitée|cliquez ici)",
        r"(sponsorisé par|partenariat payé|publicité|contenu sponsorisé|promu)",
        r"(contenu adulte|xxx|escort|viagra|cialis|porno)",
    ],
    "nl": [
        r"(casino|sportweddenschappen|gokkasten|roulette|blackjack|poker)",
        r"(bitcoin|crypto|forex|passief inkomen|snel rijk)",
        r"(abonneer en bespaar|tijdelijke aanbieding|klik hier)",
        r"(gesponsord door|betaalde samenwerking|advertentie|gesponsord)",
        r"(volwasseneninhoud|xxx|escort|viagra|cialis|porno)",
    ],
    "ja": [
        r"(カジノ|ブックメーカー|スロット|ルーレット|ブラックジャック|ポーカー)",
        r"(ビットコイン|クリプト|仮想通貨|FX|不労所得|一攫千金)",
        r"(今すぐ登録|期間限定|こちらをクリック)",
        r"(スポンサー|提供|広告|プロモーション|宣伝)",
        r"(アダルト|18禁|エロ|AV|ポルノ|出会い系)",
    ],
    "ru": [
        r"(казино|ставки на спорт|слоты|рулетка|блэкджек|покер)",
        r"(биткоин|крипто|форекс|пассивный доход|быстро разбогатеть)",
        r"(подпишись и сэкономь|ограниченное предложение|нажми здесь)",
        r"(спонсируется|рекламный материал|реклама|продвигается)",
        r"(для взрослых|xxx|эскорт|виагра|сиалис|порно)",
    ],
    "ar": [
        r"(كازينو|رهان رياضي|سلوت|روليت|بلاك[ -]?جاك|بوكر)",
        r"(بيتكوين|كريبتو|فوركس|دخل سلبي|اغتني بسرعة|الثراء السريع)",
        r"(اشترك ووفر|عرض لفترة محدودة|اضغط هنا)",
        r"(برعاية|محتوى مدفوع|إعلان|مُروَّج)",
        r"(محتوى للكبار|اباحي|بورنو|فياجرا|سياليس|مرافقة)",
    ],
    "zh": [
        r"(赌场|体育投注|老虎机|轮盘|二十一点|扑克)",
        r"(比特币|加密货币|外汇|被动收入|快速致富)",
        r"(订阅省钱|限时优惠|点击这里|点击查看)",
        r"(赞助|付费合作|广告|推广)",
        r"(成人内容|色情|黄片|伟哥|西力士|陪伴服务)",
    ],
}

HEADER_PATTERNS: Dict[str, List[str]] = {
    "en": [
        r"^\s*(home|login|sign in|register|about|privacy policy|terms of service)\s*$",
        r"^\s*(copyright|all rights reserved|©\s*\d{4})\s*$",
        r"^\s*share\s*:\s*$",
        r"^\s*navigation\s*$",
    ],
    "ko": [
        r"^\s*(홈|로그인|회원가입|내\s*정보|개인정보처리방침|이용약관)\s*$",
        r"^\s*(목차|본문|바로가기|전체\s*메뉴|카테고리)\s*$",
        r"^\s*(저작권자|무단\s*전재\s*및\s*재배포\s*금지|ⓒ\s*\d{4})\s*$",
        r"^\s*(이전글|다음글|관련\s*기사|댓글\s*\d+)\s*$",
        r"^\s*(공유|구독|공지|보도자료)\s*$",
    ],
    "es": [
        r"^\s*(inicio|entrar|iniciar sesión|registrarse|acerca de|política de privacidad|términos de servicio)\s*$",
        r"^\s*(derechos reservados|©\s*\d{4})\s*$",
        r"^\s*compartir\s*:\s*$",
        r"^\s*navegación\s*$",
    ],
    "fr": [
        r"^\s*(accueil|connexion|se connecter|s'inscrire|à propos|politique de confidentialité|conditions d'utilisation)\s*$",
        r"^\s*(tous droits réservés|©\s*\d{4})\s*$",
        r"^\s*partager\s*:\s*$",
        r"^\s*navigation\s*$",
    ],
    "nl": [
        r"^\s*(home|inloggen|aanmelden|registreren|over|privacybeleid|gebruiksvoorwaarden)\s*$",
        r"^\s*(alle rechten voorbehouden|©\s*\d{4})\s*$",
        r"^\s*delen\s*:\s*$",
        r"^\s*navigatie\s*$",
    ],
    "ja": [
        r"^\s*(ホーム|ログイン|サインイン|新規登録|会社概要|プライバシーポリシー|利用規約)\s*$",
        r"^\s*(著作権|無断転載禁止|©\s*\d{4})\s*$",
        r"^\s*共有\s*:\s*$",
        r"^\s*ナビゲーション\s*$",
    ],
    "ru": [
        r"^\s*(главная|вход|войти|регистрация|о нас|политика конфиденциальности|условия использования)\s*$",
        r"^\s*(все права защищены|©\s*\d{4})\s*$",
        r"^\s*поделиться\s*:\s*$",
        r"^\s*навигация\s*$",
    ],
    "ar": [
        r"^\s*(الرئيسية|تسجيل الدخول|إنشاء حساب|حول|سياسة الخصوصية|شروط الخدمة)\s*$",
        r"^\s*(جميع الحقوق محفوظة|©\s*\d{4})\s*$",
        r"^\s*مشاركة\s*:\s*$",
        r"^\s*التنقل\s*$",
    ],
    "zh": [
        r"^\s*(首页|登录|注册|关于我们|隐私政策|服务条款)\s*$",
        r"^\s*(版权所有|©\s*\d{4})\s*$",
        r"^\s*分享\s*:\s*$",
        r"^\s*导航\s*$",
    ],
}

TRANSLATION_HINTS: Dict[str, List[str]] = {
    "en": [r"translated by", r"machine[-\s]?translated", r"\bMT\s+output\b", r"google translate", r"deepl"],
    "ko": [r"(번역\s*됨|자동\s*번역|기계\s*번역|번역\s*결과)", r"(원문|영문\s*번역|한글\s*번역|의역|직역)", r"(구글\s*번역|딥엘|DeepL|파파고|Papago)"],
    "es": [r"traducido por", r"traducción automática", r"traducido automáticamente", r"google traductor|google translate", r"deepl"],
    "fr": [r"traduit par", r"traduction automatique", r"traduction machine", r"google traduction|google translate", r"deepl"],
    "nl": [r"vertaald door", r"machinale vertaling", r"automatisch vertaald", r"google translate", r"deepl"],
    "ja": [r"翻訳者|翻訳 した", r"機械 翻訳|自動 翻訳|機械翻訳|自動翻訳", r"google 翻訳|Google 翻訳|google translate", r"deepl"],
    "ru": [r"переведено\s+.*", r"машинный перевод|автоматический перевод", r"google translate|гугл переводчик", r"deepl"],
    "ar": [r"ترجمة بواسطة", r"ترجمة آلية|ترجمة تلقائية|مترجم آلي", r"google translate|ترجمة جوجل", r"deepl"],
    "zh": [r"由.*翻译", r"机器翻译|自动翻译", r"谷歌翻译|google translate", r"deepl"],
}

# =========================================
# Helpers
# =========================================

def _in_ranges(cp: int, ranges: List[Tuple[int, int]]) -> bool:
    for a, b in ranges:
        if a <= cp <= b:
            return True
    return False


def script_ratio(text: str, lang: str) -> float:
    """Approximate ratio of characters belonging to the primary scripts of a language."""
    if not text:
        return 0.0
    scripts = LANG_TO_SCRIPTS.get(lang, [])
    if not scripts:
        return 0.0
    total = len(text)
    hits = 0
    for ch in text:
        cp = ord(ch)
        for s in scripts:
            if _in_ranges(cp, SCRIPT_RANGES.get(s, [])):
                hits += 1
                break
    return hits / max(1, total)


def looks_like_header_or_boiler(text: str, lang: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    short_lines = [ln for ln in lines if len(ln) <= 40]
    pats = HEADER_PATTERNS.get(lang, []) + HEADER_PATTERNS.get("en", [])
    header_hits = sum(1 for ln in lines if any(re.search(p, ln, re.IGNORECASE) for p in pats))
    all_caps_hits = sum(1 for ln in short_lines if ALL_CAPS_LINE_RE.match(ln) and len(ln) <= 30)
    return (header_hits >= 2) or (all_caps_hits >= 3)


def is_probably_ad(text: str, lang: str) -> bool:
    t = text
    tl = t.lower()
    pats = AD_PATTERNS.get(lang, []) + AD_PATTERNS.get("en", [])
    if any(re.search(p, t if lang not in ("en", "nl", "es", "fr") else tl) for p in pats):
        return True
    tokens = t.split()
    if tokens:
        url_count = sum(1 for tok in tokens if URL_RE.search(tok))
        if url_count / max(1, len(tokens)) > 0.1:
            return True
    if PHONE_RE_KR.search(t) or PHONE_RE_INTL.search(t) or KAKAO_TELEGRAM_RE.search(t):
        return True
    return False


def is_probably_translation_by_heuristics(text: str, lang: str) -> bool:
    t = text
    tl = t.lower()
    pats = TRANSLATION_HINTS.get(lang, []) + TRANSLATION_HINTS.get("en", [])
    if any(re.search(p, t if lang not in ("en", "nl", "es", "fr") else tl) for p in pats):
        return True
    # lightweight spacing/punctuation oddity detector
    weird = len(re.findall(r"\s[\,\.\!\?\:\;]\s", t)) + len(re.findall(r"\s{2,}", t))
    if weird > 10 and len(t) < 4000:
        return True
    return False


def quick_quality_checks(text: str, lang: str, min_script_ratio: float = 0.0) -> Tuple[bool, str]:
    if not isinstance(text, str):
        return False, "non_string"
    t = text.strip()
    if len(t) < 30:
        return False, "too_short"
    if min_script_ratio > 0:
        ratio = script_ratio(t, lang)
        if ratio < min_script_ratio:
            return False, "low_script_ratio"
    urls = len(URL_RE.findall(t))
    emails = len(EMAIL_RE.findall(t))
    if (urls + emails) >= 5 and (urls + emails) / max(1, len(t.split())) > 0.05:
        return False, "urlspam"
    return True, ""


def extract_main_text(text: str, lang: str) -> str:
    if not isinstance(text, str):
        return ""
    lines = [ln.rstrip() for ln in text.splitlines()]
    filtered: List[str] = []
    pats = HEADER_PATTERNS.get(lang, []) + HEADER_PATTERNS.get("en", [])
    for ln in lines:
        ln_stripped = ln.strip()
        if not ln_stripped:
            filtered.append(ln)
            continue
        if any(re.search(p, ln_stripped, re.IGNORECASE) for p in pats):
            continue
        if len(ln_stripped) <= 30 and ALL_CAPS_LINE_RE.match(ln_stripped):
            continue
        filtered.append(ln)
    cleaned = "\n".join(filtered).strip()
    blocks = [blk.strip() for blk in re.split(r"\n\s*\n", cleaned) if blk.strip()]
    if len(blocks) >= 3:
        return max(blocks, key=len)
    return cleaned

# =========================================
# LLM judge (persistent model instance)
# =========================================

LLM_SYSTEM_PROMPT = (
    "You are a strict data cleaner for multilingual text (EN/KO/AR/NL/ES/FR/JA/RU/ZH). "
    "Decide which category the text belongs to:\n"
    "1) KEEP_MAIN_TEXT (good main content; titles allowed)\n"
    "2) AD_SPAM (ads/promos/casinos/도박/رهان/ставки/賭博/广告 等)\n"
    "3) HEADER_BOILERPLATE (navigation/menu/footer/login/privacy/etc.)\n"
    "4) MACHINE_TRANSLATED (obvious machine-translated text)\n\n"
    "Return EXACTLY ONE LABEL from {KEEP_MAIN_TEXT, AD_SPAM, HEADER_BOILERPLATE, MACHINE_TRANSLATED}."
)

LLM_USER_TEMPLATE = (
    "Classify the following text. Only return one label.\n\n"
    "LANG: {lang}\n\n"
    "TEXT:\n"
    "<<<\n"
    "{text}\n"
    ">>>"
)

LABELS = {"KEEP_MAIN_TEXT", "AD_SPAM", "HEADER_BOILERPLATE", "MACHINE_TRANSLATED"}

class LLMJudge:
    """Keep a single LLMModel in memory and reuse it across batches."""
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-32B",
        tensor_parallel_size: int = 1,
        max_new_tokens: int = 4,
        temperature: float = 0.0,
    ):
        from model import LLMModel
        self.model = LLMModel(
            model=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device="cuda",
            dtype="auto",
            model_dir=None,
            system_prompt=LLM_SYSTEM_PROMPT,
            tensor_parallel_size=tensor_parallel_size,
        )

    def classify(self, texts: List[str], lang: str) -> List[str]:
        prompts = [LLM_USER_TEMPLATE.format(text=txt[:6000], lang=lang) for txt in texts]
        raw = self.model(prompts)
        labels = []
        for r in raw:
            m = re.search(r"(KEEP_MAIN_TEXT|AD_SPAM|HEADER_BOILERPLATE|MACHINE_TRANSLATED)", str(r).upper())
            labels.append(m.group(1) if m else "KEEP_MAIN_TEXT")
        return labels

# =========================================
# Pipeline
# =========================================

def clean_with_pipeline(
    dataset_name: str,
    output_path: str,
    use_llm: bool = False,
    llm_model: str = "Qwen/Qwen3-32B",
    tensor_parallel_size: int = 1,
    batch_size: int = 128,
    lang: str = "en",
    min_script_ratio: float = 0.0,
    expect_korean: bool = False,  # maps to lang="ko" if True
) -> str:
    """
    Loads df via utils.load_local_dataset(dataset_name) (expects 'text' column).
    Applies heuristics + optional LLM filtering. Saves {dataset_name}_cleaned.csv.

    lang: one of {en, ko, ar, nl, es, fr, ja, ru, zh}
    min_script_ratio: require minimum fraction of characters from the target script(s)
    """
    if expect_korean and lang == "en":
        lang = "ko"

    if lang not in LANG_TO_SCRIPTS:
        raise ValueError(f"Unsupported --lang '{lang}'. Supported: {sorted(LANG_TO_SCRIPTS.keys())}")

    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, f"{dataset_name}_cleaned.csv")

    df = load_local_dataset(dataset_name)
    if "text" not in df.columns:
        raise ValueError("DataFrame must include a 'text' column.")

    # 1) main-text extraction (lightweight)
    tqdm.pandas(desc="extract_main_text")
    df["main_text"] = df["text"].progress_apply(lambda x: extract_main_text(x, lang))

    # 2) heuristic drop flags
    keep_mask: List[bool] = []
    reasons: List[str] = []

    for t in tqdm(df["main_text"].astype(str).tolist(), desc="heuristic_filter"):
        ok, reason = quick_quality_checks(t, lang=lang, min_script_ratio=min_script_ratio)
        if not ok:
            keep_mask.append(False)
            reasons.append(reason)
            continue
        if looks_like_header_or_boiler(t, lang=lang):
            keep_mask.append(False)
            reasons.append("boilerplate")
            continue
        if is_probably_ad(t, lang=lang):
            keep_mask.append(False)
            reasons.append("ad_spam")
            continue
        if is_probably_translation_by_heuristics(t, lang=lang):
            keep_mask.append(False)
            reasons.append("translation_heuristic")
            continue
        keep_mask.append(True)
        reasons.append("heuristic_keep")

    df["heuristic_keep"] = keep_mask
    df["heuristic_reason"] = reasons

    # 3) Optional LLM refinement (single persistent instance)
    if use_llm:
        judge = LLMJudge(
            model_name=llm_model,
            tensor_parallel_size=tensor_parallel_size,
            max_new_tokens=4,
            temperature=0.0,
        )
        idx_to_score = df.index[df["heuristic_keep"] == True].tolist()
        texts_to_score = df.loc[idx_to_score, "main_text"].astype(str).tolist()

        llm_labels: List[str] = []
        for i in tqdm(range(0, len(texts_to_score), batch_size), desc="llm_filter"):
            chunk = texts_to_score[i:i+batch_size]
            llm_labels.extend(judge.classify(chunk, lang=lang))

        df.loc[idx_to_score, "llm_label"] = llm_labels

        df["final_keep"] = df.apply(
            lambda r: (r["heuristic_keep"] is True) and (str(r.get("llm_label", "KEEP_MAIN_TEXT")).upper() == "KEEP_MAIN_TEXT"),
            axis=1,
        )
        def final_reason(row):
            if not row["heuristic_keep"]:
                return row["heuristic_reason"]
            lbl = str(row.get("llm_label", "KEEP_MAIN_TEXT")).upper()
            if lbl != "KEEP_MAIN_TEXT":
                return f"llm_{lbl.lower()}"
            return "keep"
        df["final_reason"] = df.apply(final_reason, axis=1)
    else:
        df["llm_label"] = None
        df["final_keep"] = df["heuristic_keep"]
        df["final_reason"] = df["heuristic_reason"].where(df["heuristic_keep"] == False, other="keep")

    cleaned = df[df["final_keep"] == True].copy()
    cleaned = cleaned.drop(columns=["heuristic_keep"], errors="ignore")
    cleaned.rename(columns={"main_text": "text_clean"}, inplace=True)

    cols = [c for c in ["id", "unique_id"] if c in cleaned.columns]
    cols += ["text_clean", "final_reason"]
    if "text" in cleaned.columns:
        cols.append("text")
    if "llm_label" in cleaned.columns:
        cols.append("llm_label")

    cleaned.to_csv(out_file, index=False, quoting=csv.QUOTE_MINIMAL, columns=[c for c in cols if c in cleaned.columns])

    kept = len(cleaned)
    total = len(df)
    dropped = total - kept
    summary = Counter(df["final_reason"].fillna("unknown").tolist())
    print(f"\nSaved: {out_file}")
    print(f"Total: {total} | Kept: {kept} | Dropped: {dropped}")
    print("Drop reasons:", json.dumps(summary, ensure_ascii=False, indent=2))
    return out_file

# =========================================
# CLI
# =========================================

def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Clean a local text dataset: remove ads, headers, and machine-translated texts (multilingual)."
        )
    )
    ap.add_argument("--dataset_name", type=str, required=True, help="Name handled by utils.load_local_dataset()")
    ap.add_argument("--output_path", type=str, required=True, help="Directory to save {dataset_name}_cleaned.csv")
    ap.add_argument("--use_llm", action="store_true", help="Enable LLM refinement (default off)")
    ap.add_argument("--llm_model", type=str, default="Qwen/Qwen3-32B", help="Model name for LLMModel")
    ap.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallelism for local LLM")
    ap.add_argument("--batch_size", type=int, default=128, help="LLM scoring batch size")
    ap.add_argument(
        "--lang",
        type=str,
        default="en",
        choices=sorted(LANG_TO_SCRIPTS.keys()),
        help="Language code: en, ko, ar, nl, es, fr, ja, ru, zh",
    )
    # Minimum fraction of characters in the language's target script.
    ap.add_argument("--min_script_ratio", type=float, default=0.0, help="Minimum target-script char ratio to keep (e.g., 0.3)")
    # Korean-specific aliases.
    ap.add_argument("--expect_korean", action="store_true", help="Set the target language to Korean, equivalent to --lang ko")
    ap.add_argument("--min_hangul_ratio", type=float, default=None, help="Set --min_script_ratio when that option is zero")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    min_ratio = args.min_script_ratio
    if args.min_hangul_ratio is not None and min_ratio == 0.0:
        min_ratio = args.min_hangul_ratio

    clean_with_pipeline(
        dataset_name=args.dataset_name,
        output_path=args.output_path,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size,
        lang=args.lang,
        min_script_ratio=min_ratio,
        expect_korean=args.expect_korean,
    )
