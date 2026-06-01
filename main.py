import requests
import smtplib
import time
import random
import json
import re
from email.mime.text import MIMEText
from datetime import datetime
from openai import OpenAI
import os
import sys

# ---------- 配置 ----------
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_SOURCES_URL = "https://api.openalex.org/sources"
MAX_PAPERS_TO_FETCH = 100          # 一次获取的论文数
TOP_N_PAPERS = 30                  # 最终推送前 30 篇
BATCH_TRANSLATE_SIZE = 5           # 每批翻译的论文数（避免 token 超限）

# ---------- 1. 获取风电论文（按最新日期） ----------
def fetch_papers(query, limit=MAX_PAPERS_TO_FETCH):
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    params = {
        "query": query,
        "limit": limit,
        "sort": "publicationDate",   # 按最新发表排序
        "fields": "title,url,abstract,publicationDate,externalIds,journal"
    }

    print(f"🔍 Semantic Scholar 请求: limit={limit}, query={query}")
    for i in range(5):
        try:
            resp = requests.get(SEMANTIC_SCHOLAR_URL, params=params, headers=headers, timeout=30)
            if resp.status_code == 429:
                wait = (2 ** i) * 5 + random.uniform(0, 5)
                print(f"⏳ 429 限流，等待 {wait:.1f}s ...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            papers = data.get("data", [])
            print(f"✅ 获取到 {len(papers)} 篇论文")
            return papers
        except Exception as e:
            print(f"❌ 请求失败 {i+1}/5: {e}")
            time.sleep(10)
    return []

# ---------- 2. 通过 OpenAlex 获取期刊影响因子（2yr_mean_citedness） ----------
def get_journal_impact_factor(issn_list):
    """
    输入 ISSN 列表，返回 dict: { issn: { 'name': ..., 'if': ... } }
    """
    if not issn_list:
        return {}
    # OpenAlex 的 sources 端点支持用 ISSN 批量过滤
    filters = "issn:" + "|".join(issn_list)
    params = {
        "filter": filters,
        "per_page": 200,       # 一次最多 200
        "select": "issn,display_name,summary_stats"
    }
    print(f"📊 查询 OpenAlex 期刊指标，ISSN 数量: {len(issn_list)}")
    try:
        resp = requests.get(OPENALEX_SOURCES_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if_dict = {}
        for src in results:
            issn_array = src.get("issn", [])
            if not issn_array:
                continue
            # 取第一个 ISSN
            issn_val = issn_array[0]
            name = src.get("display_name", "Unknown")
            stats = src.get("summary_stats", {})
            # 2yr_mean_citedness 类似于影响因子，若无则用 h-index 或 0
            impact = stats.get("2yr_mean_citedness", 0)
            if_dict[issn_val] = {
                "name": name,
                "if": impact
            }
        print(f"✅ 获得 {len(if_dict)} 本期刊指标")
        return if_dict
    except Exception as e:
        print(f"❌ OpenAlex 查询失败: {e}")
        return {}

# ---------- 3. 给论文附加期刊影响因子 ----------
def enrich_papers_with_if(papers):
    # 收集所有的 ISSN（或电子 ISSN）
    issn_set = set()
    for p in papers:
        ext_ids = p.get("externalIds", {}) or {}
        issn = ext_ids.get("Issn") or ext_ids.get("Eissn")
        if issn:
            issn_set.add(issn)
    issn_list = list(issn_set)

    # 获取影响因子映射
    if_map = get_journal_impact_factor(issn_list)

    enriched = []
    for p in papers:
        ext_ids = p.get("externalIds", {}) or {}
        issn = ext_ids.get("Issn") or ext_ids.get("Eissn")
        journal_info = p.get("journal", {}) or {}
        journal_name = journal_info.get("name") or "Unknown"

        if issn and issn in if_map:
            impact = if_map[issn]["if"]
            # 如果 OpenAlex 有更好的期刊名，可以用它
            journal_name = if_map[issn]["name"] or journal_name
        else:
            impact = 0.0

        enriched.append({
            "title": p.get("title", ""),
            "url": p.get("url", ""),
            "abstract": p.get("abstract") or "",
            "publicationDate": p.get("publicationDate", "Unknown"),
            "journal": journal_name,
            "impactFactor": impact,
            "issn": issn
        })
    # 按影响因子降序排序
    enriched.sort(key=lambda x: x["impactFactor"], reverse=True)
    print(f"📈 按 IF 排序完成，最高 IF: {enriched[0]['impactFactor']:.2f}" if enriched and enriched[0]['impactFactor']>0 else "⚠️ 部分论文缺乏影响因子")
    return enriched[:TOP_N_PAPERS]

# ---------- 4. DeepSeek 分批翻译 ----------
def translate_batch(papers_subset):
    """
    翻译一个子集（列表），返回与子集对应的 translations 列表
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return [{"title_zh": "", "abstract_zh": ""} for _ in papers_subset]

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    items = []
    for i, p in enumerate(papers_subset):
        items.append({
            "index": i,
            "title": p["title"],
            "abstract": p["abstract"]   # 全文
        })

    prompt = "请将以下风电领域论文的标题和摘要翻译成中文，摘要需完整翻译。\n"
    for item in items:
        prompt += f"\n[{item['index']}]\n标题: {item['title']}\n摘要: {item['abstract']}\n"
    prompt += "\n请返回一个 JSON 数组，每个元素包含 index, title_zh, abstract_zh，直接返回 JSON 不要加任何解释。\n示例：[{\"index\": 0, \"title_zh\": \"...\", \"abstract_zh\": \"...\"}, ...]"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4000  # 根据摘要总长度调整
        )
        content = response.choices[0].message.content
        # 提取 JSON 部分
        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        translations = json.loads(content)
        result = [{"title_zh": "", "abstract_zh": ""} for _ in papers_subset]
        for t in translations:
            idx = t.get("index")
            if idx is not None and idx < len(result):
                result[idx]["title_zh"] = t.get("title_zh", "")
                result[idx]["abstract_zh"] = t.get("abstract_zh", "")
        return result
    except Exception as e:
        print(f"⚠️ 翻译子集失败: {e}")
        return [{"title_zh": "", "abstract_zh": ""} for _ in papers_subset]

def translate_all_papers(papers):
    """分批翻译所有论文，返回完整的 translations 列表"""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("未配置 DEEPSEEK_API_KEY，跳过翻译。")
        return [{"title_zh": "", "abstract_zh": ""} for _ in papers]

    all_translations = []
    total = len(papers)
    for start in range(0, total, BATCH_TRANSLATE_SIZE):
        end = min(start + BATCH_TRANSLATE_SIZE, total)
        batch = papers[start:end]
        print(f"🌐 翻译第 {start+1}-{end} 篇 (共 {total})")
        batch_trans = translate_batch(batch)
        all_translations.extend(batch_trans)
    return all_translations

# ---------- 5. 构建邮件正文 ----------
def build_email_body(papers, translations):
    lines = [f"📬 风电热点论文日报（按期刊影响因子排序） - {datetime.now().strftime('%Y-%m-%d')}\n"]
    lines.append(f"共筛选出 {len(papers)} 篇高影响力期刊论文\n")
    for i, p in enumerate(papers):
        title_en = p["title"]
        url = p["url"]
        abstract_en = p["abstract"]
        pub_date = p["publicationDate"]
        journal = p["journal"]
        impact = p["impactFactor"]

        # 获取中文翻译
        title_zh = ""
        abstract_zh = ""
        if translations and i < len(translations):
            title_zh = translations[i].get("title_zh", "")
            abstract_zh = translations[i].get("abstract_zh", "")

        lines.append(f"{'='*50}")
        lines.append(f"📌 第 {i+1} 篇 | 影响因子: {impact:.2f} | 发表: {pub_date}")
        lines.append(f"📰 期刊: {journal}")
        lines.append(f"📄 英文标题: {title_en}")
        if title_zh:
            lines.append(f"📄 中文标题: {title_zh}")
        lines.append(f"🔗 链接: {url}")
        lines.append(f"📝 英文摘要:\n{abstract_en}")
        if abstract_zh:
            lines.append(f"📝 中文摘要:\n{abstract_zh}")
        lines.append("")
    return "\n".join(lines)

# ---------- 6. 发送邮件 ----------
def send_email(content):
    sender = os.environ.get("EMAIL_SENDER")
    password = os.environ.get("EMAIL_PASSWORD")
    receiver = os.environ.get("EMAIL_RECEIVER")

    if not all([sender, password, receiver]):
        print("邮件配置缺失，跳过发送。")
        print(content)
        return

    msg = MIMEText(content, "plain", "utf-8")
    msg["Subject"] = f"风电高IF论文日报 {datetime.now().strftime('%Y-%m-%d')}"
    msg["From"] = sender
    msg["To"] = receiver

    smtp_host = os.environ.get("SMTP_HOST", "smtp.qq.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 465))
    try:
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(sender, password)
            server.sendmail(sender, [receiver], msg.as_string())
        print("✅ 邮件发送成功！")
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")

# ---------- 7. 主流程 ----------
if __name__ == "__main__":
    # 更全面的风电关键词
    query = (
        '"wind turbine" OR "wind farm" OR "wind energy" OR '
        '"offshore wind" OR "onshore wind" OR "wind power" OR '
        '"wind turbine layout optimization" OR "wind turbine CFD" OR '
        '"wind turbine wake" OR "wind turbine structure" OR '
        '"wind turbine blade" OR "wind resource assessment"'
    )
    translate_mode = os.environ.get("TRANSLATE_MODE", "true").lower() == "true"

    print("🚀 开始获取风电论文...")
    papers_raw = fetch_papers(query, limit=MAX_PAPERS_TO_FETCH)
    if not papers_raw:
        send_email("今日未获取到风电论文，请检查脚本日志。")
        sys.exit(0)

    # 附加影响因子并排序取前30
    top_papers = enrich_papers_with_if(papers_raw)

    # 翻译（如启用）
    translations = None
    if translate_mode:
        translations = translate_all_papers(top_papers)
    else:
        translations = [{"title_zh": "", "abstract_zh": ""} for _ in top_papers]

    # 构建并发送邮件
    body = build_email_body(top_papers, translations)
    send_email(body)
