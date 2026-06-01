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


# ---------- 1. 获取风电热点论文（带 API Key 和指数退避） ----------
def fetch_hot_papers(query, limit=10, year=""):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "sort": "citationCount",
        "fields": "title,url,abstract,publicationDate,citationCount"
    }
    if year:
        params["year"] = f"{year}-"

    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    print(f"🔍 请求参数: {params}")
    print(f"🔑 使用 API Key: {'是' if api_key else '否'}")

    max_retries = 5
    for i in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            print(f"📡 状态码: {resp.status_code}")
            if resp.status_code == 429:
                wait = (2 ** i) * 5 + random.uniform(0, 5)
                print(f"⏳ 触发限流，等待 {wait:.1f} 秒后重试...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            papers = data.get("data", [])
            print(f"✅ 获取到 {len(papers)} 篇论文")
            if papers:
                print(f"   示例: {papers[0].get('title')}")
            return papers
        except Exception as e:
            print(f"❌ 请求失败，重试 {i+1}/{max_retries}: {e}")
            time.sleep(10)
    return []

# ---------- 2. 批量翻译（使用 deepseek） ----------
def translate_papers_batch(papers):
    """
    使用 DeepSeek 批量翻译论文标题和摘要。
    返回列表，每个元素含 'title_zh', 'abstract_zh'
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("未配置 DEEPSEEK_API_KEY，跳过翻译。")
        return [{"title_zh": "", "abstract_zh": ""} for _ in papers]

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # 构建待翻译项
    items = []
    for i, p in enumerate(papers):
        title = p.get("title", "")
        abstract = (p.get("abstract") or "")[:500]
        items.append({"index": i, "title": title, "abstract": abstract})

    prompt = "请将以下风电领域论文的标题和摘要翻译成中文。\n"
    for item in items:
        prompt += f"\n[{item['index']}]\n标题: {item['title']}\n摘要: {item['abstract']}\n"
    prompt += "\n请返回一个 JSON 数组，每个元素包含 index, title_zh, abstract_zh，直接返回 JSON 不要加任何解释。\n示例：[{\"index\": 0, \"title_zh\": \"...\", \"abstract_zh\": \"...\"}, ...]"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",      # 也可用 "deepseek-reasoner" 但 chat 就够了
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=3000
        )
        content = response.choices[0].message.content
        # 提取 JSON 部分
        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        translations = json.loads(content)
        result = [{"title_zh": "", "abstract_zh": ""} for _ in papers]
        for t in translations:
            idx = t.get("index")
            if idx is not None and idx < len(result):
                result[idx]["title_zh"] = t.get("title_zh", "")
                result[idx]["abstract_zh"] = t.get("abstract_zh", "")
        return result
    except Exception as e:
        print(f"DeepSeek 翻译失败: {e}")
        return [{"title_zh": "", "abstract_zh": ""} for _ in papers]

# ---------- 3. 一句话总结（可选） ----------
def ai_summarize_paper(title, abstract):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key or not abstract:
        return ""
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = f"请用中文一句话总结下面风电领域论文的核心贡献，不要超过50个字。\n标题：{title}\n摘要：{abstract[:500]}"
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"AI总结失败: {e}")
        return ""

# ---------- 4. 构建邮件正文 ----------
def build_email_body(papers, translations=None, use_ai=False):
    lines = [f"📬 风电热点论文日报 - {datetime.now().strftime('%Y-%m-%d')}\n"]
    lines.append(f"共检索到 {len(papers)} 篇高引用论文\n")
    for i, p in enumerate(papers):
        title_en = p.get("title", "无标题")
        url = p.get("url", "")
        abstract_en = (p.get("abstract") or "无摘要")[:300].replace("\n", " ")
        citation = p.get("citationCount", 0)
        pub_date = p.get("publicationDate") or "未知"

        title_zh = ""
        abstract_zh = ""
        if translations and i < len(translations):
            title_zh = translations[i].get("title_zh", "")
            abstract_zh = translations[i].get("abstract_zh", "")

        lines.append(f"{'='*40}")
        lines.append(f"📌 第{i+1}篇 | 引用量：{citation} | 发表：{pub_date}")
        lines.append(f"📄 英文标题：{title_en}")
        if title_zh:
            lines.append(f"📄 中文标题：{title_zh}")
        lines.append(f"🔗 链接：{url}")

        if use_ai and abstract_en:
            summary = ai_summarize_paper(title_en, abstract_en)
            if summary:
                lines.append(f"🤖 AI一句话：{summary}")

        lines.append(f"📝 英文摘要：{abstract_en}...")
        if abstract_zh:
            lines.append(f"📝 中文摘要：{abstract_zh}...")
        lines.append("")
    return "\n".join(lines)

# ---------- 5. 发送邮件 ----------
def send_email(content, subject_prefix="风电热点论文"):
    sender = os.environ.get("EMAIL_SENDER")
    password = os.environ.get("EMAIL_PASSWORD")
    receiver = os.environ.get("EMAIL_RECEIVER")

    if not all([sender, password, receiver]):
        print("邮件配置缺失，跳过发送。")
        print(content)
        return

    msg = MIMEText(content, "plain", "utf-8")
    msg["Subject"] = f"{subject_prefix} - {datetime.now().strftime('%Y-%m-%d')}"
    msg["From"] = sender
    msg["To"] = receiver

    smtp_host = os.environ.get("SMTP_HOST", "smtp.qq.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 465))
    try:
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(sender, password)
            server.sendmail(sender, [receiver], msg.as_string())
        print("邮件发送成功！")
    except Exception as e:
        print(f"邮件发送失败: {e}")

# ---------- 6. 主流程 ----------
if __name__ == "__main__":
    query = '"wind energy" OR "wind turbine" OR "wind power"'
    use_ai = os.environ.get("USE_AI", "false").lower() == "true"
    translate_mode = os.environ.get("TRANSLATE_MODE", "false").lower() == "true"

    print(f"开始检索: {query}")
    papers = fetch_hot_papers(query, limit=10, year="2024")  # 改为2024，保证有高引用论文

    if not papers:
        print("未检索到论文，发送空报告邮件。")
        send_email("今日未获取到风电热点论文，请检查脚本日志。", "风电论文日报-错误")
        sys.exit(0)

    translations = None
    if translate_mode:
        print("正在进行批量翻译...")
        translations = translate_papers_batch(papers)

    body = build_email_body(papers, translations=translations, use_ai=use_ai)
    send_email(body)
