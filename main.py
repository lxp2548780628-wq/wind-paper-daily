import requests
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import os
import sys
import json
import re

# ---------- 1. 获取风电热点论文 ----------
def fetch_hot_papers(query, limit=10, year=""):
    """
    从 Semantic Scholar 获取论文，按引用量排序
    year: 可限定年份，如 "2026"，留空则不限制
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "sort": "citationCount",       # 关键：按引用量降序 -> 热点
        "fields": "title,url,abstract,publicationDate,citationCount"
    }
    # 可选：限制年份，让热点不至于太老
    if year:
        params["year"] = f"{year}-"    # 格式如 2026-
    
    # 增加重试机制，防止网络波动
    for i in range(3):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [])
        except Exception as e:
            print(f"请求失败，重试 {i+1}/3: {e}")
    return []

# ---------- 2. （可选）用 Claude 生成一句话总结 ----------
def ai_summarize_paper(title, abstract):
    """
    调用 Anthropic Claude API，生成中文一句话总结
    如果不想使用 AI 或没有 Key，本函数可被跳过
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not abstract:
        return ""  # 没有Key或没有摘要就直接返回空
    
    # 使用官方 Python SDK 需要安装 anthropic，这里直接用 requests 调用
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    prompt = f"请用中文一句话总结下面风电领域论文的核心贡献，不要超过50个字。\n标题：{title}\n摘要：{abstract[:500]}"
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", 
                           headers=headers, json=payload, timeout=20)
        r.raise_for_status()
        content = r.json()["content"][0]["text"]
        return content.strip()
    except Exception as e:
        print(f"AI总结失败: {e}")
        return ""


# ----------  新增翻译函数 ----------
def translate_papers_batch(papers):
    """
    使用 Claude 批量翻译论文标题和摘要。
    返回一个列表，每个元素是包含 'title_zh', 'abstract_zh' 的字典。
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("未配置 ANTHROPIC_API_KEY，跳过翻译。")
        return [{"title_zh": "", "abstract_zh": ""} for _ in papers]

    # 构建待翻译的内容，用序号分隔，方便解析
    items = []
    for i, p in enumerate(papers):
        title = p.get("title", "")
        abstract = (p.get("abstract") or "")[:500]  # 限制长度避免 token 爆炸
        items.append({
            "index": i,
            "title": title,
            "abstract": abstract
        })

    # 构造提示词，要求返回严格的 JSON 数组
    prompt = "请将以下风电领域论文的标题和摘要翻译成中文。\n"
    for item in items:
        prompt += f"\n[{item['index']}]\n标题: {item['title']}\n摘要: {item['abstract']}\n"
    prompt += "\n请返回一个 JSON 数组，每个元素包含 index, title_zh, abstract_zh，直接返回 JSON 不要加任何解释。\n示例：[{\"index\": 0, \"title_zh\": \"...\", \"abstract_zh\": \"...\"}, ...]"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 3000,  # 根据论文数量调整
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        r = requests.post("https://api.anthropic.com/v1/messages",
                           headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        content = r.json()["content"][0]["text"]
        # 提取 JSON 部分（可能被包裹在 ``` 里）
        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        translations = json.loads(content)
        # 按 index 排序后提取
        result = [{"title_zh": "", "abstract_zh": ""} for _ in papers]
        for t in translations:
            idx = t.get("index")
            if idx is not None and idx < len(result):
                result[idx]["title_zh"] = t.get("title_zh", "")
                result[idx]["abstract_zh"] = t.get("abstract_zh", "")
        return result
    except Exception as e:
        print(f"批量翻译失败: {e}")
        # 失败时返回空翻译，不影响主流程
        return [{"title_zh": "", "abstract_zh": ""} for _ in papers]
        
# ---------- 3. 格式化邮件内容 ----------
def build_email_body(papers, translations=None, use_ai_summary=False):
    lines = [f"📬 风电热点论文日报 - {datetime.now().strftime('%Y-%m-%d')}\n"]
    lines.append(f"共检索到 {len(papers)} 篇高引用论文\n")
    for i, p in enumerate(papers):
        title_en = p.get("title", "无标题")
        url = p.get("url", "")
        abstract_en = (p.get("abstract") or "无摘要")[:300].replace("\n", " ")
        citation = p.get("citationCount", 0)
        pub_date = p.get("publicationDate") or "未知"

        # 获取翻译内容
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

        # AI 一句话总结（如果启用）
        if use_ai_summary and abstract_en:
            summary = ai_summarize_paper(title_en, abstract_en)
            if summary:
                lines.append(f"🤖 AI一句话：{summary}")

        lines.append(f"📝 英文摘要：{abstract_en}...")
        if abstract_zh:
            lines.append(f"📝 中文摘要：{abstract_zh}...")
        lines.append("")
    return "\n".join(lines)
    

# ---------- 4. 发送邮件 ----------
def send_email(content, subject_prefix="风电热点论文"):
    sender = os.environ.get("EMAIL_SENDER")      # 发件人地址
    password = os.environ.get("EMAIL_PASSWORD")  # 授权码
    receiver = os.environ.get("EMAIL_RECEIVER")  # 收件人地址，一般同发件人

    if not all([sender, password, receiver]):
        print("邮件配置缺失，跳过发送。请检查 GitHub Secrets。")
        # 至少把内容打印到日志，方便查看
        print(content)
        return

    msg = MIMEText(content, "plain", "utf-8")
    msg["Subject"] = f"{subject_prefix} - {datetime.now().strftime('%Y-%m-%d')}"
    msg["From"] = sender
    msg["To"] = receiver

    # QQ邮箱SMTP，如果是其他邮箱请修改 host 和 port
    # 163: smtp.163.com, 465
    smtp_host = os.environ.get("SMTP_HOST", "smtp.qq.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 465))
    try:
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(sender, password)
            server.sendmail(sender, [receiver], msg.as_string())
        print("邮件发送成功！")
    except Exception as e:
        print(f"邮件发送失败: {e}")

# ---------- 5. 主流程 ----------
if __name__ == "__main__":
    query = '"wind energy" OR "wind turbine" OR "wind power"'
    use_ai = os.environ.get("USE_AI", "false").lower() == "true"
    translate_mode = os.environ.get("TRANSLATE_MODE", "false").lower() == "true"

    print(f"开始检索: {query}")
    papers = fetch_hot_papers(query, limit=10, year="2026")
    if not papers:
        send_email("今日未获取到风电热点论文，请检查脚本日志。", "风电论文日报-错误")
        sys.exit(0)

    # 翻译流程
    translations = None
    if translate_mode:
        print("正在进行批量翻译...")
        translations = translate_papers_batch(papers)

    body = build_email_body(papers, translations=translations, use_ai_summary=use_ai)
    send_email(body)
