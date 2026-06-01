import requests
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import os
import sys

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

# ---------- 3. 格式化邮件内容 ----------
def build_email_body(papers, use_ai=False):
    lines = [f"📬 风电热点论文日报 - {datetime.now().strftime('%Y-%m-%d')}\n"]
    lines.append(f"共检索到 {len(papers)} 篇高引用论文\n")
    for i, p in enumerate(papers, 1):
        title = p.get("title", "无标题")
        url = p.get("url", "")
        abstract = (p.get("abstract") or "无摘要")[:200].replace("\n", " ")
        citation = p.get("citationCount", 0)
        pub_date = p.get("publicationDate") or "未知"
        lines.append(f"{'='*40}")
        lines.append(f"📌 第{i}篇 | 引用量：{citation} | 发表：{pub_date}")
        lines.append(f"📄 标题：{title}")
        lines.append(f"🔗 链接：{url}")
        # 如果启用AI总结且摘要存在
        if use_ai and abstract:
            summary = ai_summarize_paper(title, abstract)
            if summary:
                lines.append(f"🤖 AI一句话：{summary}")
        lines.append(f"📝 摘要：{abstract}...\n")
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
    # 关键词可根据需要调整
    query = '"wind energy" OR "wind turbine" OR "wind power"'
    # 是否启用AI总结（通过环境变量控制，默认不启用）
    use_ai = os.environ.get("USE_AI", "false").lower() == "true"
    
    print(f"开始检索: {query}")
    papers = fetch_hot_papers(query, limit=10, year="2026")  # 只获取2026年论文以体现热度
    if not papers:
        print("未检索到论文，可能是API限制或网络问题。")
        # 发送一个空报告邮件
        send_email("今日未获取到风电热点论文，请检查脚本日志。", "风电论文日报-错误")
        sys.exit(0)
    
    body = build_email_body(papers, use_ai=use_ai)
    send_email(body)
