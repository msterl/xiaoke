import os
import re
import sys
import json
import time
import sqlite3
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict
import requests
import gradio as gr
from duckduckgo_search import DDGS
from git import Repo

# ===================== 配置 =====================
API_KEY = os.getenv("API_KEY", "")
API_HOST = os.getenv("API_HOST", "https://api.openai.com/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

TIMEOUT = 60
MAX_RETRY = 3
PROJECT_DIR = "agent_project"
MEMORY_FILE = "agent_memory.json"
AUTO_GIT = True

# ===================== 数据结构 =====================
@dataclass
class Task:
    tid: int
    agent: str
    filename: str
    desc: str
    done: bool = False
    code: str = ""
    error: str = ""

@dataclass
class Project:
    name: str
    tasks: List[Task] = field(default_factory=list)
    memory: List[str] = field(default_factory=list)

# ===================== 记忆模块 =====================
def load_mem():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "utf-8") as f:
            return json.load(f)
    return []

def save_mem(text):
    mem = load_mem()
    mem.append(f"[{time.ctime()}] {text}")
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(mem, f, indent=2, ensure_ascii=False)

# ===================== AI 调用 =====================
def ai(system: str, user: str) -> str:
    if not API_KEY:
        return "请设置 API_KEY 环境变量"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": 0.1
    }
    try:
        res = requests.post(API_HOST, headers=headers, json=payload, timeout=TIMEOUT)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"API 异常：{str(e)}"

# ===================== 联网搜索 =====================
def search_web(q: str, limit=3):
    try:
        r = DDGS().text(q, max_results=limit)
        return "\n".join([f"{i+1}. {x['title']}\n{x['body']}" for i, x in enumerate(r)])
    except:
        return "搜索不可用"

# ===================== Git 自动提交 =====================
def git_commit(path, msg):
    try:
        if not os.path.isdir(os.path.join(path, ".git")):
            Repo.init(path)
        repo = Repo(path)
        repo.git.add(".")
        repo.index.commit(msg)
        return "Git 提交成功"
    except Exception as e:
        return f"Git 失败：{e}"

# ===================== 1 架构智能体 =====================
class Architect:
    def split(self, prompt, mem):
        system = """
你是架构师，分配任务给智能体：
Architect, Coder, Frontend, DBA, Browser, DevOps, Debugger

输出格式：
编号,智能体,文件名,任务描述
不要多余内容。
"""
        user = f"记忆：{mem}\n需求：{prompt}"
        text = ai(system, user)
        tasks = []
        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 3)
            if len(parts) != 4:
                continue
            tid, agent, fn, desc = parts
            tasks.append(Task(int(tid), agent.strip(), fn.strip(), desc.strip()))
        return tasks

# ===================== 2 代码智能体 =====================
class Coder:
    def gen(self, task, ctx):
        s = "专业Python工程师，只输出完整可运行代码，无解释无markdown。"
        u = f"上下文：{ctx}\n任务：{task.desc}\n文件：{task.filename}"
        code = ai(s, u)
        return re.sub(r"```[\s\S]*?```", "", code).strip()

# ===================== 3 前端智能体 =====================
class Frontend:
    def gen(self, task):
        s = "生成美观现代的HTML+CSS+JS，可直接打开。"
        u = f"任务：{task.desc}"
        code = ai(s, u)
        return re.sub(r"```[\s\S]*?```", "", code).strip()

# ===================== 4 数据库智能体 =====================
class DBA:
    def gen(self, task):
        s = "生成SQLite建表语句，规范合理。"
        u = f"任务：{task.desc}"
        return ai(s, u)

# ===================== 5 浏览器自动化 =====================
class BrowserAgent:
    def gen(self, task):
        s = "使用playwright.sync_api，同步写法，稳定运行。"
        u = f"任务：{task.desc}"
        code = ai(s, u)
        return re.sub(r"```[\s\S]*?```", "", code).strip()

# ===================== 6 DevOps（Docker+依赖） =====================
class DevOps:
    def dockerfile(self):
        return """FROM python:slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python","coding_agent.py"]
"""

    def requirements(self, tasks):
        pkgs = {"requests","gradio","playwright","duckduckgo-search","gitpython"}
        return "\n".join(pkgs)

# ===================== 7 调试智能体 =====================
class Debugger:
    def fix(self, task, err):
        s = "根据报错修复代码，只输出完整代码。"
        u = f"任务：{task.desc}\n代码：{task.code}\n报错：{err}"
        code = ai(s, u)
        return re.sub(r"```[\s\S]*?```", "", code).strip()

# ===================== 8 执行模块 =====================
class Runner:
    def run(self, path):
        ext = path.split(".")[-1]
        try:
            if ext == "py":
                r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=30)
                return r.returncode == 0, r.stdout + r.stderr
            elif ext in ["html", "Dockerfile", "txt", "md"]:
                return True, "文件生成成功"
            elif ext == "sql":
                db = os.path.join(PROJECT_DIR, "data.db")
                con = sqlite3.connect(db)
                con.executescript(open(path, encoding="utf-8").read())
                con.commit()
                con.close()
                return True, "SQLite 表创建成功"
            else:
                return True, "已生成"
        except Exception as e:
            return False, str(e)

# ===================== 主调度 =====================
class MainAgent:
    def __init__(self):
        self.arch = Architect()
        self.coder = Coder()
        self.front = Frontend()
        self.dba = DBA()
        self.browser = BrowserAgent()
        self.devops = DevOps()
        self.debugger = Debugger()
        self.runner = Runner()

    def start(self, prompt):
        log = ["🚀 全能编程智能体 已启动"]
        yield "\n".join(log)
        log.append(f"📥 任务：{prompt}")
        save_mem(f"需求：{prompt}")
        yield "\n".join(log)

        log.append("🔍 正在联网搜索参考...")
        yield "\n".join(log)
        info = search_web(prompt)
        log.append(f"搜索结果摘要：{info[:150]}")
        yield "\n".join(log)

        log.append("📝 架构师正在分配任务...")
        yield "\n".join(log)
        mem = load_mem()
        tasks = self.arch.split(prompt, mem)
        if not tasks:
            log.append("❌ 任务拆分失败")
            yield "\n".join(log)
            return

        log.append(f"✅ 共 {len(tasks)} 个任务，多智能体协同")
        yield "\n".join(log)
        os.makedirs(PROJECT_DIR, exist_ok=True)
        project = Project(name=prompt[:20], tasks=tasks)

        for task in tasks:
            log.append(f"\n===== 任务 {task.tid} | {task.agent} | {task.filename} =====")
            log.append(f"说明：{task.desc}")
            yield "\n".join(log)

            ctx = "\n".join(f"{t.filename}\n{t.code}" for t in tasks if t.done)
            if task.agent == "Coder":
                task.code = self.coder.gen(task, ctx)
            elif task.agent == "Frontend":
                task.code = self.front.gen(task)
            elif task.agent == "DBA":
                task.code = self.dba.gen(task)
            elif task.agent == "Browser":
                task.code = self.browser.gen(task)
            elif task.agent == "DevOps":
                if "Dockerfile" in task.filename:
                    task.code = self.devops.dockerfile()
                elif "requirements" in task.filename:
                    task.code = self.devops.requirements(tasks)

            fpath = os.path.join(PROJECT_DIR, task.filename)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(task.code)

            ok, out = self.runner.run(fpath)
            log.append(f"运行：{'成功' if ok else '失败'}")
            yield "\n".join(log)

            retry = 0
            while not ok and retry < MAX_RETRY:
                log.append(f"🔧 调试中 {retry+1}")
                yield "\n".join(log)
                task.code = self.debugger.fix(task, out)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(task.code)
                ok, out = self.runner.run(fpath)
                retry += 1

            if ok:
                task.done = True
                log.append("✅ 完成")
            else:
                log.append("❌ 失败")
            log.append(out[:600])
            save_mem(f"{task.agent} {task.filename} {'成功' if ok else '失败'}")
            yield "\n".join(log)

        if AUTO_GIT:
            log.append("\n📤 自动提交 Git...")
            yield "\n".join(log)
            git_msg = git_commit(PROJECT_DIR, f"AI自动构建：{prompt[:30]}")
            log.append(git_msg)
            yield "\n".join(log)

        ok_cnt = sum(1 for t in tasks if t.done)
        log.append(f"\n🎉 项目完成：{ok_cnt}/{len(tasks)}")
        log.append(f"📂 项目路径：{os.path.abspath(PROJECT_DIR)}")
        log.append("✅ 已包含：AI + 前端 + 数据库 + 浏览器 + Docker + Git + 搜索 + 记忆")
        yield "\n".join(log)

# ===================== 网页界面 =====================
def web_ui():
    with gr.Blocks(title="全能编程智能体") as app:
        gr.Markdown("# 🚀 全能编程智能体 · 完整版")
        gr.Markdown("多智能体 | 前端 | 数据库 | 浏览器 | Docker | Git | 联网 | 记忆 | 自动调试")
        inp = gr.Textbox(label="输入你的项目需求", lines=2,
            placeholder="例如：做一个用户登录系统，带数据库、前端页面、自动截图、Docker部署")
        btn = gr.Button("▶️ 全自动执行", variant="primary")
        out = gr.Textbox(label="执行日志", lines=26)
        btn.click(MainAgent().start, inputs=inp, outputs=out)
    return app

# ===================== 启动 =====================
if __name__ == "__main__":
    web_ui().queue().launch(server_name="127.0.0.1", server_port=7860, share=False)
