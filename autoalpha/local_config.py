"""
本地配置文件加载：让密钥与模型设置不必每次重启都重配一遍。

**为什么之前不落盘**：凭据不该进仓库或随配置被打包走，所以最初只在
进程内存里持有。代价是每次重启都要在页面上重填一遍 —— 这是真实的使用
摩擦，而不是安全需求本身。

**折中**：读 `.env`，但它是 gitignore 的，且永远不写进仓库。
安全边界从「绝不落盘」改成「落盘但不进版本控制、权限 600、优先环境变量」。

查找顺序（先找到的生效，不覆盖已存在的进程环境变量）：
    1. 进程环境变量            —— 用户 export 的永远优先，便于临时切换
    2. 项目根目录 `.env`        —— 每个项目一套
    3. `~/.autoalpha/.env`     —— 用户级默认，多个项目共用

支持 `KEY=VALUE`、`export KEY=VALUE`、引号包裹的值、`#` 注释与空行。
不引入 python-dotenv：格式简单，自己解析 20 行，省一个依赖。
"""

from __future__ import annotations

import os
import stat
from typing import Dict, List, Optional

# 允许从本地文件读取的键。白名单是为了避免配置文件意外覆盖掉
# PATH、PYTHONPATH 之类会改变进程行为的变量。
ALLOWED_KEYS = {
    "LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL",
    "LLM_PRICE_INPUT", "LLM_PRICE_OUTPUT", "LLM_MAX_TOKENS",
    "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "DEEPSEEK_API_KEY",
    "AUTOALPHA_KLINES", "AUTOALPHA_CONTACT_EMAIL",
}


def project_env_path() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, ".env")


def user_env_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".autoalpha", ".env")


def parse_env(text: str) -> Dict[str, str]:
    """解析 .env 文本。容错：坏行跳过而不是让整个文件失效。"""
    out: Dict[str, str] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if not key:
            continue
        # 去掉成对引号；不处理转义，密钥里一般也没有
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        # 行尾注释：仅在值未加引号且 # 前面有空白时才当作注释
        elif " #" in val:
            val = val.split(" #", 1)[0].rstrip()
        out[key] = val
    return out


def load_local_env(paths: Optional[List[str]] = None,
                   verbose: bool = False) -> List[str]:
    """
    把本地 .env 读进 os.environ，但不覆盖已有环境变量。
    返回实际生效的文件路径，供启动横幅显示「配置来自哪里」。
    """
    candidates = paths if paths is not None else [
        project_env_path(), user_env_path()]
    loaded: List[str] = []
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            print(f"[Config] 读取 {path} 失败: {e}")
            continue
        values = parse_env(text)
        applied = False
        for key, val in values.items():
            if key not in ALLOWED_KEYS:
                continue
            if key in os.environ:
                continue          # 进程环境变量优先
            os.environ[key] = val
            applied = True
        if applied:
            loaded.append(path)
            if verbose:
                print(f"[Config] 已加载本地配置 {path}")
    return loaded


def save_project_env(values: Dict[str, str],
                     path: Optional[str] = None) -> str:
    """
    把配置写进项目 .env（合并已有内容，不整文件覆盖）。

    只在用户明确要求保存时调用 —— 页面上改配置默认仍只写内存。
    文件权限设为 600：里面有密钥。
    """
    path = path or project_env_path()
    existing: Dict[str, str] = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            existing = parse_env(f.read())
    existing.update({k: v for k, v in values.items() if v})

    lines = [
        "# AutoAlpha 本地配置 —— 本文件已在 .gitignore 中，不会被提交。",
        "# 含密钥，权限已设为 600。进程环境变量优先于此文件。",
        "",
    ]
    for key in sorted(existing):
        val = existing[key]
        # 值含空格或 # 时加引号，否则解析回去会被截断
        if any(c in val for c in (" ", "#", "\t")):
            val = '"' + val.replace('"', '\\"') + '"'
        lines.append(f"{key}={val}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return path
