# 项目设置说明

## 什么是 UV?

UV 是现代 Python 项目管理工具，用于替代 pip + virtualenv：
- 📦 自动创建虚拟环境
- 🔒 锁定依赖版本 (uv.lock)
- ⚡ 比 pip 快 10 倍
- 🎯 简单的命令行界面

## 安装 UV

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 或使用 Homebrew
brew install uv
```

## 快速开始

### 第一次设置

```bash
git clone <repo-url> kokoro-tts
cd kokoro-tts

# 1. 安装依赖 (自动创建虚拟环境)
uv sync

# 2. 下载模型文件
make download

# 3. 启动服务
make run
```

### 日常使用

```bash
# 查看可用命令
make help

# 运行测试
make test

# 添加新依赖
uv add <package-name>

# 删除依赖
uv remove <package-name>

# 更新所有依赖
uv sync --upgrade

# 在虚拟环境中运行命令
uv run python <script.py>
```

## 核心文件说明

| 文件 | 用途 |
|------|------|
| `pyproject.toml` | 项目配置和依赖定义 |
| `uv.lock` | 依赖版本锁定 (自动生成，上传到 Git) |
| `.python-version` | Python 版本指定 (3.11) |
| `Makefile` | 快捷命令集合 |
| `requirements.txt` | PIP 备用依赖列表 |

## 常见问题

### Q: 如何切换 Python 版本？
```bash
# 编辑 .python-version
echo "3.12" > .python-version
uv sync
```

### Q: 如何添加开发依赖？
```bash
uv add --group dev pytest
```

### Q: 为什么 models/ 在 Git 中？
模型文件在 `.gitignore` 中被忽略，太大无法上传。用脚本自动下载。

### Q: 如何共享依赖版本？
提交 `uv.lock` 到 Git，所有人都会用相同版本。

### Q: 虚拟环境在哪里？
`.venv/` 目录 (自动创建，不上传 Git)

## UV vs Pip

| 功能 | Pip | UV |
|------|-----|-----|
| 安装依赖 | ✓ | ✓ (更快) |
| 虚拟环境 | 需要 venv | 自动 |
| 版本锁定 | 需要 pip-tools | 自动 (uv.lock) |
| Python 版本管理 | ✗ | ✓ |
| 命令行速度 | 慢 | ⚡ |

## 更多资源

- [UV 官方文档](https://docs.astral.sh/uv/)
- [项目 README](README.md)
- [快速开始](QUICK_START.md)
