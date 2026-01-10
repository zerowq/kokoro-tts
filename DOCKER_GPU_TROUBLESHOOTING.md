# Docker GPU 加速问题排查记录

**日期**: 2026-01-10  
**目标**: 在GPU服务器上运行Kokoro TTS Docker容器并启用GPU加速  
**硬件**: Tesla V100-SXM2-32GB x8, 驱动版本 535.274.02

---

## 问题1: Dockerfile COPY 命令语法错误

### 现象
```bash
docker build 失败
ERROR: inconsistent graph state in edge
```

### 原因
```dockerfile
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
```
语法错误：尝试同时复制两个文件/目录到一个路径

### 解决方案
```dockerfile
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY --from=ghcr.io/astral-sh/uv:latest /uvx /bin/uvx
```

---

## 问题2: 代码中的运行时错误

### 2.1 `start_time` 未定义
**位置**: `src/engines/kokoro_engine.py:77`  
**现象**: `NameError: name 'start_time' is not defined`  
**解决**: 在 `_load_model()` 方法开始添加 `start_time = time.time()`

### 2.2 `logger` 未导入
**位置**: `src/main.py:113`  
**现象**: `NameError: name 'logger' is not defined`  
**解决**: 添加 `from loguru import logger`

### 2.3 流式API调用错误
**位置**: `src/core/service.py:199`  
**现象**: `AttributeError: 'NoneType' object has no attribute 'create'`  
**原因**: 直接调用 `self.kokoro._kokoro.create()` 而没有先调用 `_load_model()`  
**解决**: 改为调用 `self.kokoro.synthesize()`

---

## 问题3: Docker构建时区交互提示

### 现象
构建过程卡在时区选择界面：
```
Please select the geographic area in which you live...
Geographic area:
```

### 解决方案
在Dockerfile中添加环境变量跳过交互：
```dockerfile
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC
```

并在apt-get命令前添加：
```dockerfile
RUN DEBIAN_FRONTEND=noninteractive TZ=UTC apt-get install ...
```

---

## 问题4: GPU加速核心问题 - CUDA版本匹配

### 问题链路

#### 尝试1: CUDA 12.2 + torch 2.4.0 + onnxruntime-gpu (默认)
**现象**: `Error 803: system has unsupported display driver / cuda driver combination`  
**原因**: torch的CUDA初始化与驱动不兼容

#### 尝试2: CUDA 11.8 + torch 2.1.0 (cu118)
**现象**: 同样报 `Error 803`  
**原因**: 驱动535需要CUDA 12.x才能正常工作

#### 尝试3: CUDA 12.2 + torch 2.1.0 (cu121) + onnxruntime-gpu 1.17.0 (PyPI)
**现象**: 
```
📊 [ONNX] Available Providers: ['AzureExecutionProvider', 'CPUExecutionProvider']
```
没有CUDAExecutionProvider

**原因**: PyPI的onnxruntime-gpu没有CUDA支持

#### 尝试4: CUDA 12.2 + torch 2.1.0 (cu121) + onnxruntime-gpu 1.17.0 (构建时验证通过)
**现象**: 
- 构建时：`✅ Installed providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', ...]`
- 运行时：`📊 [ONNX] Available Providers: ['AzureExecutionProvider', 'CPUExecutionProvider']`
- 版本不一致：构建安装1.17.0，运行时变成1.23.2

**原因**: `uv pip install kokoro-onnx` 时自动升级了onnxruntime

**解决**: 使用 `--no-deps` 安装kokoro-onnx

#### 尝试5: NumPy版本冲突
**现象**: 
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.3.5
AttributeError: _ARRAY_API not found
```

**原因**: onnxruntime-gpu 1.17.0用NumPy 1.x编译，但环境中是NumPy 2.x  
**解决**: 在安装onnxruntime前先降级NumPy到1.x

#### 尝试6: kokoro-onnx依赖缺失
**现象**: 
```
ModuleNotFoundError: No module named 'colorlog'
ModuleNotFoundError: No module named 'espeakng_loader'
```

**原因**: 使用 `--no-deps` 安装kokoro-onnx后缺少依赖  
**解决**: 手动安装 `colorlog espeakng-loader`

#### 尝试7: phonemizer 依赖缺失
**现象**: 
```
ModuleNotFoundError: No module named 'phonemizer'
```

**原因**: 使用 `--no-deps` 安装 `kokoro-onnx` 后，核心依赖 `phonemizer` 未被安装。此外，`phonemizer` 需要系统库 `espeak-ng`。

**解决**: 
1. 在 Dockerfile 中通过 `apt-get` 安装 `espeak-ng`。
2. 在 `pip install` 时手动添加 `phonemizer`。

#### 尝试8: joblib 依赖缺失
**现象**: 
```
ModuleNotFoundError: No module named 'joblib'
```

**原因**: `phonemizer` 依赖 `joblib`，在 `--no-deps` 安装模式下需要手动补齐。

**解决**: 在 `pip install` 时手动添加 `joblib`。

#### 尝试9: 陷入依赖地狱 (dlinfo, joblib...)
**现象**: 先后报错 `No module named 'joblib'`, `No module named 'dlinfo'`。

**原因**: `phonemizer` 有多层深度依赖。使用 `--no-deps` 手动安装极其容易遗漏。

**终极解决策略**: 
1. `phonemizer` 及其工具链**正常安装**（不带 `--no-deps`），让其自动补全所有零碎依赖。
2. 仅对 `kokoro-onnx` 保持 `--no-deps`，因为它的依赖项（onnxruntime, numpy）是我们重点保护和定制的对象。

#### 尝试10: Docker Hub 镜像拉取超时
**现象**: 
```
ERROR: failed to authorize: failed to fetch anonymous token: Get "https://auth.docker.io/token...": net/http: TLS handshake timeout
```

**原因**: 所在服务器无法直接访问 Docker Hub (docker.io)。

**解决**: 
1. 将 `FROM` 替换为 NVIDIA 官方源：`nvcr.io/nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04`。
2. 或者在宿主机 `/etc/docker/daemon.json` 配置镜像加速器。

#### 尝试11: TensorRT 库缺失与 CUDA 803 错误
**现象**: 
1. `Failed to load library libonnxruntime_providers_tensorrt.so with error: libnvinfer.so.10`
2. `CUDA failure 803: system has unsupported display driver / cuda driver combination`

**原因**: 
- ONNX Runtime 1.18+ 强行尝试加载 TensorRT 10，但镜像中没有。
- CUDA 12 镜像的 `compat` 目录与宿主机 535 驱动产生冲突导致 803 错误。

**终极解决策略 (架构降级保证稳定性)**: 
1. **基础镜像**: 降级至 `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`。
2. **依赖库**: 移除 `LD_LIBRARY_PATH` 中的 `compat` 路径。
3. **软件栈**: 使用 `torch cu118` + `onnxruntime-gpu 1.17.1`。
4. **运行代码**: 禁用 TensorRT Execution Provider，仅保留 `CUDAExecutionProvider`。

#### 尝试12: 配置文件路径错误 (voices.json vs voices-v1.0.bin)
**现象**: 
```
_pickle.UnpicklingError: Failed to interpret file '/app/models/kokoro/voices.json' as a pickle
```

**原因**: `kokoro-onnx` 库需要一个二进制的音色库文件（通常是 `.bin` 或 `.npy`），而配置文件 `src/config.py` 中错误地指向了文本格式的 `voices.json`。

**解决**: 在 `src/config.py` 中将 `KOKORO_VOICES` 修改为 `voices-v1.0.bin`。

#### 尝试13: EspeakWrapper 属性缺失 (导入顺序问题)
**现象**: 
```
AttributeError: type object 'EspeakWrapper' has no attribute 'set_data_path'
```

**原因**: `espeakng-loader` 必须在 `phonemizer` 之前导入才会生效。如果顺序反了，`EspeakWrapper` 就没有 `set_data_path` 补丁。此外，官方 `phonemizer` 包在某些环境下初始化较慢。

**解决**: 
1. 在代码中导入 `Kokoro` 之前先 `import espeakng_loader`。
2. 在 Dockerfile 中切换到 `phonemizer-fork` 以获得更好的兼容性。

#### 当前问题: 验证加速效果
**现象**: 
```
libcublas.so.11: cannot open shared object file: No such file or directory
libcublasLt.so.11: cannot open shared object file: No such file or directory
```

**原因**: 
- 容器内是CUDA 12.2（库文件是libcublas.so.12）
- onnxruntime-gpu 1.17.0是用CUDA 11编译的（需要libcublas.so.11）

**矛盾**: 
- 用CUDA 11.8镜像 → torch报Error 803（驱动不匹配）
- 用CUDA 12.2镜像 → onnxruntime报libcublas.so.11缺失

---

## 问题5: 关键经验教训

### 5.1 构建时vs运行时的区别
- **错误认知**: "构建时没有GPU所以检测不到CUDA"
- **正确理解**: `ort.get_available_providers()` 返回的是**编译时的能力**，不需要GPU存在
- 如果构建时显示有CUDAExecutionProvider，运行时也应该有（除非包被覆盖）

### 5.2 包版本被覆盖
- `uv pip install` 会自动解析依赖并升级包
- 需要用 `--no-deps` 或明确锁定版本防止覆盖

### 5.3 CUDA版本匹配三要素
1. **基础镜像CUDA版本** (如nvidia/cuda:12.2.2)
2. **torch编译的CUDA版本** (如torch cu121)
3. **onnxruntime-gpu编译的CUDA版本** (如onnxruntime-gpu for CUDA 11)

三者必须兼容，同时还要匹配**宿主机驱动版本**

---

## 当前解决方案尝试

### 策略
- 基础镜像: `nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04` (匹配驱动535)
- torch: `2.1.0+cu121`
- onnxruntime-gpu: 尝试 `1.18.0` (可能支持CUDA 12)

### Dockerfile关键配置
```dockerfile
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# NumPy先降级
RUN pip install --no-cache-dir "numpy<2.0.0"

# torch cu121
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 尝试CUDA 12的onnxruntime
RUN pip install --no-cache-dir onnxruntime-gpu==1.18.0 \
    --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ || \
    pip install --no-cache-dir onnxruntime-gpu==1.17.0

# kokoro-onnx用--no-deps防止覆盖
RUN pip install --no-cache-dir --no-deps kokoro-onnx colorlog espeakng-loader
```

---

## 下一步行动

1. **验证onnxruntime-gpu 1.18.0是否支持CUDA 12**
   - 如果构建通过且运行时能用GPU → 问题解决
   - 如果还报libcublas.so.11 → 1.18.0也是CUDA 11编译的

2. **如果1.18.0仍是CUDA 11编译**，备选方案：
   - 方案A: 在CUDA 12镜像中创建CUDA 11库的软链接
   - 方案B: 自己编译onnxruntime-gpu (CUDA 12版本)
   - 方案C: 接受CPU模式运行

3. **终极验证命令**
```bash
docker logs kokoro-tts-server | grep -E "ONNX Runtime version|Available Providers|libcublas"
```

---

## 参考资料

- [ONNX Runtime CUDA EP官方文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [NVIDIA CUDA兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/)
- 驱动535支持: CUDA 11.8 - 12.2

---

## 时间线

| 时间 | 问题 | 解决状态 |
|------|------|---------|
| 初始 | Dockerfile语法错误 | ✅ 已解决 |
| - | 代码bug (start_time, logger) | ✅ 已解决 |
| - | 时区交互提示 | ✅ 已解决 |
| - | torch Error 803 | ✅ 已解决 (用CUDA 12.2) |
| - | onnxruntime无CUDA支持 | ✅ 已解决 (用1.17.0+NumPy降级) |
| - | 包版本被覆盖 | ✅ 已解决 (--no-deps) |
| - | 依赖缺失 (colorlog/espeak-ng) | ✅ 已解决 |
| - | phonemizer 缺失 | ✅ 已解决 |
| - | joblib 缺失 | ✅ 已解决 |
| - | dlinfo 缺失/依赖地狱 | ✅ 已解决 (策略调整) |
| - | Docker Hub 拉取超时 | ✅ 已解决 (迁至nvcr.io) |
| - | TensorRT 缺失/803 错误 | ✅ 已解决 (架构降级) |
| - | 配置文件路径错误 | ✅ 已解决 |
| - | EspeakWrapper 属性缺失 | ✅ 已解决 (补丁导入顺序) |
| 当前 | 最终系统验证 | ⏳ 测试中 |
