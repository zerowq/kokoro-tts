# 流式播放延迟优化方案

## 当前问题
- GPU合成速度：0.5秒
- 实际播放延迟：2-3秒
- **瓶颈**：浏览器 `<audio>` 标签需要缓冲数据才触发 `canplay`

---

## 优化方案

### 方案1：减小分块大小 ⭐ **推荐（改动最小）**

**原理**：让第一个chunk更快完成，减少首包延迟

**修改位置**：`src/core/service.py` 第169-181行

```python
# 当前：120字符才分块
if len(chunk) > 120:

# 改为：40字符分块（首包更快）
if len(chunk) > 40:
```

**预期效果**：
- 首个chunk从0.266s → 约0.1s
- 浏览器更快收到数据，减少缓冲等待
- **总延迟从2-3秒 → 1-1.5秒**

---

### 方案2：使用MediaSource API ⭐⭐ **最佳效果**

**原理**：绕过浏览器缓冲，实时追加音频流

**修改位置**：`static/index.html` 第265-306行

**替换代码**：
```javascript
synthBtn.onclick = async () => {
    const text = textInput.value.trim();
    if (!text) return;

    const selectedOption = voiceSelect.options[voiceSelect.selectedIndex];
    const voice = voiceSelect.value;
    const lang = selectedOption.getAttribute('data-lang');
    const speed = speedInput.value;

    synthBtn.disabled = true;
    loader.style.display = 'block';
    statusField.innerText = "🚀 正在流式合成...";
    audioSection.style.display = 'none';

    const queryParams = new URLSearchParams({text, voice, lang, speed});
    const streamUrl = `/api/tts/stream?${queryParams.toString()}`;

    // 使用 MediaSource API
    const mediaSource = new MediaSource();
    audioPlayer.src = URL.createObjectURL(mediaSource);
    audioSection.style.display = 'block';

    mediaSource.addEventListener('sourceopen', async () => {
        const sourceBuffer = mediaSource.addSourceBuffer('audio/wav; codecs="1"');
        
        const response = await fetch(streamUrl);
        const reader = response.body.getReader();

        let firstChunk = true;
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;

            // 等待 buffer 空闲
            if (sourceBuffer.updating) {
                await new Promise(resolve => sourceBuffer.addEventListener('updateend', resolve, {once: true}));
            }

            sourceBuffer.appendBuffer(value);

            if (firstChunk) {
                audioPlayer.play();
                statusField.innerText = "✅ 正在播放...";
                synthBtn.disabled = false;
                loader.style.display = 'none';
                firstChunk = false;
            }
        }

        if (!sourceBuffer.updating) {
            mediaSource.endOfStream();
        }
    });

    audioPlayer.onerror = () => {
        statusField.innerText = "❌ 播放错误";
        synthBtn.disabled = false;
        loader.style.display = 'none';
    };
};
```

**预期效果**：
- 收到首个chunk后立即播放
- **总延迟 < 0.5秒**（接近合成速度）

---

### 方案3：Web Audio API ⭐⭐⭐ **极致性能（复杂）**

**原理**：直接解码PCM数据，无缓冲延迟

**复杂度**：需要重写整个播放逻辑，包括：
- 后端返回纯PCM数据（无WAV头）
- 前端手动解码和播放
- 实现播放控制（暂停、进度条等）

**不推荐**：改动太大，收益有限

---

## 推荐实施顺序

1. **先试方案1**（5分钟改完）：
   - 改一行代码（120→40）
   - 立即测试效果
   - 如果满意就完成

2. **如需更好效果，再试方案2**（30分钟）：
   - 替换前端JavaScript
   - 实现真正流式播放
   - 延迟接近合成速度

---

## 测试验证

修改后测试：
```bash
# 查看后端日志
docker logs -f kokoro-tts-server

# 浏览器打开开发者工具 Network 标签
# 观察音频流的时间线：
# - TTFB (Time To First Byte)：首包时间
# - Content Download：持续下载时间
```

**成功标志**：
- 用户点击后 < 1秒听到声音
- 后端日志显示chunk快速生成

