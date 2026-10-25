# 独立模型连接、平台接入与部署

文本与图像连接分别持有自己的服务商、Base URL、模型和密钥。规划、参考检索、风格化及评审使用文本连接；生图与精修使用图像连接。请求运行时按任务隔离连接，不在两个连接之间隐式借用密钥。

## 配置优先级

每一组连接分别读取：`TEXT_*` / `IMAGE_*` 环境变量 > `configs/connections.local.yaml` > 原有 `configs/model_config.yaml` 中的新连接配置或旧式字段。

旧配置仅作为兼容回退；不会自动覆盖或迁移原文件。界面切换平台会清空该组密钥，避免将一个平台的密钥发往另一个平台。模型名称以对应账号实际可用模型为准。

可复制 `configs/connections.template.yaml` 为 `configs/connections.local.yaml`，或在界面点击保存。以下示例展示两组不同密钥，均为占位内容：

```yaml
connections:
  text:
    provider: newapi
    base_url: https://newapi.omgteam.me/v1
    api_key: "填写文本连接自己的 Key"
    model: "填写账号可用的多模态文本模型名称"
    image_api: images
  image:
    provider: newapi
    base_url: https://newapi.omgteam.me/v1
    api_key: "填写图像连接自己的 Key"
    model: "填写账号可用的图像模型名称"
    image_api: images
```

也可以将图像连接改为其他平台；两组连接完全不必使用相同地址或 Key。

环境变量包括 `TEXT_PROVIDER`、`TEXT_BASE_URL`、`TEXT_API_KEY`、`TEXT_MODEL`，以及对应的 `IMAGE_PROVIDER`、`IMAGE_BASE_URL`、`IMAGE_API_KEY`、`IMAGE_MODEL`、`IMAGE_API`。

## 平台与能力

平台预设提供地址和默认协议，不能代替账号授权或平台的具体模型能力。

| 平台值 | 默认 Base URL | 使用说明 |
| --- | --- | --- |
| `newapi` | `https://newapi.omgteam.me/v1` | Omgt / New API；按渠道选择 Images、聊天返图或 Gemini 协议 |
| `openai` | `https://api.openai.com/v1` | 文本聊天与 Images 接口；具体模型应支持相应用途 |
| `openrouter` | `https://openrouter.ai/api/v1` | 图像默认走聊天返图协议，需要选择具备图像输出能力的模型 |
| `siliconflow` | `https://api.siliconflow.cn/v1` | 图像请求适配 `image_size` 和 `images[]` 响应；参考图能力依具体模型而定 |
| `dashscope` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 百炼千问兼容图像接口，编辑采用 generations 的 JSON `image` 扩展字段 |
| `deepseek` | `https://api.deepseek.com/v1` | 仅文本；使用它规划时关闭参考检索、将评审轮次设为 0，另配图像连接 |
| `gemini` | `https://generativelanguage.googleapis.com` | Gemini 原生多模态与生图接口 |
| `evolink` | `https://api.evolink.ai` | 现有异步任务生图接口，轮询和下载不会因临时失败自动重提生图 |
| `custom` | 手动填写 | 自建或其他 OpenAI 兼容入口；支持独立文本、图片地址 |

如果某平台的图像服务使用其专有协议，而非本工具实现的 Images / Chat / Gemini / Evolink 接口，仅修改平台名称不能让该接口自动兼容。请使用该平台提供的兼容渠道，或为专有接口添加适配器。

## Omgt / New API 的具体设置

1. 分别在“文本模型连接”和“图像模型连接”选择 **Omgt / New API**。
2. Base URL 填写 `https://newapi.omgteam.me/v1`。填写根地址也会自动补全 `/v1`。
3. 两侧分别填写要使用的 API Key。可用同一个，也可以用权限、额度不同的两个 Key。
4. 分别点击“读取模型列表 / 检查连接”，从该 Key 的模型列表选择模型；读取列表需要有效 Key。
5. 使用参考图或评审时，文本侧应选择支持图像输入的多模态模型。
6. 图像侧默认选择 **OpenAI Images**。如果账号渠道明确通过聊天接口返回图片，选择 **聊天接口返图**；如需原生 `generateContent`，选择 **Gemini 原生**。

模型列表读取只使用只读请求，不产生本工具的生成调用。它不能证明模型的编辑、尺寸、分辨率权限均可用。通用 Images 路由对应 `/v1/images/generations`，参考图编辑对应 `/v1/images/edits`；渠道必须实际支持对应接口。

资料：[Omgt 服务](https://newapi.omgteam.me/)、[New API 官方图像接口文档](https://doc.newapi.pro/en/api/openai-image/)。

## 图像接口协议

| 配置值 | 协议 | 适用情况 |
| --- | --- | --- |
| `images` | Images（按平台适配） | OpenAI / New API 标准编辑使用 multipart；百炼、硅基流动采用对应 JSON 扩展字段 |
| `chat` | OpenAI Chat 图像输出 | `/chat/completions` 通过图片输出模型返回图像 |
| `gemini` | Gemini 原生 | `generateContent` 多模态协议，支持原图与文本输入 |
| `evolink` | Evolink 异步任务 | 提交任务、轮询同一任务、单独下载结果 |

上传参考图后，由图像连接对应的服务商处理该图片。图片格式和尺寸能力以模型实际响应为准；程序会显示实际图像像素。

平台参数依据：[硅基流动图像接口](https://docs.siliconflow.com/en/api-reference/images/images-generations)、[百炼千问图像接口](https://help.aliyun.com/zh/model-studio/qwen-image-generation-and-editing-api-reference)。百炼可将 Base URL 替换为账号的工作空间专属地址。

## 费用、失败与恢复

- 同一批相同输入的自动检索共享结果；空参考库跳过检索。随机参考模式保持候选独立采样。
- 默认每个候选仅生成实际使用的初图；只有显式勾选“额外生成规划阶段中间图”才额外生图。
- 生图请求上限控制调用次数，不是金额保证。文本输入、图片输入、输出 tokens、分辨率和服务商费率仍影响账单。
- 文本失败或输出无效时停止当前候选，不把 `Error` 当作图像提示词。其他候选继续执行。
- 每个候选结束后保存结果。取消停止后续调用，但无法撤销已经由服务端受理的计费请求。
- 新版历史结果可只重试失败或未完成候选。每次重试是新一轮调用，会按当前模型连接继续请求。

## 中文与交付格式

“图中文字语言”支持 `auto`、`zh`、`en`、`bilingual`；术语表可逐行写 `注意力机制 = Attention` 一类映射，传入规划、风格化、生成及评审阶段。文字准确率仍取决于模型，最终论文图应人工核对。

学术配图保留 PNG。统计图代码绘制可导出 PNG、SVG、PDF；代码执行默认关闭，启用后在独立子进程中运行并限制时间，仍具有当前用户的部分系统权限，不等同于安全沙箱。统计图中文字体来自本机字体列表，例如 Microsoft YaHei 或 Noto Sans CJK SC。

## 命令行

先配置本地连接文件或环境变量，再准备 `input.json`：

```json
{"content": "论文的方法内容", "visual_intent": "说明要画出的结构与关系"}
```

```bash
python main.py --input-file input.json --num-candidates 3 --max-concurrent 2 --max_critic_rounds 3 --retrieval_setting auto --max-image-calls 12 --output-language bilingual
```

可分别传入 `--text-provider`、`--image-provider`、`--text-base-url`、`--image-base-url`、`--text-model`、`--image-model` 和 `--image-api`。Key 使用本地配置或 `TEXT_API_KEY` / `IMAGE_API_KEY`，无需放入命令行参数。

统计图需要 `--task_name plot --allow-plot-code-execution`。关闭评审时使用 `--max_critic_rounds 0`；关闭参考库使用 `--retrieval_setting none`。

## Docker

`.dockerignore` 排除 `model_config.yaml`、`connections.local.yaml`、`.env`、`.venv`、`data`、`results` 等本地文件。镜像使用非 root 用户。模型配置在运行时提供，不在构建时复制。

```bash
docker build -t paperbanana-cn .
mkdir -p results data
docker run --rm -p 127.0.0.1:8080:8080 \
  -v "$(pwd)/configs/connections.local.yaml:/app/configs/connections.local.yaml:ro" \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/data:/app/data:ro" \
  paperbanana-cn
```

上例适用于 macOS/Linux shell。Windows PowerShell 可使用 Docker Desktop 的绝对路径挂载，例如 `--mount "type=bind,source=$($PWD.Path)\configs\connections.local.yaml,target=/app/configs/connections.local.yaml,readonly"`。

如需旧配置兼容回退，再将原 `configs/model_config.yaml` 以只读方式挂到容器同名位置。宿主机结果目录须允许容器用户写入。只读挂载的配置应在宿主机修改，容器界面中的保存按钮无法改写该文件。

也可以使用 `--env-file` 在运行时提供 `TEXT_*` / `IMAGE_*` 环境变量。访问 `http://127.0.0.1:8080`；桌面启动脚本的空闲端口选择不适用于固定映射的 Docker 端口。
