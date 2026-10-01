# CLAUDE.md

本檔案提供 Claude Code（claude.ai/code）在此儲存庫中工作時的指引。

## 溝通語言

- 與使用者的所有溝通一律使用**繁體中文**。
- 程式碼、識別字、指令與既有英文註解維持原樣；新增的程式註解請沿用周遭程式碼的語言風格（目前專案皆為英文）。

## 專案概述

**TRELLIS.2**（Microsoft）是一個 4B 參數的大型 3D 生成模型，專注於高擬真度的**影像轉 3D（image-to-3D）**，並支援**依形狀生成 PBR 材質（texturing）**。核心是名為 **O-Voxel** 的「無場（field-free）」稀疏體素表示法，能處理開放曲面、非流形幾何與內部封閉結構，並帶有完整 PBR 屬性（Base Color、Metallic、Roughness、Alpha）。

- 論文：https://arxiv.org/abs/2512.14692
- 預訓練權重：Hugging Face `microsoft/TRELLIS.2-4B`
- 授權：MIT（nvdiffrast / nvdiffrec 另有各自授權）

## 環境需求與安裝

- **官方僅在 Linux 上測試**，需要 ≥24GB 顯存的 NVIDIA GPU（A100/H100 驗證過）；`setup.sh` 也支援 AMD ROCm（`hip`）。
- `o-voxel/third_party/eigen` 是 git submodule，目前**尚未初始化**；編譯 o-voxel 前須執行 `git submodule update --init --recursive`。

### 本機開發環境（Windows 11 + conda `ai_server`）

**所有開發、執行、安裝一律在 conda 環境 `ai_server` 中進行**，不要另建 `trellis2` 環境，也不要使用 `setup.sh --new-env`。

- 啟用：`conda activate ai_server`；在 Claude Code 的 Bash/PowerShell 工具中，直接呼叫 `C:\Users\ADMIN\miniconda3\envs\ai_server\python.exe`（`python -m pip ...` 亦同），避免 `conda run` 的編碼問題。
- 環境現況：Python 3.12、PyTorch 2.8.0 + CUDA 12.9（`torch.cuda.get_arch_list()` 含 `sm_120`）、transformers 4.56.1、gradio 5.44.1。
- 硬體與工具鏈：NVIDIA GeForce RTX 5090（Blackwell，compute capability 12.0，32GB）、CUDA Toolkit 12.9（`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`）、Visual Studio 2022 MSVC 14.44。
- `ai_server` 是**多個專案共用**的環境：升級或降級共用套件（torch、transformers、gradio 等）前要先詢問使用者。

Windows 不在官方支援範圍內，`setup.sh`（`getopt`、`sudo apt`、`/tmp`、需以 `source` 執行）無法使用，須手動安裝。與官方 Linux 流程的差異：

- **PyTorch 版本**：官方用 torch 2.6.0 + cu124，但 cu124 不支援 RTX 5090（sm_120），必須沿用 `ai_server` 的 torch 2.8.0 + cu129，**不要**照 `setup.sh` 降版。
- **注意力後端**：flash-attn 沒有官方 Windows wheel，在 Windows 上替 sm_120 編譯很困難 → 改裝 xformers（需與 torch 2.8.0/cu129 相容的版本），並設定 `ATTN_BACKEND=xformers`（稀疏注意力只支援 `xformers`/`flash_attn`/`flash_attn_3`，不能用 `sdpa`）。
- **稀疏卷積後端**：預設 `flex_gemm`（FlexGEMM）以 Triton 實作，Windows 需改裝 `triton-windows`（版本對應 torch 2.8）；是否能在 Windows 正常運作尚未驗證，不行時可改試 `SPARSE_CONV_BACKEND=spconv`。
- **CUDA 擴充（o-voxel、CuMesh、FlexGEMM、nvdiffrast v0.4.0、nvdiffrec renderutils 分支）**：在「x64 Native Tools Command Prompt for VS 2022」中、啟用 `ai_server` 後，設定 `CUDA_HOME` 指向 v12.9、`TORCH_CUDA_ARCH_LIST=12.0`，再以 `pip install <路徑> --no-build-isolation` 編譯。o-voxel 的 `-O3` 等 GCC 旗標在 MSVC 下僅會出現警告。
- **其他套件**：`pillow-simd` 與 `sudo apt install libjpeg-dev` 跳過，使用一般 `pillow`；其餘依 `setup.sh` 的 `--basic` 清單安裝（`utils3d` 須釘選相同 commit）。
- **gradio**：`app.py` 使用 gradio 6 的 `demo.launch(css=..., head=...)` 寫法（`setup.sh` 釘選 6.0.1），而 `ai_server` 為 5.44.1；升級前需徵得同意，否則需調整 `app.py`。
- **Hugging Face**：首次執行會下載 `microsoft/TRELLIS.2-4B`、DINOv3（`facebook/dinov3-*`，為需同意授權的 gated 模型，需先 `huggingface-cli login`）與 BiRefNet（`trust_remote_code=True`）。

### 官方 Linux 安裝方式（參考）

```sh
# 建立 conda 環境 trellis2（Python 3.10、PyTorch 2.6.0 + CUDA 12.4）並安裝所有相依套件
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
. ./setup.sh --help   # 查看各旗標
```

`setup.sh` 會把外部擴充 clone 到 `/tmp/extensions` 再以 `pip install --no-build-isolation` 安裝；`--o-voxel` 則是複製本地 `o-voxel/` 後安裝。**修改 `o-voxel/` 原始碼（含 `src/` 中的 C++/CUDA）後必須重新安裝**才會生效。

## 常用指令

本專案**沒有測試套件、linter 或建置系統**；CI 只有 CodeQL（`.github/workflows/codeql.yml`）。驗證變更的方式是執行範例或 `train.py --tryrun`。

```sh
python example.py              # 影像 → 3D，輸出 sample.mp4 與 sample.glb
python example_texturing.py    # 網格 + 參考圖 → PBR 材質，輸出 textured.glb
python app.py                  # Gradio 影像轉 3D 網頁 Demo（暫存檔放在 ./tmp/<session>）
python app_texturing.py        # Gradio 材質生成 Demo

# 訓練（--tryrun：只建構資料集/模型/trainer，不實際訓練，可用來檢查設定檔）
python train.py --config configs/gen/ss_flow_img_dit_1_3B_64_bf16.json \
  --output_dir results/ss_flow --data_dir '<JSON 字串或路徑>' [--tryrun]
```

`train.py` 其他參數：`--load_dir`（預設 = output_dir）、`--ckpt latest|none|<step>`、`--auto_retry`（預設 3，失敗時自動從最新 checkpoint 重試）、`--profile`、`--num_nodes`/`--node_rank`/`--num_gpus`/`--master_addr`/`--master_port`（多 GPU 以 `mp.spawn` 啟動）。

`--data_dir` 通常是 JSON 字串，鍵為資料集名稱，值為各類預處理資料的目錄（`base`、`mesh_dump`、`dual_grid`、`pbr_voxel`、`asset_stats`、`ss_latent`、`shape_latent`、`pbr_latent`、`render_cond` 等），需與所選 dataset 類別相符，完整範例見 `README.md`。

## 執行期環境變數

| 變數 | 用途 | 可選值（預設） |
| --- | --- | --- |
| `ATTN_BACKEND` | 稠密注意力後端（`trellis2/modules/attention/config.py`），也作為稀疏注意力的後備值 | `flash_attn`（預設）、`flash_attn_3`、`xformers`、`sdpa`、`naive` |
| `SPARSE_ATTN_BACKEND` | 稀疏注意力後端 | `flash_attn`（預設）、`flash_attn_3`、`xformers` |
| `SPARSE_CONV_BACKEND` | 稀疏卷積後端（`trellis2/modules/sparse/config.py`） | `flex_gemm`（預設）、`spconv`、`torchsparse`、`none` |
| `ATTN_DEBUG` / `SPARSE_DEBUG` | 設為 `1` 開啟除錯 | |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 範例與 app 中設定以節省顯存 | |
| `OPENCV_IO_ENABLE_OPENEXR=1` | 讀取 `assets/hdri/*.exr` 環境貼圖所需 | |

這些變數在**模組 import 時讀取**，必須在 `import trellis2` 之前設定。不支援 flash-attn 的 GPU（如 V100）請改用 `xformers`。

## 架構

### 三階段生成流程（`trellis2/pipelines/trellis2_image_to_3d.py`）

1. **Sparse Structure（SS）**：`SparseStructureFlowModel` 以 DINOv3 影像特徵為條件，在低解析度稠密網格（32 或 64）上生成佔據結構，再由 `SparseStructureDecoder` 解碼出稀疏體素座標。
2. **Shape SLat**：`SLatFlowModel` / `ElasticSLatFlowModel` 在稀疏體素上生成形狀結構化潛變數，由 `FlexiDualGridVaeDecoder` 解碼成網格（Flexible Dual Grid）。
3. **Texture SLat**：以形狀潛變數為條件生成 PBR 材質潛變數，由 `SparseUnetVaeDecoder` 解碼成體素屬性。

`pipeline.run(image, pipeline_type=...)` 支援 `'512'`、`'1024'`、`'1024_cascade'`（預設）、`'1536_cascade'`；cascade 會先在 512 生成形狀再上採樣到高解析度。輸出為 `MeshWithVoxel`（`trellis2/representations/mesh/base.py`），PBR 通道配置為 `base_color[0:3]`、`metallic[3:4]`、`roughness[4:5]`、`alpha[5:6]`，最後用 `o_voxel.postprocess.to_glb(...)` 進行重網格化、減面、UV 展開與材質烘焙輸出 GLB。

`Trellis2TexturingPipeline`（`trellis2_texturing.py`）只跑第 3 階段：輸入既有網格 + 參考影像，以 `texturing_pipeline.json` 載入。

預設 `low_vram=True`：`pipeline.cuda()` 只記錄裝置，各模型在使用時才搬上 GPU、用完移回 CPU。

### 以名稱字串驅動的延遲載入登錄表

`trellis2/models`、`datasets`、`trainers`、`pipelines`、`modules/sparse` 的 `__init__.py` 都使用同一模式：`__attributes` 字典（類別名 → 子模組）+ 模組層級 `__getattr__` 延遲 import。JSON 設定檔中的 `"name"` 就是透過 `getattr(models, name)` 等方式解析。

**新增模型/資料集/trainer/sampler 時，必須在對應 `__init__.py` 的 `__attributes` 中註冊**（並同步更新檔尾 `if __name__ == '__main__':` 區塊中給 Pylance 用的 import），否則設定檔無法找到該類別。

### 設定檔與 checkpoint

- `configs/scvae/*.json`：SC-VAE（形狀 `shape_vae_*`、材質 `tex_vae_*`）；`configs/gen/*.json`：flow 模型（`ss_flow_*`、`slat_flow_img2shape_*`、`slat_flow_imgshape2tex_*`）。`*_ft_512` / `*_ft1024` 為高解析度微調設定，使用前需修改其中的 `finetune_ckpt` 欄位。
- 設定檔結構：`models`（名稱 → `{name, args}`）、`dataset`（`{name, args}`）、`trainer`（`{name, args}`，含 optimizer、EMA、混合精度、grad clip、`t_schedule`、`image_cond_model` 等）。CLI 參數與 JSON 會合併成單一 `EasyDict`。
- `models.from_pretrained(path)` 讀取 `{path}.json` + `{path}.safetensors`（本地或 HF `repo/sub/path`）；`Pipeline.from_pretrained` 讀取 `pipeline.json`，其中 `args.models` 列出各子模型路徑，取樣器/正規化參數也在此。
- 訓練輸出：`<output_dir>/ckpts/{name}_step{NNNNNNN}.pt`、`{name}_ema{rate}_step*.pt`、`misc_step*.pt`（`--ckpt latest` 依此判斷最新步數）、`tb_logs/`（TensorBoard）、`command.txt`、`config.json`、`*_model_summary.txt`。

### 主要目錄

- `trellis2/modules/sparse/`：`SparseTensor` / `VarLenTensor`（`basic.py`）與稀疏版的 conv、attention（full / windowed / serialized + RoPE）、norm、spatial（上/下採樣、spatial↔channel）、transformer 區塊。conv 後端實作分別在 `conv_flex_gemm.py`、`conv_spconv.py`、`conv_torchsparse.py`。
- `trellis2/modules/attention/`、`modules/transformer/`：稠密版本，供 SS flow 使用。
- `trellis2/models/`：SS VAE / flow、SLat flow、`sc_vaes/`（`fdg_vae.py` 形狀、`sparse_unet_vae.py` 材質）、`sparse_elastic_mixin.py`（Elastic：訓練時依輸入 token 數，由 `utils/elastic_utils.py` 的 `MemoryController` 動態決定要對多少個 transformer block 啟用 gradient checkpointing）。
- `trellis2/trainers/`：`BasicTrainer`（DDP、EMA、混合精度、checkpoint、snapshot）→ `vae/` 與 `flow_matching/`（含 CFG、影像/文字條件 mixin）。
- `trellis2/pipelines/samplers/`：`FlowEulerSampler`、`FlowEulerCfgSampler`、`FlowEulerGuidanceIntervalSampler`；`rembg/BiRefNet.py` 去背（輸入若已有非全不透明的 alpha 通道則直接使用）。
- `trellis2/renderers/`：基於 nvdiffrast 的網格 / PBR 網格（`EnvMap`，split-sum 透過 nvdiffrec）/ 體素渲染器。
- `o-voxel/`：獨立的 pip 套件 `o_voxel`（Python + `src/` 中的 C++/CUDA 擴充）：網格 ↔ O-Voxel 轉換（`convert/`）、`.vxz` 壓縮格式與 Z-order / Hilbert 序列化（`io/`、`serialize.py`）、光柵化、`postprocess.to_glb`。範例在 `o-voxel/examples/`。
- `data_toolkit/`：訓練資料前處理流程（詳見 `data_toolkit/README.md`）：`build_metadata.py` → `download.py` → `dump_mesh.py` / `dump_pbr.py` / `asset_stats.py` → `dual_grid.py` / `voxelize_pbr.py`（O-Voxel 化，可接著訓練 SC-VAE）→ `encode_shape_latent.py` / `encode_pbr_latent.py` / `encode_ss_latent.py` → `render_cond.py`（訓練 flow 模型所需）。每個步驟後都要重跑 `build_metadata.py` 更新 metadata；大多數腳本支援 `--rank` / `--world_size` 分散處理；部分步驟透過 `blender_script/` 呼叫 Blender。

## 注意事項

- 輸出的 GLB 預設為 `OPAQUE` 模式，材質 alpha 通道雖保留但未啟用。
- 呼叫渲染前會先 `mesh.simplify(16777216)`，這是 nvdiffrast 的面數上限。
- `.gitignore` **沒有**排除 `results/`、`datasets/`、`tmp/` 及生成的 `.glb` / `.mp4`，提交前請確認不要把這些大型輸出檔加入版控。
