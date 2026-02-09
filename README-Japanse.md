# Qwen3-TTS 音声クローン WebUI

YouTubeのURL、マイクロフォン録音、またはアップロードされたオーディオファイルから音声をクローンし、クローンされた音声で任意のテキストを合成するWebUIアプリケーション。

## 使用方法 Youtube
<a href="https://www.youtube.com/watch?v=QbgzrLisIhQ">
  <img src="https://img.youtube.com/vi/QbgzrLisIhQ/0.jpg" width="480" alt="動画タイトル">
</a>

## 機能

- 指定されたタイムスタンプでYouTube URLから3秒のオーディオクリップを自動ダウンロード
- マイクロフォンからオーディオを録音
- 独自のオーディオファイルをアップロード（WAV、MP3、FLAC、OGG、OPUS、M4A、AAC、WMA、WebM）
- [pywhispercpp](https://github.com/absadiki/pywhispercpp)による自動転写（large-v3-turbo）
- [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)による音声クローン
- クローンプロファイルの保存と読み込み（LoRAスタイルの音声再現）
- [Gradio 6.x](https://www.gradio.app/)で構築されたWebUI

## 関連リンク

| コンポーネント | URL |
|---|---|
| Qwen3-TTS 公式ブログ | <https://qwen.ai/blog?id=qwen3tts-0115> |
| Qwen3-TTS GitHub | <https://github.com/QwenLM/Qwen3-TTS> |
| Qwen3-TTS-12Hz-1.7B-Base (HuggingFace) | <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base> |
| Qwen3-TTS-Tokenizer-12Hz (HuggingFace) | <https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz> |
| pywhispercpp (GitHub) | <https://github.com/absadiki/pywhispercpp> |
| whisper.cpp (GitHub) | <https://github.com/ggml-org/whisper.cpp> |
| yt-dlp (GitHub) | <https://github.com/yt-dlp/yt-dlp> |
| Gradio (公式サイト) | <https://www.gradio.app/> |
| uv パッケージマネージャー (ドキュメント) | <https://docs.astral.sh/uv/> |

## 要件

- Mac mini M4（Apple Silicon、24 GBユニファイドメモリ）
- macOS 15.7.2以降
- Python 3.11
- [uv](https://docs.astral.sh/uv/) パッケージマネージャー

## セットアップ

### 1. システム依存関係

```bash
brew install sox portaudio ffmpeg
```

### 2. プロジェクトセットアップ

```bash
# uvをまだインストールしていない場合はインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# プロジェクトディレクトリに移動
cd qwen-voice-clone-webui

# 依存関係をインストール
uv sync
```

### 3. モデルダウンロード（オプション：事前ダウンロード）

```bash
# モデルは初回実行時に自動ダウンロードされますが、事前にダウンロードすることも可能です：
uv run hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./model/Qwen3-TTS-12Hz-1.7B-Base
uv run hf download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ./model/Qwen3-TTS-Tokenizer-12Hz
```

ローカルモデルを使用する場合は、`src/voice_clone_webui/config.py`の`QWEN_TTS_MODEL_ID`をローカルパスを指すように更新してください。

### 4. 起動

```bash
uv run python -m voice_clone_webui.app
```

ブラウザで`http://localhost:7860`を開いてください。

## 使用方法

1. **オーディオを取得**: 参照オーディオを提供する3つの方法から1つを選択：
   - **YouTube URL**: 時間パラメータ付きのURLを貼り付けて「Fetch Audio」をクリック
   - **マイクロフォン**: 約3秒のオーディオを直接録音
   - **オーディオファイルのアップロード**: 独自のオーディオファイルをアップロードして「Process File」をクリック
2. **時間パラメータ付きURL**: YouTubeで動画を「共有したい開始時間」まで再生して一時停止。動画下の「共有」をクリック。共有ダイアログで「開始時間」オプションをチェック。時間パラメータ付きのURLがクリップボードにコピーされます。
3. **転写の確認**: 自動転写結果を確認し、必要に応じて編集
4. **プロファイルの保存**: 名前を入力して「Save Profile」をクリック — 次回はドロップダウンから読み込み可能
5. **音声生成**: テキストを入力して「Generate Speech」をクリック

## ディレクトリ構造

```
qwen-voice-clone-webui/
├── pyproject.toml
├── src/voice_clone_webui/   # アプリケーションソースコード
├── voice_profiles/          # 保存されたクローンプロファイル (.pt)
└── tmp/                     # 一時ファイル
```

---

## クイックセットアップ概要

```bash
# 1. プロジェクトディレクトリを作成
mkdir qwen-voice-clone-webui
cd qwen-voice-clone-webui

# 2. 上記のファイルをすべて配置し、依存関係をインストール
uv sync

# 3. 起動
uv run python -m voice_clone_webui.app
```

## 技術的注記

**Apple Silicon MPSサポート**: デフォルト設定では`dtype=torch.float32`と`attn_implementation="sdpa"`を使用しています。環境で`qwen-tts-demo`が既に`--dtype bfloat16`で動作している場合は、`config.py`の`TORCH_DTYPE_STR`を`"bfloat16"`に変更できます。

**プロファイル保存**: `create_voice_clone_prompt()`の戻り値は、参照オーディオのWAVデータと転写テキストと共に`torch.save`を介して`.pt`ファイルとして保存されます。これにより、プロファイルを読み込むだけで即座に音声クローン生成が可能になります。

**オーディオファイルのアップロード**: アップロードされたオーディオファイルは、転写と音声クローンの前にffmpegを使用して16kHzモノラルWAVに自動変換されます。サポートされる形式には、WAV、MP3、FLAC、OGG、OPUS、M4A、AAC、WMA、WebMが含まれます。アップロードファイルの長さ制限はありませんが、短いクリップ（明瞭な音声の約3-10秒）で最良の音声クローン結果が得られる傾向があります。

**Gradio 6.x互換性**: `gr.Blocks()`コンストラクタは`theme`/`css`引数なしで使用されます。これらは必要に応じて`launch()`に渡されます。`gr.Audio`などのコンポーネントは、最新のAPI仕様に従って`sources=["microphone"]`または`sources=["upload"]`を使用します。

**遅延初期化**: WhisperモデルとQwen TTSモデルの両方が起動時間を短縮するために初回使用時に読み込まれます。

## 免責事項

- このツールはYouTubeからオーディオを取得するために[yt-dlp](https://github.com/yt-dlp/yt-dlp)を使用します。YouTubeの利用規約では、サードパーティツールによるダウンロードが制限されています。自己責任で使用してください。
- 音声クローンは音声の所有者の同意がある場合、または個人的な実験や研究目的でのみ使用すべきです。
