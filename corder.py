"""Corder — 中英文會議 / 電腦音訊逐字稿小工具。

- 轉錄音訊或影片檔（mp3 / m4a / wav / mp4 / mkv ...）
- 即時錄製電腦播放的聲音（線上會議、影片）、麥克風，或兩者混合並同步轉錄
- 匯出 TXT / SRT
"""

import os
import queue
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

APP_DIR = Path(__file__).resolve().parent
RECORD_DIR = APP_DIR / "recordings"
SAMPLE_RATE = 16000
BLOCK = SAMPLE_RATE // 2  # 每次讀取 0.5 秒


def _add_cuda_dll_dirs():
    """讓 ctranslate2 找得到 pip 安裝的 cuBLAS / cuDNN。"""
    base = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not base.is_dir():
        return
    for bin_dir in base.glob("*/bin"):
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


_add_cuda_dll_dirs()

MODELS = {
    "large-v3-turbo（推薦：快又準）": "large-v3-turbo",
    "large-v3（最準，較慢）": "large-v3",
    "medium": "medium",
    "small（最快，較不準）": "small",
}
LANGUAGES = {
    "中文（含中英夾雜）": "zh",
    "英文": "en",
    "自動偵測": None,
}
SOURCES = ["電腦音訊（會議、影片）", "麥克風", "電腦音訊 + 麥克風"]
ZH_PROMPT = "以下是繁體中文的會議逐字稿，內容可能夾雜 English 專有名詞。"


def fmt_ts(sec, srt=False):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}" if srt else f"{h:02d}:{m:02d}:{s:02d}"


class Engine:
    """包裝 faster-whisper：載入模型並轉錄。"""

    def __init__(self):
        self.model = None
        self.model_name = None
        self.device = None
        self.lock = threading.Lock()
        self._cc = None

    def load(self, name, status):
        if self.model is not None and self.model_name == name:
            return
        from faster_whisper import WhisperModel

        status(f"載入模型 {name}（第一次使用會自動下載，請稍候）…")
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() == 0:
                raise RuntimeError("no cuda")
            model = WhisperModel(name, device="cuda", compute_type="float16")
            device = "GPU"
        except Exception:
            model = WhisperModel(name, device="cpu", compute_type="int8")
            device = "CPU"
        self.model, self.model_name, self.device = model, name, device
        status(f"模型 {name} 已就緒（{device}）")

    def to_traditional(self, text):
        if self._cc is None:
            from opencc import OpenCC

            self._cc = OpenCC("s2twp")
        return self._cc.convert(text)

    def transcribe(self, audio, language, prompt=None):
        """回傳 (segments 產生器, info)。audio 可以是檔案路徑或 16kHz float32 陣列。"""
        if language == "zh":
            prompt = (ZH_PROMPT + (prompt or ""))[-200:]
        with self.lock:
            return self.model.transcribe(
                audio,
                language=language,
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                initial_prompt=prompt,
            )


class Recorder(threading.Thread):
    """從電腦音訊 (WASAPI loopback) / 麥克風 擷取聲音，存成 WAV 並切段送出轉錄。"""

    def __init__(self, source_idx, out_queue, wav_path):
        super().__init__(daemon=True)
        self.source_idx = source_idx
        self.out_queue = out_queue
        self.wav_path = wav_path
        self.stop_event = threading.Event()
        self.error = None
        self.level = 0.0

    def _open_devices(self):
        import soundcard as sc

        devs = []
        if self.source_idx in (0, 2):
            spk = sc.default_speaker()
            devs.append(sc.get_microphone(id=str(spk.name), include_loopback=True))
        if self.source_idx in (1, 2):
            devs.append(sc.default_microphone())
        return devs

    def run(self):
        import warnings

        warnings.filterwarnings("ignore", message=".*discontinuity.*")
        try:
            devs = self._open_devices()
            recs = [d.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=1024) for d in devs]
            for r in recs:
                r.__enter__()
            with wave.open(str(self.wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                while not self.stop_event.is_set():
                    mixed = np.zeros(BLOCK, dtype=np.float32)
                    for r in recs:
                        mixed += r.record(numframes=BLOCK)[:, 0]
                    mixed = np.clip(mixed, -1.0, 1.0)
                    self.level = float(np.sqrt(np.mean(mixed**2)))
                    wf.writeframes((mixed * 32767).astype(np.int16).tobytes())
                    self.out_queue.put(mixed)
            for r in recs:
                r.__exit__(None, None, None)
        except Exception as e:  # 裝置錯誤等
            self.error = e
        finally:
            self.out_queue.put(None)


class LiveTranscriber(threading.Thread):
    """把錄音串流在停頓處切段，逐段轉錄。"""

    MIN_SEC, MAX_SEC = 6, 20

    def __init__(self, engine, language, audio_queue, emit):
        super().__init__(daemon=True)
        self.engine, self.language = engine, language
        self.audio_queue, self.emit = audio_queue, emit
        self.offset = 0.0  # 目前緩衝區開頭在整段錄音中的秒數
        self.prev_text = ""

    def _cut_point(self, buf):
        n = len(buf)
        if n < self.MIN_SEC * SAMPLE_RATE:
            return 0
        frame = SAMPLE_RATE // 10
        rms = np.sqrt(np.mean(buf[: n - n % frame].reshape(-1, frame) ** 2, axis=1))
        quiet = max(0.003, 0.15 * float(np.median(rms)))
        if rms[-5:].max() < quiet:  # 最後 0.5 秒是停頓
            return n
        if n >= self.MAX_SEC * SAMPLE_RATE:  # 太長了，在最後 4 秒最安靜處切
            tail = rms[-40:]
            return (len(rms) - 40 + int(np.argmin(tail)) + 1) * frame
        return 0

    def _process(self, chunk):
        if np.sqrt(np.mean(chunk**2)) < 0.001:  # 完全沒聲音
            return
        segments, _ = self.engine.transcribe(chunk, self.language, self.prev_text)
        for seg in segments:
            text = seg.text.strip()
            if text:
                self.emit(self.offset + seg.start, self.offset + seg.end, text)
                self.prev_text = (self.prev_text + text)[-120:]

    def run(self):
        buf = np.zeros(0, dtype=np.float32)
        done = False
        while not done:
            block = self.audio_queue.get()
            if block is None:
                done = True
            else:
                buf = np.concatenate([buf, block])
            cut = len(buf) if done else self._cut_point(buf)
            if cut:
                chunk, buf = buf[:cut], buf[cut:]
                self._process(chunk)
                self.offset += cut / SAMPLE_RATE
        self.emit(None, None, None)  # 結束訊號


class App:
    def __init__(self, root):
        self.root = root
        self.engine = Engine()
        self.ui_queue = queue.Queue()
        self.segments = []  # [(start, end, text)]
        self.busy = False
        self.cancel = threading.Event()
        self.recorder = None
        self.last_recording = None
        self.record_started = 0.0

        root.title("Corder 逐字稿小幫手")
        root.geometry("860x640")
        root.minsize(640, 420)
        self._build_ui()
        root.after(100, self._poll)

    # ---------- UI ----------
    def _build_ui(self):
        style = ttk.Style()
        font = ("Microsoft JhengHei UI", 10)
        style.configure(".", font=font)
        style.configure("Big.TButton", font=("Microsoft JhengHei UI", 11, "bold"), padding=6)

        top = ttk.Frame(self.root, padding=(10, 10, 10, 4))
        top.pack(fill="x")
        ttk.Label(top, text="模型").pack(side="left")
        self.model_var = tk.StringVar(value=next(iter(MODELS)))
        ttk.Combobox(top, textvariable=self.model_var, values=list(MODELS), state="readonly", width=26).pack(
            side="left", padx=(4, 14)
        )
        ttk.Label(top, text="語言").pack(side="left")
        self.lang_var = tk.StringVar(value=next(iter(LANGUAGES)))
        ttk.Combobox(top, textvariable=self.lang_var, values=list(LANGUAGES), state="readonly", width=16).pack(
            side="left", padx=(4, 14)
        )
        self.trad_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="輸出繁體中文", variable=self.trad_var).pack(side="left")
        self.ts_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="顯示時間", variable=self.ts_var, command=self._rerender).pack(
            side="left", padx=10
        )

        actions = ttk.Frame(self.root, padding=(10, 4))
        actions.pack(fill="x")
        self.file_btn = ttk.Button(actions, text="📂 轉錄音訊/影片檔", style="Big.TButton", command=self.pick_file)
        self.file_btn.pack(side="left")
        ttk.Separator(actions, orient="vertical").pack(side="left", fill="y", padx=12)
        self.source_var = tk.StringVar(value=SOURCES[0])
        self.source_box = ttk.Combobox(
            actions, textvariable=self.source_var, values=SOURCES, state="readonly", width=20
        )
        self.source_box.pack(side="left")
        self.rec_btn = ttk.Button(actions, text="🔴 開始錄音轉錄", style="Big.TButton", command=self.toggle_record)
        self.rec_btn.pack(side="left", padx=6)
        self.level_bar = ttk.Progressbar(actions, length=80, maximum=0.2)
        self.level_bar.pack(side="left", padx=4)
        self.rec_time = ttk.Label(actions, text="")
        self.rec_time.pack(side="left", padx=4)

        body = ttk.Frame(self.root, padding=(10, 4))
        body.pack(fill="both", expand=True)
        self.text = tk.Text(body, wrap="word", font=("Microsoft JhengHei UI", 12), padx=10, pady=8, undo=True)
        sb = ttk.Scrollbar(body, command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        self.text.tag_configure("ts", foreground="#888888")

        bottom = ttk.Frame(self.root, padding=(10, 4, 10, 10))
        bottom.pack(fill="x")
        ttk.Button(bottom, text="複製全部", command=self.copy_all).pack(side="left")
        ttk.Button(bottom, text="存成 TXT", command=lambda: self.save("txt")).pack(side="left", padx=4)
        ttk.Button(bottom, text="存成 SRT 字幕", command=lambda: self.save("srt")).pack(side="left")
        ttk.Button(bottom, text="清除", command=self.clear).pack(side="left", padx=4)
        self.redo_btn = ttk.Button(bottom, text="🎯 用錄音檔重新精準轉錄", command=self.redo_recording)
        self.stop_btn = ttk.Button(bottom, text="停止", command=self.cancel.set)
        self.progress = ttk.Progressbar(bottom, length=160, maximum=100)
        self.progress.pack(side="right")

        self.status_var = tk.StringVar(value="就緒。選一個檔案，或開始錄音。")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w", padding=(8, 2)).pack(
            fill="x", side="bottom"
        )

    def _set_busy(self, busy):
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.file_btn.configure(state=state)
        self.redo_btn.configure(state=state)
        if not self.recorder:
            self.rec_btn.configure(state=state)
        if busy and not self.recorder:
            self.stop_btn.pack(side="right", padx=6)
        else:
            self.stop_btn.pack_forget()

    # ---------- 跨執行緒 → UI ----------
    def status(self, msg):
        self.ui_queue.put(("status", msg))

    def _emit_segment(self, start, end, text):
        self.ui_queue.put(("segment", (start, end, text)))

    def _poll(self):
        try:
            while True:
                kind, data = self.ui_queue.get_nowait()
                if kind == "status":
                    self.status_var.set(data)
                elif kind == "segment":
                    self._add_segment(*data)
                elif kind == "progress":
                    self.progress["value"] = data
                elif kind == "done":
                    self._set_busy(False)
                elif kind == "live_done":
                    self._live_finished()
                elif kind == "error":
                    messagebox.showerror("發生錯誤", data)
        except queue.Empty:
            pass
        if self.recorder:
            self.level_bar["value"] = self.recorder.level
            el = int(time.time() - self.record_started)
            self.rec_time.configure(text=f"{el // 60:02d}:{el % 60:02d}")
        self.root.after(100, self._poll)

    def _add_segment(self, start, end, text):
        if text is None:
            return
        if self.trad_var.get():
            text = self.engine.to_traditional(text)
        self.segments.append((start, end, text))
        self._render_one(start, text)
        self.text.see("end")

    def _render_one(self, start, text):
        if self.ts_var.get():
            self.text.insert("end", f"[{fmt_ts(start)}] ", "ts")
            self.text.insert("end", text + "\n")
        else:
            self.text.insert("end", text + ("\n" if text.endswith(("。", "？", "！", ".", "?", "!")) else " "))

    def _rerender(self):
        self.text.delete("1.0", "end")
        for start, _, text in self.segments:
            self._render_one(start, text)

    # ---------- 檔案轉錄 ----------
    def pick_file(self):
        path = filedialog.askopenfilename(
            title="選擇音訊或影片檔",
            filetypes=[
                ("音訊 / 影片", "*.mp3 *.m4a *.wav *.flac *.ogg *.aac *.wma *.opus *.mp4 *.mkv *.mov *.avi *.webm"),
                ("所有檔案", "*.*"),
            ],
        )
        if path:
            self.transcribe_file(path)

    def redo_recording(self):
        if self.last_recording and Path(self.last_recording).exists():
            self.transcribe_file(self.last_recording)
        else:
            messagebox.showinfo("沒有錄音檔", "目前沒有可重新轉錄的錄音。")

    def transcribe_file(self, path):
        if self.segments and not messagebox.askyesno("清除目前內容？", "將清除目前的逐字稿並開始新的轉錄，確定嗎？"):
            return
        self.clear()
        self.cancel.clear()
        self._set_busy(True)
        model, lang = MODELS[self.model_var.get()], LANGUAGES[self.lang_var.get()]
        threading.Thread(target=self._file_worker, args=(path, model, lang), daemon=True).start()

    def _file_worker(self, path, model, lang):
        try:
            self.engine.load(model, self.status)
            name = Path(path).name
            self.status(f"正在轉錄：{name} …")
            t0 = time.time()
            segments, info = self.engine.transcribe(path, lang)
            total = max(info.duration, 1)
            for seg in segments:
                if self.cancel.is_set():
                    self.status("已停止。")
                    break
                self._emit_segment(seg.start, seg.end, seg.text.strip())
                self.ui_queue.put(("progress", min(100, seg.end / total * 100)))
            else:
                self.ui_queue.put(("progress", 100))
                self.status(f"完成：{name}（音檔 {fmt_ts(total)}，耗時 {time.time() - t0:.0f} 秒，{self.engine.device}）")
        except Exception as e:
            self.status("轉錄失敗。")
            self.ui_queue.put(("error", f"{type(e).__name__}: {e}"))
        finally:
            self.ui_queue.put(("done", None))

    # ---------- 即時錄音 ----------
    def toggle_record(self):
        if self.recorder:
            self.recorder.stop_event.set()
            self.rec_btn.configure(state="disabled", text="收尾中…")
            self.status("停止錄音，正在轉錄最後一段…")
            return
        if self.segments and not messagebox.askyesno("清除目前內容？", "將清除目前的逐字稿並開始新的錄音，確定嗎？"):
            return
        self.clear()
        self._set_busy(True)
        self.rec_btn.configure(state="disabled")
        model, lang = MODELS[self.model_var.get()], LANGUAGES[self.lang_var.get()]
        src = SOURCES.index(self.source_var.get())
        threading.Thread(target=self._start_live, args=(model, lang, src), daemon=True).start()

    def _start_live(self, model, lang, src):
        try:
            self.engine.load(model, self.status)
        except Exception as e:
            self.ui_queue.put(("error", f"模型載入失敗：{e}"))
            self.ui_queue.put(("done", None))
            self.root.after(0, lambda: self.rec_btn.configure(state="normal"))
            return
        self.root.after(0, lambda: self._begin_recording(lang, src))

    def _begin_recording(self, lang, src):
        RECORD_DIR.mkdir(exist_ok=True)
        wav = RECORD_DIR / f"錄音_{datetime.now():%Y%m%d_%H%M%S}.wav"
        audio_q = queue.Queue()
        self.recorder = Recorder(src, audio_q, wav)
        self.live = LiveTranscriber(self.engine, lang, audio_q, self._live_emit)
        self.last_recording = wav
        self.record_started = time.time()
        self.recorder.start()
        self.live.start()
        self.rec_btn.configure(state="normal", text="⏹ 停止錄音")
        self.source_box.configure(state="disabled")
        self.status(f"錄音中（{SOURCES[src]}）… 說話停頓時會自動出字。錄音檔：{wav.name}")

    def _live_emit(self, start, end, text):
        if text is None:
            self.ui_queue.put(("live_done", None))
        else:
            self._emit_segment(start, end, text)

    def _live_finished(self):
        err = self.recorder.error if self.recorder else None
        self.recorder = None
        self.rec_btn.configure(state="normal", text="🔴 開始錄音轉錄")
        self.source_box.configure(state="readonly")
        self.level_bar["value"] = 0
        self._set_busy(False)
        self.redo_btn.pack(side="right", padx=6)
        if err:
            self.status("錄音失敗。")
            messagebox.showerror("錄音失敗", f"無法擷取音訊：{err}")
        else:
            self.status(f"錄音完成，已存到 {self.last_recording}。可按「用錄音檔重新精準轉錄」取得更完整的結果。")

    # ---------- 匯出 ----------
    def copy_all(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.text.get("1.0", "end").strip())
        self.status("已複製到剪貼簿。")

    def save(self, kind):
        if not self.segments:
            messagebox.showinfo("沒有內容", "目前還沒有逐字稿。")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=f".{kind}",
            initialfile=f"逐字稿_{datetime.now():%Y%m%d_%H%M}.{kind}",
            filetypes=[(kind.upper(), f"*.{kind}")],
        )
        if not path:
            return
        if kind == "srt":
            content = "\n".join(
                f"{i}\n{fmt_ts(s, True)} --> {fmt_ts(e, True)}\n{t}\n" for i, (s, e, t) in enumerate(self.segments, 1)
            )
        else:
            content = self.text.get("1.0", "end").strip() + "\n"
        Path(path).write_text(content, encoding="utf-8-sig")
        self.status(f"已儲存：{path}")

    def clear(self):
        self.segments.clear()
        self.text.delete("1.0", "end")
        self.progress["value"] = 0


def main():
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)  # 高解析度螢幕不模糊
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
