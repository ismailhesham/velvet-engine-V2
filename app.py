import os
import sys

os.environ["USE_TORCHVISION"] = "0"
sys.modules["torchvision"] = None

import re
import json
import uuid
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import soundfile as sf
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

from director import KineticDirector
from media_fetcher import KineticMediaFetcher


class MaskSecretsFilter(logging.Filter):
    """Prevents accidental API key leakage in application logs."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for env_key in ("DIRECTOR_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "PIXABAY_API_KEY", "PEXELS_API_KEY"):
            val = os.getenv(env_key, "")
            if val and len(val) > 6 and val in msg:
                record.msg = msg.replace(val, "***REDACTED***")
                record.args = ()
        return True


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger().addFilter(MaskSecretsFilter())
logger = logging.getLogger("velvet.kinetic")

VELVET_CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap');

:root, .dark {
    --body-background-fill: #070709 !important;
    --background-fill-primary: #0e0e13 !important;
    --background-fill-secondary: #14141b !important;
    --border-color-primary: rgba(212, 175, 55, 0.22) !important;
    --color-accent: #d4af37 !important;
    --body-text-color: #f4f0e8 !important;
    --block-title-text-color: #d4af37 !important;
}

body, .gradio-container {
    background: radial-gradient(circle at top center, #14131a 0%, #070709 65%) !important;
    font-family: 'Cairo', sans-serif !important;
    color: #f4f0e8 !important;
    max-width: 1360px !important;
    margin: 0 auto !important;
}

.velvet-header {
    text-align: center;
    padding: 24px 20px 18px 20px;
    margin-bottom: 18px;
    background: linear-gradient(180deg, rgba(212,175,55,0.08) 0%, rgba(14,14,19,0.4) 100%);
    border: 1px solid rgba(212, 175, 55, 0.28);
    border-radius: 16px;
}

.velvet-badge {
    display: inline-block;
    padding: 4px 14px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    color: #d4af37;
    border: 1px solid rgba(212, 175, 55, 0.45);
    border-radius: 999px;
    background: rgba(212, 175, 55, 0.08);
    margin-bottom: 8px;
}

.velvet-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 36px !important;
    font-weight: 700 !important;
    color: #f6f2eb !important;
    margin: 0 !important;
}

.velvet-card {
    background: rgba(16, 16, 22, 0.85) !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 14px !important;
    padding: 16px !important;
}

.script-stats-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(212, 175, 55, 0.06);
    border: 1px solid rgba(212, 175, 55, 0.2);
    border-radius: 10px;
    padding: 10px 18px;
    margin-top: 8px;
    font-size: 13.5px;
    color: #d8d2c7;
}

.script-stats-bar span b {
    color: #d4af37;
    font-size: 15px;
}

#produce-btn {
    background: linear-gradient(135deg, #e5c158 0%, #b88e22 100%) !important;
    color: #09090b !important;
    font-weight: 800 !important;
    font-size: 16px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 24px !important;
    margin-top: 12px !important;
    cursor: pointer !important;
}
"""


def update_script_stats(text: str) -> str:
    words = len(text.split()) if text and text.strip() else 0
    est_minutes = round(words / 145.0, 1)
    est_beats = max(1, round(words / 10.0)) if words > 0 else 0
    return f"""
    <div class="script-stats-bar">
        <span>📝 الكلمات: <b>{words:,}</b> كلمة</span>
        <span>⏱️ المدة المتوقعة: <b>~{est_minutes}</b> دقيقة</span>
        <span>🎞️ المشاهد الحركية المقدرة: <b>~{est_beats}</b> مشهد</span>
    </div>
    """


class AudioAndSFXEngine:
    def __init__(self, work_dir: str, voice: str = "bm_george", speed: float = 1.0):
        self.work_dir = work_dir
        self.audio_dir = os.path.join(work_dir, "audio_segments")
        os.makedirs(self.audio_dir, exist_ok=True)
        self.voice = voice
        self.speed = float(speed)
        self.sample_rate = 24000
        self.pipeline = None

    def _init_kokoro(self):
        if self.pipeline is not None:
            return
        from kokoro import KPipeline
        lang_code = "b" if self.voice.startswith("b") else "a"
        self.pipeline = KPipeline(lang_code=lang_code)

    def _synthesize_segment(self, text: str, out_wav: str) -> float:
        self._init_kokoro()
        clean_txt = re.sub(r"\s+", " ", (text or "").strip())
        if not clean_txt:
            silence = np.zeros(int(self.sample_rate * 1.5), dtype=np.float32)
            sf.write(out_wav, silence, self.sample_rate)
            return 1.5

        chunks: List[np.ndarray] = []
        for _, _, audio in self.pipeline(clean_txt, voice=self.voice, speed=self.speed, split_pattern=r"\n+"):
            if audio is not None and len(audio) > 0:
                chunks.append(np.asarray(audio, dtype=np.float32))

        if not chunks:
            chunks.append(np.zeros(int(self.sample_rate * 1.5), dtype=np.float32))

        pad = np.zeros(int(self.sample_rate * 0.14), dtype=np.float32)
        full_audio = np.concatenate(chunks + [pad])
        sf.write(out_wav, full_audio, self.sample_rate)
        return float(len(full_audio)) / float(self.sample_rate)

    def _align_whisperx(self, wav_path: str, scenes: List[Dict[str, Any]]) -> None:
        try:
            import torch
            import whisperx
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            model = whisperx.load_model("base.en", device, compute_type=compute_type)
            audio = whisperx.load_audio(wav_path)
            result = model.transcribe(audio, batch_size=16)
            model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
            aligned = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

            all_words: List[Dict[str, Any]] = []
            for seg in aligned.get("segments", []):
                for w in seg.get("words", []):
                    if "start" in w and "end" in w:
                        all_words.append({
                            "word": str(w.get("word") or "").strip(),
                            "start": float(w["start"]),
                            "end": float(w["end"])
                        })

            for s in scenes:
                s_start = float(s.get("start_time", 0.0))
                s_end = float(s.get("end_time", 0.0))
                s["word_timestamps"] = [w for w in all_words if (s_start - 0.15) <= w["start"] <= (s_end + 0.15)]
        except Exception as e:
            logger.warning("WhisperX alignment fallback used: %s", e)
            for s in scenes:
                s["word_timestamps"] = []

    def generate_and_master(self, timeline: Dict[str, Any], enable_music: bool = True, enable_sfx: bool = True) -> str:
        scenes = timeline.get("scenes") or []
        seg_arrays: List[np.ndarray] = []
        cursor = 0.0

        for s in scenes:
            sid = int(s.get("scene_id") or 1)
            seg_wav = os.path.join(self.audio_dir, f"seg_{sid:04d}.wav")
            spoken = s.get("spoken_text") or s.get("raw_text") or ""
            dur = self._synthesize_segment(spoken, seg_wav)
            data, _ = sf.read(seg_wav, dtype="float32")
            seg_arrays.append(data)

            s["start_time"] = round(cursor, 3)
            cursor += dur
            s["end_time"] = round(cursor, 3)
            s["duration"] = round(dur, 3)

        timeline["audio_master"] = {"total_duration_sec": round(cursor, 2)}
        voice_only = np.concatenate(seg_arrays) if seg_arrays else np.zeros(self.sample_rate * 2, dtype=np.float32)
        voice_wav = os.path.join(self.work_dir, "voice_master.wav")
        sf.write(voice_wav, voice_only, self.sample_rate)

        self._align_whisperx(voice_wav, scenes)

        master_audio = voice_only.copy()
        if enable_music and len(master_audio) > 0:
            t = np.arange(len(master_audio), dtype=np.float32) / float(self.sample_rate)
            drone = (
                0.016 * np.sin(2 * np.pi * 55.0 * t)
                + 0.010 * np.sin(2 * np.pi * 82.41 * t)
                + 0.006 * np.sin(2 * np.pi * 110.0 * t)
            ).astype(np.float32)
            master_audio += drone

        if enable_sfx and len(master_audio) > 0:
            rng = np.random.default_rng(42)
            for idx, s in enumerate(scenes):
                start_idx = int(float(s.get("start_time", 0.0)) * self.sample_rate)
                t_type = s.get("template_type", "KINETIC_TEXT")
                if t_type in ("NUMBER_COUNTER", "DOCUMENT_CARD") and start_idx < len(master_audio):
                    boom_len = min(int(0.55 * self.sample_rate), len(master_audio) - start_idx)
                    bt = np.arange(boom_len, dtype=np.float32) / float(self.sample_rate)
                    freq_env = 85.0 * np.exp(-4.5 * bt) + 36.0
                    amp_env = 0.14 * np.exp(-5.0 * bt)
                    boom = amp_env * np.sin(2 * np.pi * freq_env * bt)
                    master_audio[start_idx:start_idx + boom_len] += boom.astype(np.float32)
                elif idx > 0 and t_type == "FRAMED_IMAGE" and start_idx < len(master_audio):
                    w_len = min(int(0.24 * self.sample_rate), len(master_audio) - start_idx)
                    wt = np.arange(w_len, dtype=np.float32) / float(self.sample_rate)
                    env = np.sin(np.pi * (wt / 0.24)) ** 2 * 0.035
                    noise = rng.normal(0, 1, w_len).astype(np.float32)
                    master_audio[start_idx:start_idx + w_len] += (noise * env)

        peak = float(np.max(np.abs(master_audio))) if len(master_audio) > 0 else 1.0
        if peak > 0.95:
            master_audio = (master_audio / peak) * 0.92

        final_wav = os.path.join(self.work_dir, "final_mix.wav")
        sf.write(final_wav, master_audio, self.sample_rate)
        return final_wav


class KineticRenderer:
    def __init__(
        self,
        work_dir: str,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        style_preset: str = "Investigative Essay",
        enable_captions: bool = False,
        enable_source_labels: bool = True
    ):
        self.work_dir = work_dir
        self.renders_dir = os.path.join(work_dir, "scene_renders")
        os.makedirs(self.renders_dir, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.enable_captions = enable_captions
        self.enable_source_labels = enable_source_labels

        palette_map = {
            "Investigative Essay": (212, 175, 55),
            "Archival Historical": (230, 140, 60),
            "Natural Environment": (16, 185, 129),
        }
        self.accent = palette_map.get(style_preset, (212, 175, 55))
        self.bg_color = (12, 13, 17)
        self.cream = (248, 249, 250)
        self.dim_text = (75, 79, 88)
        self.muted = (158, 164, 176)

        self._font_cache: Dict[Tuple[int, bool], ImageFont.FreeTypeFont] = {}
        self._bg_cache: Optional[Image.Image] = None
        self.vcodec = self._detect_encoder()

    def _detect_encoder(self) -> str:
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1", "-c:v", "h264_nvenc", "-f", "null", "-"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return "h264_nvenc"
        except Exception:
            return "libx264"

    def _get_font(self, size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
        key = (size, bold)
        if key in self._font_cache:
            return self._font_cache[key]
        scaled = max(14, int(round(size * (self.height / 1080.0))))
        paths = [
            "./assets/fonts/Inter-Bold.ttf" if bold else "./assets/fonts/Inter-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for p in paths:
            if os.path.exists(p):
                try:
                    font = ImageFont.truetype(p, scaled)
                    self._font_cache[key] = font
                    return font
                except Exception:
                    continue
        font = ImageFont.load_default()
        self._font_cache[key] = font
        return font

    def _wrap_lines(self, text: str, font: ImageFont.FreeTypeFont, max_w: int, max_lines: int = 4) -> List[str]:
        words = (text or "").split()
        dummy = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        lines, curr = [], []
        for w in words:
            trial = " ".join(curr + [w])
            if (dummy.textbbox((0, 0), trial, font=font)[2]) <= max_w or not curr:
                curr.append(w)
            else:
                lines.append(" ".join(curr))
                curr = [w]
                if len(lines) >= max_lines:
                    break
        if curr and len(lines) < max_lines:
            lines.append(" ".join(curr))
        return lines

    def _static_bg(self) -> Image.Image:
        if self._bg_cache is not None:
            return self._bg_cache.copy()
        base = Image.new("RGB", (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(base)
        step = max(60, int(self.width * 0.045))
        grid_col = (20, 22, 29)
        for gx in range(0, self.width, step):
            draw.line([gx, 0, gx, self.height], fill=grid_col, width=1)
        for gy in range(0, self.height, step):
            draw.line([0, gy, self.width, gy], fill=grid_col, width=1)
        self._bg_cache = base
        return self._bg_cache.copy()

    def _draw_glow_text(self, draw: ImageDraw.ImageDraw, pos: Tuple[int, int], text: str, font: ImageFont.FreeTypeFont, fill: Tuple[int, int, int], anchor: str = "mm"):
        x, y = pos
        draw.text((x + 3, y + 4), text, font=font, fill=(4, 5, 8), anchor=anchor)
        draw.text((x, y), text, font=font, fill=fill, anchor=anchor)

    def _get_lit_count(self, scene: Dict[str, Any], rel_t: float, duration: float) -> int:
        progress = min(1.0, max(0.0, rel_t / max(0.1, duration)))
        abs_time = float(scene.get("start_time", 0.0)) + rel_t
        word_ts = scene.get("word_timestamps") or []
        words = (str(scene.get("raw_text") or "")).split() or ["..."]
        if word_ts:
            lit = 0
            for idx, w_info in enumerate(word_ts):
                if abs_time >= float(w_info.get("start", 0.0)) - 0.04:
                    lit = idx + 1
            return lit
        return max(1, int(len(words) * min(1.0, progress * 1.25)))

    def _render_frame(self, scene: Dict[str, Any], rel_t: float, duration: float, lit_override: Optional[int] = None) -> Image.Image:
        t_type = scene.get("template_type", "KINETIC_TEXT")
        t_data = scene.get("template_data") or {}

        canvas = self._static_bg()
        draw = ImageDraw.Draw(canvas)
        cx, cy = self.width // 2, self.height // 2
        safe_w = int(self.width * 0.82)

        if t_type == "NUMBER_COUNTER":
            raw_num = re.sub(r"[^0-9.\-]", "", str(t_data.get("target_number") or "100"))
            target = float(raw_num) if raw_num else 100.0
            prefix = str(t_data.get("prefix") or "")
            suffix = str(t_data.get("suffix") or "")
            label = str(t_data.get("label") or "KEY FIGURE").upper()

            c_prog = min(1.0, rel_t / max(0.12, min(1.0, duration * 0.45)))
            c_eased = 1.0 - pow(1.0 - c_prog, 3)
            start_v = target * 0.85 if target > 15 else 0.0
            curr = start_v + (target - start_v) * c_eased
            curr_val = int(round(curr)) if target >= 10 else round(curr, 1)

            num_str = f"{prefix}{curr_val:,}{suffix}"
            f_num = self._get_font(148, bold=True)
            f_lbl = self._get_font(26, bold=True)
            f_desc = self._get_font(25, bold=False)

            self._draw_glow_text(draw, (cx, cy - int(self.height * 0.06)), num_str, f_num, self.accent)
            bar_w = int(110 * c_eased)
            draw.rectangle([cx - bar_w, cy + int(self.height * 0.05), cx + bar_w, cy + int(self.height * 0.055)], fill=self.accent)
            draw.text((cx, cy + int(self.height * 0.10)), label, fill=self.cream, font=f_lbl, anchor="mm")
            for i, line in enumerate(self._wrap_lines(str(scene.get("raw_text") or ""), f_desc, safe_w, 2)):
                draw.text((cx, cy + int(self.height * 0.16) + i * 36), line, fill=self.muted, font=f_desc, anchor="mm")

        elif t_type == "DOCUMENT_CARD":
            hdr = str(t_data.get("document_header") or "EVIDENCE // RECORD").upper()
            body = str(t_data.get("document_body") or scene.get("raw_text") or "")
            hl = str(t_data.get("highlight_phrase") or "").lower()

            cw, ch = int(self.width * 0.68), int(self.height * 0.54)
            x0, y0 = cx - cw // 2, cy - ch // 2
            x1, y1 = x0 + cw, y0 + ch

            draw.rectangle([x0 + 14, y0 + 16, x1 + 14, y1 + 16], fill=(5, 6, 8))
            draw.rectangle([x0, y0, x1, y1], fill=(244, 241, 234))
            draw.rectangle([x0, y0, x1, y0 + 54], fill=(22, 25, 33))
            draw.rectangle([x0, y0, x0 + 8, y1], fill=self.accent)

            f_hdr = self._get_font(19, bold=True)
            f_body = self._get_font(32, bold=True)
            draw.text((x0 + 30, y0 + 27), hdr[:58], fill=self.accent, font=f_hdr, anchor="lm")

            lines = self._wrap_lines(body, f_body, cw - 84, 5)
            hl_prog = min(1.0, max(0.0, (rel_t - 0.15) / 0.55))
            for idx, line in enumerate(lines):
                ly = y0 + 100 + idx * int(50 * (self.height / 1080.0))
                is_hl = bool(hl and any(tok in line.lower() for tok in hl.split()[:3])) or (idx == len(lines) // 2)
                if is_hl and hl_prog > 0.05:
                    bbox = draw.textbbox((x0 + 40, ly), line, font=f_body)
                    bw = int((bbox[2] - bbox[0] + 14) * hl_prog)
                    draw.rectangle([x0 + 34, ly - 4, x0 + 34 + bw, bbox[3] + 8], fill=self.accent)
                draw.text((x0 + 40, ly), line, fill=(18, 20, 26), font=f_body)

        else:
            raw_text = str(scene.get("raw_text") or "")
            words = raw_text.split() or ["..."]
            hl_raw = t_data.get("highlight_words") or []
            if isinstance(hl_raw, str):
                hl_raw = hl_raw.split()
            hl_tokens = {re.sub(r"[^a-z0-9]", "", w.lower()) for w in hl_raw if len(w) > 2}

            lit_count = lit_override if lit_override is not None else self._get_lit_count(scene, rel_t, duration)

            kicker = str(t_data.get("kicker") or "").upper().strip()
            f_size = 56 if len(words) <= 10 else (46 if len(words) <= 18 else 38)
            f_word = self._get_font(f_size, bold=True)
            f_kick = self._get_font(19, bold=True)

            per_line = 6 if len(words) <= 12 else 7
            lines = [words[i:i + per_line] for i in range(0, len(words), per_line)][:4]
            spacing = int((f_size + 26) * (self.height / 1080.0))
            start_y = cy - ((len(lines) - 1) * spacing) // 2

            if kicker and self.enable_source_labels:
                draw.text((cx, start_y - int(65 * (self.height / 1080.0))), kicker[:44], fill=self.accent, font=f_kick, anchor="mm")

            g_idx = 0
            for l_idx, l_words in enumerate(lines):
                total_w = draw.textbbox((0, 0), " ".join(l_words), font=f_word)[2]
                curr_x = cx - total_w // 2
                ly = start_y + l_idx * spacing
                for w in l_words:
                    spoken = g_idx < lit_count
                    clean_w = re.sub(r"[^a-z0-9]", "", w.lower())
                    is_key = (clean_w in hl_tokens) or any(ch.isdigit() for ch in w)
                    ww = draw.textbbox((0, 0), w + " ", font=f_word)[2]
                    if spoken:
                        col = self.accent if is_key else self.cream
                        self._draw_glow_text(draw, (curr_x, ly), w, f_word, col, anchor="lm")
                    else:
                        draw.text((curr_x, ly), w, fill=self.dim_text, font=f_word, anchor="lm")
                    curr_x += ww
                    g_idx += 1

        if self.enable_captions and t_type != "KINETIC_TEXT":
            sub_font = self._get_font(22, bold=True)
            sub_txt = str(scene.get("raw_text") or "")[:95]
            draw.text((cx, int(self.height * 0.92)), sub_txt, fill=self.cream, font=sub_font, anchor="mm")

        return canvas

    def _encode_flags(self) -> List[str]:
        preset = "p1" if self.vcodec == "h264_nvenc" else "ultrafast"
        return ["-c:v", self.vcodec, "-preset", preset, "-pix_fmt", "yuv420p", "-r", str(self.fps), "-an"]

    def render_scene(self, scene: Dict[str, Any]) -> str:
        sid = int(scene["scene_id"])
        dur = max(1.2, float(scene.get("duration", 3.0)))
        t_type = scene.get("template_type", "KINETIC_TEXT")
        t_data = scene.get("template_data") or {}
        media_path = scene.get("local_media_path")
        out_clip = os.path.join(self.renders_dir, f"clip_{sid:04d}.mp4")

        if media_path and os.path.exists(media_path):
            try:
                bg = self._static_bg()
                fg = Image.open(media_path).convert("RGB")
                fg.thumbnail((int(self.width * 0.66), int(self.height * 0.62)), Image.Resampling.LANCZOS)

                pad = 10
                cap_h = 52 if self.enable_source_labels else 10
                fw, fh = fg.width + pad * 2, fg.height + pad * 2 + cap_h
                frame_img = Image.new("RGB", (fw, fh), (20, 23, 30))
                df = ImageDraw.Draw(frame_img)
                df.rectangle([0, 0, fw - 1, fh - 1], outline=self.accent, width=2)
                frame_img.paste(fg, (pad, pad))

                if self.enable_source_labels:
                    cap_txt = str(t_data.get("caption") or scene.get("visual_goal") or "VISUAL EVIDENCE").upper()[:46]
                    df.text((fw // 2, fg.height + pad + cap_h // 2), cap_txt, fill=self.accent, font=self._get_font(20, True), anchor="mm")

                bg.paste(frame_img, ((self.width - fw) // 2, (self.height - fh) // 2))
                framed_jpg = os.path.join(self.renders_dir, f"framed_{sid:04d}.jpg")
                bg.save(framed_jpg, quality=92)

                frames = max(1, int(dur * self.fps))
                up_w = int(self.width * 1.12)
                vf = f"scale={up_w}:-1,zoompan=z='min(zoom+0.0007,1.10)':d={frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={self.width}x{self.height}:fps={self.fps},setsar=1"
                subprocess.run(["ffmpeg", "-y", "-i", framed_jpg, "-t", f"{dur:.3f}", "-vf", vf] + self._encode_flags() + [out_clip], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return out_clip
            except Exception as e:
                logger.warning("Fallback to kinetic text for scene %d: %s", sid, e)

        frames = max(1, int(dur * self.fps))
        cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{self.width}x{self.height}", "-pix_fmt", "rgb24", "-r", str(self.fps), "-i", "-"] + self._encode_flags() + [out_clip]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        state_cache: Dict[int, bytes] = {}
        for f_idx in range(frames):
            rel_t = f_idx / float(self.fps)
            if t_type == "KINETIC_TEXT":
                lit = self._get_lit_count(scene, rel_t, dur)
                if lit not in state_cache:
                    state_cache[lit] = self._render_frame(scene, rel_t=rel_t, duration=dur, lit_override=lit).tobytes()
                proc.stdin.write(state_cache[lit])
            else:
                step_key = int(f_idx // 2)
                if step_key not in state_cache:
                    state_cache[step_key] = self._render_frame(scene, rel_t=rel_t, duration=dur).tobytes()
                proc.stdin.write(state_cache[step_key])

        proc.stdin.close()
        proc.wait()
        return out_clip


def run_velvet_pipeline(
    script_text: str,
    provider: str,
    model_id: str,
    llm_api_key: str,
    pexels_key: str,
    pixabay_key: str,
    voice_name: str,
    render_mode: str,
    style_preset: str,
    resolution_choice: str,
    fps_choice: int,
    preview_duration_sec: int,
    enable_captions: bool,
    enable_source_labels: bool,
    enable_music: bool,
    enable_ambience_sfx: bool,
    media_providers: list,
    progress=gr.Progress()
):
    if not script_text or not script_text.strip():
        raise gr.Error("من فضلك الصق السكريبت الإنجليزي أولاً في مساحة العمل!")

    effective_llm_key = (llm_api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("DIRECTOR_API_KEY") or "").strip()
    if not effective_llm_key:
        raise gr.Error("مفتاح API للمخرج مطلوب! أدخل مفتاح DeepSeek في إعدادات الـ API.")

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join("./workspace", "jobs", job_id)
    os.makedirs(job_dir, exist_ok=True)
    log_path = os.path.join(job_dir, "render_log.txt")
    thumbs_dir = os.path.join(job_dir, "live_thumbs")
    os.makedirs(thumbs_dir, exist_ok=True)

    def write_log(line: str):
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")

    write_log(f"[JOB {job_id}] Started | provider={provider} | model={model_id} | mode={render_mode} | style={style_preset}")

    res_map = {
        "1080p Full HD (1920x1080)": (1920, 1080),
        "720p HD Fast (1280x720)": (1280, 720),
        "480p Quick Draft (854x480)": (854, 480),
    }
    is_preview = (render_mode == "Preview Render (Fast Draft)")
    width, height = (1280, 720) if is_preview else res_map.get(resolution_choice, (1920, 1080))
    fps = 24 if is_preview else int(fps_choice or 30)

    # 1. تقسيم المشاهد
    progress(0.08, desc="1/4 🧠 تحليل السكريبت وتقسيم المشاهد الحركية...")
    yield None, None, None, None, None, "🧠 **جاري تحليل السكريبت وتقسيم المشاهد...**", []

    director = KineticDirector(api_key=effective_llm_key, model_id=(model_id or "DeepSeek-V4.1-Flash").strip())
    timeline = director.process_full_script(script_text, job_id=job_id)
    scenes_all = timeline.get("scenes", [])
    write_log(f"[STAGE 1] Planned {len(scenes_all)} scenes.")

    # 2. الصوت والمزامنة
    progress(0.25, desc=f"2/4 🎙️ توليد التعليق الصوتي لـ {len(scenes_all)} مشهد...")
    yield None, None, None, None, None, f"🎙️ **تم تخطيط {len(scenes_all)} مشهد — جاري توليد الصوت ومزامنة الكلمات...**", []

    audio_eng = AudioAndSFXEngine(work_dir=job_dir, voice=voice_name)
    mastered_wav = audio_eng.generate_and_master(timeline, enable_music=enable_music, enable_sfx=enable_ambience_sfx)
    write_log(f"[STAGE 2] Audio mastered ({timeline.get('audio_master', {}).get('total_duration_sec')}s).")

    # 3. جلب الصور الحقيقية
    progress(0.45, desc="3/4 🎥 جلب الصور الأرشيفية الحقيقية المطابقة للمشاهد...")
    yield None, None, None, None, None, "🎥 **جاري سحب الصور الأرشيفية الحقيقية من Wikimedia...**", []

    fetcher = KineticMediaFetcher(work_dir=job_dir)
    timeline = fetcher.populate_timeline_media(timeline)

    media_manifest = [
        {"scene_id": s["scene_id"], "template_type": s.get("template_type"), "local_media_path": s.get("local_media_path")}
        for s in scenes_all if s.get("local_media_path")
    ]
    timeline_path = os.path.join(job_dir, "master_timeline.json")
    media_manifest_path = os.path.join(job_dir, "media_manifest.json")
    source_manifest_path = os.path.join(job_dir, "source_manifest.json")

    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2, ensure_ascii=False)
    with open(media_manifest_path, "w", encoding="utf-8") as f:
        json.dump(media_manifest, f, indent=2, ensure_ascii=False)
    with open(source_manifest_path, "w", encoding="utf-8") as f:
        json.dump({"job_id": job_id, "total_scenes": len(scenes_all), "evidence_scenes": media_manifest}, f, indent=2, ensure_ascii=False)

    # 4. الريندر الحي مشهد بمشهد مع عرض الصور مباشرة في الواجهة
    renderer = KineticRenderer(
        work_dir=job_dir,
        width=width,
        height=height,
        fps=fps,
        style_preset=style_preset,
        enable_captions=enable_captions,
        enable_source_labels=enable_source_labels
    )

    max_prev = float(preview_duration_sec) if is_preview else None
    scenes_to_render = scenes_all
    if max_prev and max_prev > 0:
        sub, acc = [], 0.0
        for s in scenes_all:
            sub.append(s)
            acc += float(s.get("duration") or 3.0)
            if acc >= max_prev:
                break
        scenes_to_render = sub

    total_s = len(scenes_to_render)
    live_gallery = []
    clips_dict = {}

    def render_with_preview(scene_obj: Dict[str, Any]):
        sid = int(scene_obj["scene_id"])
        dur = max(1.2, float(scene_obj.get("duration", 3.0)))
        thumb_path = os.path.join(thumbs_dir, f"thumb_{sid:04d}.jpg")
        if scene_obj.get("local_media_path") and os.path.exists(scene_obj["local_media_path"]):
            preview_img = Image.open(scene_obj["local_media_path"]).convert("RGB")
        else:
            preview_img = renderer._render_frame(scene_obj, rel_t=dur * 0.65, duration=dur)
        preview_img.thumbnail((640, 360))
        preview_img.save(thumb_path, quality=85)

        clip_path = renderer.render_scene(scene_obj)
        caption = f"#{sid} [{scene_obj.get('template_type')}] {str(scene_obj.get('raw_text', ''))[:45]}..."
        return sid, clip_path, (thumb_path, caption)

    completed_count = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(render_with_preview, s): s for s in scenes_to_render}
        for fut in as_completed(futures):
            sid, clip_path, gallery_item = fut.result()
            clips_dict[sid] = clip_path
            live_gallery.append(gallery_item)
            live_gallery.sort(key=lambda x: int(x[1].split()[0].lstrip("#")))
            completed_count += 1

            pct = 0.50 + 0.45 * (completed_count / max(1, total_s))
            progress(pct, desc=f"4/4 🎬 جاري الريندر الحي: اكتمل {completed_count} من {total_s} مشهد...")
            status_live = f"⏳ **جاري الريندر الحي:** اكتمل `{completed_count} / {total_s}` مشهد (تظهر المشاهد بالأسفل فور انتهائها)"
            yield None, timeline_path, media_manifest_path, source_manifest_path, log_path, status_live, list(live_gallery)

    # تجميع الفيديو النهائي
    ordered_clips = [clips_dict[int(s["scene_id"])] for s in scenes_to_render]
    concat_txt = os.path.join(job_dir, "concat.txt")
    with open(concat_txt, "w", encoding="utf-8") as f:
        for c in ordered_clips:
            f.write(f"file '{os.path.abspath(c)}'\n")

    out_filename = "velvet_preview.mp4" if is_preview else "velvet_documentary.mp4"
    final_mp4 = os.path.join(job_dir, out_filename)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt, "-i", mastered_wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", final_mp4],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    write_log(f"[STAGE 4] Export complete: {final_mp4}")
    progress(1.0, desc=f"✨ اكتمل الإنتاج بنجاح (Job ID: {job_id})")
    status_md = f"**✅ Job ID:** `{job_id}` | **Resolution:** `{width}x{height} @ {fps}fps` | **Scenes:** `{total_s}`"
    yield final_mp4, timeline_path, media_manifest_path, source_manifest_path, log_path, status_md, list(live_gallery)


velvet_theme = gr.themes.Base(
    primary_hue="amber",
    secondary_hue="stone",
    neutral_hue="zinc",
).set(
    body_background_fill="#070709",
    block_background_fill="#101016",
    block_border_width="1px",
    block_border_color="rgba(212, 175, 55, 0.2)",
)

with gr.Blocks(theme=velvet_theme, css=VELVET_CUSTOM_CSS, title="VELVET STUDIO // Kinetic Essay Engine v4.0") as demo:
    gr.HTML("""
        <div class="velvet-header">
            <div class="velvet-badge">VELVET KINETIC ENGINE v4.0 • LIVE SCENE MONITOR</div>
            <h1 class="velvet-title">VELVET DOCUMENTARY STUDIO</h1>
            <p style="color:#9e988e; font-size:14px; margin-top:4px;">محرك إنتاج المقالات المرئية والأفلام الوثائقية — نصوص حركية متزامنة، صور أرشيفية حقيقية، ومعاينة حية للمشاهد أثناء الريندر</p>
        </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=7, elem_classes="velvet-card"):
            gr.Markdown("### ✍️ مساحة السكريبت الوثائقي (Script Workspace)")
            script_input = gr.Textbox(
                lines=15,
                placeholder="Paste your continuous English documentary script here...",
                show_label=False
            )
            stats_display = gr.HTML(update_script_stats(""))
            script_input.change(fn=update_script_stats, inputs=script_input, outputs=stats_display)

            with gr.Row():
                render_mode = gr.Radio(
                    choices=["Final Render (Full Quality)", "Preview Render (Fast Draft)"],
                    value="Preview Render (Fast Draft)",
                    label="وضع الريندر"
                )
                preview_duration_sec = gr.Slider(minimum=15, maximum=90, value=25, step=5, label="مدة المعاينة السريعة (ثانية)")

            run_btn = gr.Button("🎬 بدء إنتاج الفيلم الوثائقي (Produce Kinetic Documentary)", elem_id="produce-btn")

        with gr.Column(scale=4, elem_classes="velvet-card"):
            gr.Markdown("### 🎛️ إعدادات الصوت والإخراج")
            voice_name = gr.Radio(
                choices=[
                    ("🇬🇧 البريطاني الفخم (George)", "bm_george"),
                    ("🇺🇸 الأمريكي العميق (Michael)", "am_michael"),
                    ("🇬🇧 البريطاني الروائي (Daniel)", "bm_daniel"),
                    ("🇺🇸 الأمريكي الحماسي (Adam)", "am_adam"),
                    ("🇬🇧 البريطاني التحليلي (Lewis)", "bm_lewis")
                ],
                value="bm_george",
                label="نبرة المعلق الصوتي"
            )

            with gr.Accordion("🎨 إعدادات الصورة والصوت والمصادر", open=False):
                style_preset = gr.Dropdown(
                    choices=["Investigative Essay", "Archival Historical", "Natural Environment"],
                    value="Investigative Essay",
                    label="النمط اللوني والبصري (Visual Style Preset)"
                )
                resolution_choice = gr.Dropdown(
                    choices=["1080p Full HD (1920x1080)", "720p HD Fast (1280x720)", "480p Quick Draft (854x480)"],
                    value="1080p Full HD (1920x1080)",
                    label="دقة الفيديو"
                )
                fps_choice = gr.Dropdown(choices=[24, 30], value=30, label="معدل الإطارات (FPS)")
                media_providers = gr.CheckboxGroup(
                    choices=["wikimedia", "wikipedia"],
                    value=["wikimedia", "wikipedia"],
                    label="مصادر الصور الحقيقية المفعلة"
                )
                enable_captions = gr.Checkbox(value=False, label="تفعيل الترجمة السفلية الإضافية (Subtitles)")
                enable_source_labels = gr.Checkbox(value=True, label="إظهار شارات المصادر والعناوين (Source Labels)")
                enable_music = gr.Checkbox(value=True, label="تفعيل الخلفية الموسيقية الداكنة (Dark Ambient Drone)")
                enable_ambience_sfx = gr.Checkbox(value=True, label="تفعيل المؤثرات الصوتية السينمائية (Boom & Whoosh SFX)")

            with gr.Accordion("🔑 مفاتيح الذكاء الاصطناعي والـ API", open=True):
                provider = gr.Dropdown(
                    choices=[("DeepSeek AI (Exclusive)", "deepseek")],
                    value="deepseek",
                    label="مزود الذكاء الاصطناعي للمخرج"
                )
                model_id = gr.Textbox(value="DeepSeek-V4.1-Flash", label="اسم الموديل (مثبت)")
                llm_api_key = gr.Textbox(
                    type="password",
                    value=os.getenv("DEEPSEEK_API_KEY", ""),
                    placeholder="sk-...",
                    label="مفتاح DeepSeek API Key"
                )
                pixabay_key = gr.Textbox(
                    type="password",
                    value=os.getenv("PIXABAY_API_KEY", ""),
                    placeholder="اختياري...",
                    label="مفتاح Pixabay API (اختياري)"
                )
                pexels_key = gr.Textbox(
                    type="password",
                    value=os.getenv("PEXELS_API_KEY", ""),
                    placeholder="اختياري...",
                    label="مفتاح Pexels API (اختياري)"
                )

    with gr.Row(elem_classes="velvet-card"):
        with gr.Column(scale=7):
            gr.Markdown("### 📽️ شاشة العرض النهائية والمعاينة الحية للمشاهد")
            job_status_md = gr.Markdown("**الحالة:** جاهز لبدء مشروع جديد")
            live_gallery_out = gr.Gallery(
                label="🎞️ المشاهد التي تم رندرتها (تظهر لحظة بلحظة أثناء الريندر)",
                columns=3,
                height=280
            )
            video_out = gr.Video(label="الفيديو الوثائقي الجاهز", height=440)
        with gr.Column(scale=4):
            gr.Markdown("### 📂 ملفات المشروع والتوثيق")
            json_out = gr.File(label="master_timeline.json")
            media_manifest_out = gr.File(label="media_manifest.json")
            source_manifest_out = gr.File(label="source_manifest.json")
            log_out = gr.File(label="render_log.txt")

    run_btn.click(
        fn=run_velvet_pipeline,
        inputs=[
            script_input, provider, model_id, llm_api_key, pexels_key, pixabay_key, voice_name,
            render_mode, style_preset, resolution_choice, fps_choice, preview_duration_sec,
            enable_captions, enable_source_labels, enable_music, enable_ambience_sfx, media_providers
        ],
        outputs=[video_out, json_out, media_manifest_out, source_manifest_out, log_out, job_status_md, live_gallery_out]
    )

if __name__ == "__main__":
    ui_auth = None
    raw_auth = os.getenv("VELVET_UI_AUTH", "").strip()
    if ":" in raw_auth:
        u, p = raw_auth.split(":", 1)
        ui_auth = (u, p)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, auth=ui_auth)
