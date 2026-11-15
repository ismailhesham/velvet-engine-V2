import os
import re
import uuid
import json
import random
import logging
import subprocess
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import soundfile as sf
import gradio as gr
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from director import KineticDirector
from media_fetcher import KineticMediaFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("kinetic.app")


class AudioAndSFXEngine:
    """
    محرك الصوت المتكامل:
    - يولّد التعليق الصوتي لكل مشهد (Kokoro TTS) ويستخرج التوقيت الدقيق لكل كلمة (WhisperX).
    - يدمج خلفية موسيقية هادية (Dark Ambient Drone) + مؤثرات صوتية سينمائية (Sub-Bass Boom & Whoosh)
      عند انتقال المشاهد وظهور الأرقام والوثائق.
    """
    def __init__(self, work_dir: str, voice: str = "am_michael", speed: float = 1.0):
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
            result = model.transcribe(audio, batch_size=8)
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

    def generate_and_master(self, timeline: Dict[str, Any], enable_sfx: bool = True) -> str:
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

        voice_only = np.concatenate(seg_arrays) if seg_arrays else np.zeros(self.sample_rate * 2, dtype=np.float32)
        voice_wav = os.path.join(self.work_dir, "voice_master.wav")
        sf.write(voice_wav, voice_only, self.sample_rate)

        self._align_whisperx(voice_wav, scenes)

        # Mix subtle dark ambient bed + cinematic transition SFX (Sub-Bass Boom & Soft Whoosh)
        master_audio = voice_only.copy()
        if enable_sfx and len(master_audio) > 0:
            t = np.arange(len(master_audio), dtype=np.float32) / float(self.sample_rate)
            drone = (
                0.016 * np.sin(2 * np.pi * 55.0 * t)
                + 0.010 * np.sin(2 * np.pi * 82.41 * t)
                + 0.006 * np.sin(2 * np.pi * 110.0 * t)
            ).astype(np.float32)
            master_audio += drone

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
    """
    محرك الريندر البصري (MagnatesMedia / Moon Style):
    - خلفية داكنة أنيقة بنسيج شبكي خفيف.
    - نصوص حركية ضخمة تتلون كلمة بكلمة مع الصوت وتبرز الكلمات المفتاحية بالذهبي.
    - صور حقيقية بإطارات عصرية وحركة زووم ناعمة + إحصائيات عملاقة + بطاقات أدلة.
    """
    def __init__(self, work_dir: str, width: int = 1920, height: int = 1080, fps: int = 30):
        self.work_dir = work_dir
        self.renders_dir = os.path.join(work_dir, "scene_renders")
        os.makedirs(self.renders_dir, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self.bg_color = (12, 13, 17)
        self.cream = (248, 249, 250)
        self.dim_text = (75, 79, 88)
        self.muted = (158, 164, 176)
        self.accent = (245, 197, 24)

        random.seed(42)
        self.particles = [
            {
                "x": random.uniform(0, self.width),
                "y": random.uniform(0, self.height),
                "vx": random.uniform(-6, 6),
                "vy": random.uniform(-10, -3),
                "r": random.uniform(1.0, 2.0),
                "alpha": random.randint(15, 45)
            }
            for _ in range(30)
        ]
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
        scaled = max(14, int(round(size * (self.height / 1080.0))))
        paths = [
            "./assets/fonts/Inter-Bold.ttf" if bold else "./assets/fonts/Inter-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for p in paths:
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, scaled)
                except Exception:
                    continue
        return ImageFont.load_default()

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
        base = Image.new("RGBA", (self.width, self.height), (*self.bg_color, 255))
        layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        cx, cy = self.width // 2, self.height // 2
        max_r = int(self.height * 0.72)
        for r in range(max_r, 40, -60):
            alpha = int(22 * (1 - r / max_r))
            draw.ellipse([cx - int(r * 1.55), cy - r, cx + int(r * 1.55), cy + r], fill=(34, 38, 48, alpha))
        step = max(60, int(self.width * 0.045))
        for gx in range(0, self.width, step):
            draw.line([gx, 0, gx, self.height], fill=(255, 255, 255, 6), width=1)
        for gy in range(0, self.height, step):
            draw.line([0, gy, self.width, gy], fill=(255, 255, 255, 6), width=1)
        self._bg_cache = Image.alpha_composite(base, layer)
        return self._bg_cache.copy()

    def _draw_glow_text(self, canvas: Image.Image, pos: Tuple[int, int], text: str, font: ImageFont.FreeTypeFont, fill: Tuple[int, int, int], anchor: str = "mm"):
        glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        ImageDraw.Draw(glow).text(pos, text, font=font, fill=(0, 0, 0, 95), anchor=anchor)
        glow = glow.filter(ImageFilter.GaussianBlur(radius=8))
        canvas.alpha_composite(glow)
        ImageDraw.Draw(canvas).text(pos, text, font=font, fill=(*fill, 255), anchor=anchor)

    def _render_frame(self, scene: Dict[str, Any], rel_t: float, duration: float) -> Image.Image:
        progress = min(1.0, max(0.0, rel_t / max(0.1, duration)))
        abs_time = float(scene.get("start_time", 0.0)) + rel_t
        t_type = scene.get("template_type", "KINETIC_TEXT")
        t_data = scene.get("template_data") or {}

        canvas = self._static_bg()
        overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        o_draw = ImageDraw.Draw(overlay)
        for p in self.particles:
            px = (p["x"] + p["vx"] * abs_time) % self.width
            py = (p["y"] + p["vy"] * abs_time) % self.height
            o_draw.ellipse([px - p["r"], py - p["r"], px + p["r"], py + p["r"]], fill=(248, 249, 250, p["alpha"]))
        canvas = Image.alpha_composite(canvas, overlay)
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

            self._draw_glow_text(canvas, (cx, cy - int(self.height * 0.06)), num_str, f_num, self.accent)
            bar_w = int(110 * c_eased)
            draw.rectangle([cx - bar_w, cy + int(self.height * 0.05), cx + bar_w, cy + int(self.height * 0.055)], fill=(*self.accent, 255))
            draw.text((cx, cy + int(self.height * 0.10)), label, fill=(*self.cream, 255), font=f_lbl, anchor="mm")
            for i, line in enumerate(self._wrap_lines(str(scene.get("raw_text") or ""), f_desc, safe_w, 2)):
                draw.text((cx, cy + int(self.height * 0.16) + i * 36), line, fill=(*self.muted, 240), font=f_desc, anchor="mm")

        elif t_type == "DOCUMENT_CARD":
            hdr = str(t_data.get("document_header") or "EVIDENCE // RECORD").upper()
            body = str(t_data.get("document_body") or scene.get("raw_text") or "")
            hl = str(t_data.get("highlight_phrase") or "").lower()

            scale = 1.0 + 0.045 * progress
            cw, ch = int(self.width * 0.66 * scale), int(self.height * 0.54 * scale)
            x0, y0 = cx - cw // 2, cy - ch // 2
            x1, y1 = x0 + cw, y0 + ch

            draw.rectangle([x0 + 16, y0 + 18, x1 + 16, y1 + 18], fill=(0, 0, 0, 195))
            draw.rectangle([x0, y0, x1, y1], fill=(244, 241, 234, 255))
            draw.rectangle([x0, y0, x1, y0 + 54], fill=(22, 25, 33, 255))
            draw.rectangle([x0, y0, x0 + 8, y1], fill=(*self.accent, 255))

            f_hdr = self._get_font(19, bold=True)
            f_body = self._get_font(32, bold=True)
            draw.text((x0 + 30, y0 + 27), hdr[:58], fill=(*self.accent, 255), font=f_hdr, anchor="lm")

            lines = self._wrap_lines(body, f_body, cw - 84, 5)
            hl_prog = min(1.0, max(0.0, (rel_t - 0.15) / 0.55))
            for idx, line in enumerate(lines):
                ly = y0 + 100 + idx * int(50 * (self.height / 1080.0))
                is_hl = bool(hl and any(tok in line.lower() for tok in hl.split()[:3])) or (idx == len(lines) // 2)
                if is_hl and hl_prog > 0.05:
                    bbox = draw.textbbox((x0 + 40, ly), line, font=f_body)
                    bw = int((bbox[2] - bbox[0] + 14) * hl_prog)
                    draw.rectangle([x0 + 34, ly - 4, x0 + 34 + bw, bbox[3] + 8], fill=(*self.accent, 210))
                draw.text((x0 + 40, ly), line, fill=(18, 20, 26, 255), font=f_body)

        else:
            raw_text = str(scene.get("raw_text") or "")
            word_ts = scene.get("word_timestamps") or []
            words = raw_text.split() or ["..."]
            hl_raw = t_data.get("highlight_words") or []
            if isinstance(hl_raw, str):
                hl_raw = hl_raw.split()
            hl_tokens = {re.sub(r"[^a-z0-9]", "", w.lower()) for w in hl_raw if len(w) > 2}

            lit_count = 0
            if word_ts:
                for idx, w_info in enumerate(word_ts):
                    if abs_time >= float(w_info.get("start", 0.0)) - 0.04:
                        lit_count = idx + 1
            else:
                lit_count = max(1, int(len(words) * min(1.0, progress * 1.25)))

            kicker = str(t_data.get("kicker") or "").upper().strip()
            f_size = 56 if len(words) <= 10 else (46 if len(words) <= 18 else 38)
            f_word = self._get_font(f_size, bold=True)
            f_kick = self._get_font(19, bold=True)

            per_line = 6 if len(words) <= 12 else 7
            lines = [words[i:i + per_line] for i in range(0, len(words), per_line)][:4]
            spacing = int((f_size + 26) * (self.height / 1080.0))
            start_y = cy - ((len(lines) - 1) * spacing) // 2

            if kicker:
                draw.text((cx, start_y - int(65 * (self.height / 1080.0))), kicker[:44], fill=(*self.accent, 235), font=f_kick, anchor="mm")

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
                        self._draw_glow_text(canvas, (curr_x, ly), w, f_word, col, anchor="lm")
                    else:
                        draw.text((curr_x, ly), w, fill=(*self.dim_text, 190), font=f_word, anchor="lm")
                    curr_x += ww
                    g_idx += 1

        return canvas.convert("RGB")

    def _encode_flags(self) -> List[str]:
        preset = "fast" if self.vcodec == "h264_nvenc" else "ultrafast"
        return ["-c:v", self.vcodec, "-preset", preset, "-profile:v", "main", "-pix_fmt", "yuv420p", "-r", str(self.fps), "-an"]

    def render_scene(self, scene: Dict[str, Any]) -> str:
        sid = int(scene["scene_id"])
        dur = max(1.2, float(scene.get("duration", 3.0)))
        t_data = scene.get("template_data") or {}
        media_path = scene.get("local_media_path")
        out_clip = os.path.join(self.renders_dir, f"clip_{sid:04d}.mp4")

        if media_path and os.path.exists(media_path):
            try:
                bg = self._static_bg().convert("RGB")
                fg = Image.open(media_path).convert("RGB")
                fg.thumbnail((int(self.width * 0.66), int(self.height * 0.62)), Image.Resampling.LANCZOS)

                pad, cap_h = 10, 52
                fw, fh = fg.width + pad * 2, fg.height + pad * 2 + cap_h
                frame_img = Image.new("RGB", (fw, fh), (20, 23, 30))
                df = ImageDraw.Draw(frame_img)
                df.rectangle([0, 0, fw - 1, fh - 1], outline=self.accent, width=2)
                frame_img.paste(fg, (pad, pad))

                cap_txt = str(t_data.get("caption") or scene.get("visual_goal") or "VISUAL EVIDENCE").upper()[:46]
                df.text((fw // 2, fg.height + pad + cap_h // 2), cap_txt, fill=self.accent, font=self._get_font(20, True), anchor="mm")

                bg.paste(frame_img, ((self.width - fw) // 2, (self.height - fh) // 2))
                framed_jpg = os.path.join(self.renders_dir, f"framed_{sid:04d}.jpg")
                bg.save(framed_jpg, quality=94)

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
        for f_idx in range(frames):
            img = self._render_frame(scene, rel_t=f_idx / float(self.fps), duration=dur)
            proc.stdin.write(img.tobytes())
        proc.stdin.close()
        proc.wait()
        return out_clip

    def build_movie(self, timeline: Dict[str, Any], audio_wav: str, out_mp4: str, preview_sec: Optional[float] = None) -> str:
        scenes = timeline.get("scenes") or []
        if preview_sec and preview_sec > 0:
            sub, acc = [], 0.0
            for s in scenes:
                sub.append(s)
                acc += float(s.get("duration") or 3.0)
                if acc >= preview_sec:
                    break
            scenes = sub

        clips = []
        for idx, s in enumerate(scenes, start=1):
            logger.info("Rendering scene %d/%d (%s)...", idx, len(scenes), s.get("template_type"))
            clips.append(self.render_scene(s))

        concat_txt = os.path.join(self.work_dir, "concat.txt")
        with open(concat_txt, "w", encoding="utf-8") as f:
            for c in clips:
                f.write(f"file '{os.path.abspath(c)}'\n")

        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt, "-i", audio_wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out_mp4],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return out_mp4


def generate_video_essay(
    script_text: str,
    deepseek_api_key: str,
    voice_name: str,
    speech_speed: float,
    render_mode: str,
    enable_sfx: bool,
    progress=gr.Progress()
):
    if not (script_text or "").strip():
        raise gr.Error("يرجى إدخال السكريبت الإنجليزي أولاً.")
    if not (deepseek_api_key or "").strip():
        raise gr.Error("يرجى إدخال مفتاح DeepSeek API Key.")

    job_id = uuid.uuid4().hex[:10]
    job_dir = os.path.join("./workspace/jobs", job_id)
    os.makedirs(job_dir, exist_ok=True)

    progress(0.10, desc="1/4: تقطيع السكريبت وإخراج المشاهد (DeepSeek-V4.1-Flash)...")
    director = KineticDirector(api_key=deepseek_api_key, model_id="DeepSeek-V4.1-Flash")
    timeline = director.process_full_script(script_text, job_id=job_id)

    progress(0.35, desc="2/4: توليد التعليق الصوتي والمؤثرات ومزامنة الكلمات...")
    audio_engine = AudioAndSFXEngine(work_dir=job_dir, voice=voice_name, speed=speech_speed)
    master_wav = audio_engine.generate_and_master(timeline, enable_sfx=enable_sfx)

    progress(0.60, desc="3/4: جلب الصور الأرشيفية الحقيقية المطابقة...")
    fetcher = KineticMediaFetcher(work_dir=job_dir)
    timeline = fetcher.populate_timeline_media(timeline)

    with open(os.path.join(job_dir, "timeline.json"), "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2, ensure_ascii=False)

    is_preview = "Preview" in render_mode
    width, height = (1280, 720) if is_preview else (1920, 1080)
    preview_sec = 25.0 if is_preview else None

    progress(0.75, desc="4/4: ريندر الفيديو النهائي...")
    renderer = KineticRenderer(work_dir=job_dir, width=width, height=height, fps=30)
    final_mp4 = os.path.join(job_dir, f"kinetic_essay_{job_id}.mp4")
    renderer.build_movie(timeline, audio_wav=master_wav, out_mp4=final_mp4, preview_sec=preview_sec)

    progress(1.0, desc="تم الإنتاج بنجاح!")
    summary = f"تم إنتاج {len(timeline['scenes'])} مشهد بنجاح | المسار: {final_mp4}"
    return final_mp4, summary


with gr.Blocks(theme=gr.themes.Base(primary_hue="amber", neutral_hue="slate"), title="KINETIC ESSAY STUDIO") as demo:
    gr.Markdown("# 🎬 KINETIC ESSAY STUDIO\n**محرك إنتاج المقالات المرئية الحديثة (Kinetic Typography + Real Photographic Evidence + Cinematic SFX)**")

    with gr.Row():
        with gr.Column(scale=2):
            script_box = gr.Textbox(
                label="السكريبت الإنجليزي (Full English Script)",
                placeholder="Paste your documentary or video-essay script here...",
                lines=14
            )
        with gr.Column(scale=1):
            api_key_box = gr.Textbox(label="DeepSeek API Key", type="password", placeholder="sk-...")
            model_display = gr.Textbox(label="الموديل المستخدم (مثبت)", value="DeepSeek-V4.1-Flash", interactive=False)
            voice_dropdown = gr.Dropdown(
                label="صوت المعلق (Kokoro TTS)",
                choices=["am_michael", "am_adam", "bm_george", "bm_lewis", "af_bella"],
                value="am_michael"
            )
            speed_slider = gr.Slider(label="سرعة الإلقاء", minimum=0.85, maximum=1.20, value=1.0, step=0.05)
            mode_radio = gr.Radio(
                label="وضع الريندر",
                choices=["Preview Render (Fast 25s - 720p)", "Full Production Render (1080p)"],
                value="Preview Render (Fast 25s - 720p)"
            )
            sfx_checkbox = gr.Checkbox(label="تفعيل المؤثرات الصوتية السينمائية والخلفية (SFX & Ambient)", value=True)
            run_btn = gr.Button("🚀 إنتاج الفيديو الآن", variant="primary")

    with gr.Row():
        video_out = gr.Video(label="المعاينة والتحميل")
        status_out = gr.Textbox(label="حالة الإنتاج ومسار الملف", interactive=False)

    run_btn.click(
        fn=generate_video_essay,
        inputs=[script_box, api_key_box, voice_dropdown, speed_slider, mode_radio, sfx_checkbox],
        outputs=[video_out, status_out]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)