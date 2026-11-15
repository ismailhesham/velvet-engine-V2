import os
import re
import json
import time
import logging
from typing import Dict, Any, List
import requests

logger = logging.getLogger("kinetic.director")


class KineticDirector:
    """
    مخرج الفيديو التحليلي الحديث (DeepSeek-V4.1-Flash Exclusive Edition):
    - يعمل حصرياً مع DeepSeek ومثبت على موديل DeepSeek-V4.1-Flash.
    - يقطع السكريبت لجمل سريعة (6-14 كلمة للمشهد) بإيقاع قنوات MagnatesMedia و Moon.
    - يحافظ على كل كلمة من السكريبت الأصلي بالترتيب الدقيق 100%.
    """
    def __init__(self, api_key: str = "", model_id: str = "DeepSeek-V4.1-Flash"):
        self.api_key = (api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("DIRECTOR_API_KEY") or "").strip()
        self.model_id = (model_id or "DeepSeek-V4.1-Flash").strip()

    def _get_system_prompt(self) -> str:
        return """You are an elite modern video-essay director (style: MagnatesMedia / Moon / Aperture).
Your task: Segment a continuous English script into fast, punchy visual beats (1 short sentence or clause per beat, 6-14 words, ~3-5 seconds each). NEVER put long paragraphs into a single scene!

STRICT NARRATION RULES:
1. Preserve EVERY SINGLE WORD of the user's original script in exact order inside "raw_text". Never delete, add, summarize, or reorder words.
2. Provide "spoken_text" for TTS: identical to "raw_text" except numbers/years/units are spelled out naturally for speech.

VISUAL FORMATS (KINETIC TYPOGRAPHY + REAL VISUAL EVIDENCE — NO GENERIC STOCK VIDEO):
Build the entire video using a crisp, high-contrast mix of these 4 formats:
1. "KINETIC_TEXT" (~45% of scenes): Bold, centered kinetic typography synced word-by-word to the voiceover on a dark matte canvas.
   - REQUIRED template_data: {"highlight_words": ["2-3 most impactful words from raw_text"], "kicker": "OPTIONAL SHORT CONTEXT TAG"}
2. "FRAMED_IMAGE" (~32% of scenes): Real photographic evidence (real people, places, animals, objects, historical events) fetched from Wikimedia/web with a clean editorial frame and caption.
   - REQUIRED search_queries: ["exact real-world entity or subject name", "specific photograph query"]
   - REQUIRED template_data: {"caption": "Short 2-4 word visual label"}
3. "NUMBER_COUNTER" (~13% of scenes): Huge bold statistic filling the screen whenever a striking number, dollar amount, percentage, or count is spoken.
   - REQUIRED template_data: {"target_number": 10012, "prefix": "$", "suffix": "B", "label": "SHORT STAT LABEL"}
4. "DOCUMENT_CARD" (~10% of scenes): High-contrast modern evidence card for quotes, official findings, laws, or headlines.
   - REQUIRED template_data: {"document_header": "EVIDENCE // RECORD", "document_body": "1-2 sentence quote or finding", "highlight_phrase": "key 3-6 words"}

- Alternate formats dynamically so the screen feels alive and rhythmic (never repeat FRAMED_IMAGE or NUMBER_COUNTER twice in a row).

Return valid JSON:
{
  "detected_niche": "Modern Video Essay",
  "recommended_accent_hex": "#f5c518",
  "scenes": [
    {
      "scene_id": 1,
      "raw_text": "...",
      "spoken_text": "...",
      "visual_goal": "...",
      "template_type": "KINETIC_TEXT",
      "search_queries": ["specific real-world query if FRAMED_IMAGE"],
      "template_data": {"highlight_words": ["key", "words"], "kicker": ""}
    }
  ]
}"""

    def _clean_json_text(self, raw_text: str) -> str:
        text = (raw_text or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            text = text[first_brace:last_brace + 1]
        return text.strip()

    def _tokenize_words(self, text: str) -> List[str]:
        return re.findall(r"\S+", (text or "").strip())

    def verify_and_repair_word_preservation(self, original_chunk: str, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        orig_words = self._tokenize_words(original_chunk)
        if not orig_words:
            return []

        valid_scenes = [s for s in scenes if isinstance(s, dict) and str(s.get("raw_text") or "").strip()]
        if not valid_scenes:
            return [{
                "scene_id": 1,
                "raw_text": original_chunk.strip(),
                "spoken_text": original_chunk.strip(),
                "template_type": "KINETIC_TEXT",
                "template_data": {"highlight_words": orig_words[:3]}
            }]

        reconstructed_words = []
        for s in valid_scenes:
            reconstructed_words.extend(self._tokenize_words(s.get("raw_text", "")))

        if orig_words == reconstructed_words:
            return valid_scenes

        total_orig = len(orig_words)
        total_llm = max(1, len(reconstructed_words))
        cursor = 0
        repaired_scenes = []

        for idx, s in enumerate(valid_scenes):
            if cursor >= total_orig:
                break
            if idx == len(valid_scenes) - 1:
                slice_words = orig_words[cursor:]
                cursor = total_orig
            else:
                s_len = max(1, len(self._tokenize_words(s.get("raw_text", ""))))
                alloc = max(1, int(round((s_len / total_llm) * total_orig)))
                end_idx = min(total_orig, cursor + alloc)
                for lookahead in range(max(cursor + 1, end_idx - 2), min(total_orig, end_idx + 3)):
                    if orig_words[lookahead - 1].endswith((".", "!", "?", ";", ",")):
                        end_idx = lookahead
                        break
                slice_words = orig_words[cursor:end_idx]
                cursor = end_idx

            if slice_words:
                exact_segment = " ".join(slice_words)
                s_copy = dict(s)
                s_copy["raw_text"] = exact_segment
                s_copy["spoken_text"] = exact_segment
                repaired_scenes.append(s_copy)

        if cursor < total_orig and repaired_scenes:
            repaired_scenes[-1]["raw_text"] += " " + " ".join(orig_words[cursor:])
            repaired_scenes[-1]["spoken_text"] = repaired_scenes[-1]["raw_text"]

        return repaired_scenes

    def _enforce_visual_rhythm(self, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        allowed = {"KINETIC_TEXT", "FRAMED_IMAGE", "NUMBER_COUNTER", "DOCUMENT_CARD"}
        for i, s in enumerate(scenes):
            if s.get("template_type") not in allowed:
                s["template_type"] = "FRAMED_IMAGE" if s.get("search_queries") else "KINETIC_TEXT"
            if i > 0 and scenes[i - 1].get("template_type") == s.get("template_type") == "FRAMED_IMAGE":
                s["template_type"] = "KINETIC_TEXT"
        return scenes

    def _call_deepseek(self, user_prompt: str) -> str:
        url = "https://api.deepseek.com/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        models_to_try = [self.model_id]
        if self.model_id != "deepseek-chat":
            models_to_try.append("deepseek-chat")

        last_err = None
        for model_name in models_to_try:
            for attempt in range(1, 3):
                try:
                    payload = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": self._get_system_prompt()},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.2,
                        "response_format": {"type": "json_object"}
                    }
                    resp = requests.post(url, headers=headers, json=payload, timeout=180)
                    if resp.status_code == 400 and model_name != "deepseek-chat":
                        break
                    if resp.status_code == 429:
                        time.sleep(min(20, 3 * attempt))
                        continue
                    if resp.status_code in (401, 403):
                        raise ValueError(f"DeepSeek authentication failed (HTTP {resp.status_code}). Check your API key.")
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"]
                except ValueError:
                    raise
                except Exception as e:
                    last_err = e
                    if attempt < 2:
                        time.sleep(2 * attempt)
        raise RuntimeError(f"DeepSeek API request failed: {last_err}")

    def analyze_script_chunk(self, script_chunk: str, chunk_index: int = 1) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("Missing DeepSeek API key.")

        user_prompt = (
            f"Segment this script part (Part {chunk_index}) into fast, rhythmic video-essay beats "
            f"(6-14 words per scene) using KINETIC_TEXT, FRAMED_IMAGE (real photos), NUMBER_COUNTER, and DOCUMENT_CARD. "
            f"Preserve every word in exact order:\n\n{script_chunk}"
        )

        for json_attempt in range(1, 3):
            try:
                raw_str = self._call_deepseek(user_prompt)
                parsed = json.loads(self._clean_json_text(raw_str))
                if not isinstance(parsed, dict) or not isinstance(parsed.get("scenes"), list):
                    raise ValueError("Missing scenes list")
                return parsed
            except (json.JSONDecodeError, ValueError):
                if json_attempt == 2:
                    return {
                        "detected_niche": "Modern Video Essay",
                        "recommended_accent_hex": "#f5c518",
                        "scenes": [{
                            "scene_id": 1,
                            "raw_text": script_chunk.strip(),
                            "spoken_text": script_chunk.strip(),
                            "template_type": "KINETIC_TEXT",
                            "template_data": {"highlight_words": []}
                        }]
                    }

    def process_full_script(self, full_script: str, max_words_per_chunk: int = 450, job_id: str = "") -> Dict[str, Any]:
        clean_script = (full_script or "").strip()
        if not clean_script:
            raise ValueError("Script text cannot be empty.")

        sentences = re.split(r'(?<=[.!?])\s+', clean_script)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_word_count = 0

        for sentence in sentences:
            words = len(sentence.split())
            if current_word_count + words > max_words_per_chunk and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_word_count = words
            else:
                current_chunk.append(sentence)
                current_word_count += words
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        raw_master: Dict[str, Any] = {
            "job_id": job_id,
            "detected_niche": "Modern Video Essay",
            "recommended_accent_hex": "#f5c518",
            "scenes": []
        }

        scene_id_counter = 1
        for idx, chunk in enumerate(chunks, start=1):
            result = self.analyze_script_chunk(chunk, chunk_index=idx)
            if idx == 1:
                raw_master["detected_niche"] = result.get("detected_niche") or "Modern Video Essay"
                raw_master["recommended_accent_hex"] = result.get("recommended_accent_hex") or "#f5c518"

            repaired_scenes = self.verify_and_repair_word_preservation(chunk, result.get("scenes") or [])
            repaired_scenes = self._enforce_visual_rhythm(repaired_scenes)

            for scene in repaired_scenes:
                scene["scene_id"] = scene_id_counter
                raw_master["scenes"].append(scene)
                scene_id_counter += 1

        return raw_master