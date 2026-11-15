import os
import re
import time
import hashlib
import logging
from typing import Dict, Any, List, Optional
import requests
from PIL import Image
from io import BytesIO

logger = logging.getLogger("kinetic.media")


class KineticMediaFetcher:
    """
    جالب الصور الأرشيفية والحقيقية (Wikimedia Commons + Wikipedia REST):
    - يجلب صور فوتوغرافية حقيقية مطابقة لموضوع كل مشهد (FRAMED_IMAGE).
    - يفلتر أيقونات SVG والصور الصغيرة أو التالفة لضمان جودة عرض سينمائية.
    - يحول المشهد تلقائياً إلى KINETIC_TEXT إذا لم تتوفر صورة حقيقية مطابقة.
    """
    def __init__(self, work_dir: str = "./workspace"):
        self.work_dir = work_dir
        self.cache_dir = os.path.join(work_dir, "media_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "KineticEssayStudio/4.0 (https://github.com/ismailhesham; contact@example.com)"
        })
        self.used_urls = set()

    def _download_valid_image(self, url: str, scene_id: int) -> Optional[str]:
        if not url or url in self.used_urls:
            return None
        lower_u = url.lower()
        if any(ext in lower_u for ext in (".svg", ".pdf", ".tif", ".tiff", ".ogg", ".webm", ".gif")):
            return None

        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            w, h = img.size
            if w < 480 or h < 300:
                return None

            h_digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
            out_path = os.path.join(self.cache_dir, f"scene_{scene_id:04d}_{h_digest}.jpg")
            img.save(out_path, format="JPEG", quality=92)
            self.used_urls.add(url)
            return out_path
        except Exception:
            return None

    def _search_wikimedia(self, query: str) -> List[str]:
        clean_q = re.sub(r"[^\w\s\-]", " ", (query or "")).strip()
        if not clean_q:
            return []

        urls: List[str] = []
        try:
            api_url = "https://commons.wikimedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": f"filetype:bitmap {clean_q}",
                "gsrnamespace": 6,
                "gsrlimit": 6,
                "prop": "imageinfo",
                "iiprop": "url|size",
                "iiurlwidth": 1400
            }
            resp = self.session.get(api_url, params=params, timeout=12)
            data = resp.json()
            pages = (data.get("query") or {}).get("pages") or {}
            for page in pages.values():
                ii = (page.get("imageinfo") or [{}])[0]
                thumb = ii.get("thumburl") or ii.get("url")
                if thumb:
                    urls.append(thumb)
        except Exception:
            pass

        if not urls:
            try:
                wp_url = "https://en.wikipedia.org/w/api.php"
                params = {
                    "action": "query",
                    "format": "json",
                    "generator": "search",
                    "gsrsearch": clean_q,
                    "gsrlimit": 4,
                    "prop": "pageimages",
                    "piprop": "original|thumbnail",
                    "pithumbsize": 1400
                }
                resp = self.session.get(wp_url, params=params, timeout=12)
                data = resp.json()
                pages = (data.get("query") or {}).get("pages") or {}
                for page in pages.values():
                    orig = (page.get("original") or {}).get("source")
                    thumb = (page.get("thumbnail") or {}).get("source")
                    if orig:
                        urls.append(orig)
                    elif thumb:
                        urls.append(thumb)
            except Exception:
                pass

        return urls

    def populate_timeline_media(self, timeline: Dict[str, Any]) -> Dict[str, Any]:
        scenes = timeline.get("scenes") or []
        for scene in scenes:
            if scene.get("template_type") != "FRAMED_IMAGE":
                continue

            scene_id = int(scene.get("scene_id") or 1)
            queries = list(scene.get("search_queries") or [])
            if not queries:
                fallback_q = str(scene.get("visual_goal") or scene.get("raw_text") or "")[:60]
                if fallback_q:
                    queries.append(fallback_q)

            saved_path = None
            for q in queries[:3]:
                candidate_urls = self._search_wikimedia(q)
                for u in candidate_urls:
                    saved_path = self._download_valid_image(u, scene_id)
                    if saved_path:
                        break
                if saved_path:
                    break
                time.sleep(0.15)

            if saved_path:
                scene["local_media_path"] = saved_path
                scene["media_type"] = "image"
            else:
                scene["template_type"] = "KINETIC_TEXT"
                t_data = dict(scene.get("template_data") or {})
                words = [w for w in str(scene.get("raw_text") or "").split() if len(w) > 4]
                t_data.setdefault("highlight_words", words[:2])
                scene["template_data"] = t_data

        return timeline