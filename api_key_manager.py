"""
Module quản lý xoay vòng API Key cho Gemini.
Cơ chế: Round-Robin + Auto-Retry khi bị Rate Limit.
Tương thích với google.genai.Client API mới.
"""
import os
import time
import threading
import logging
from typing import Optional, Any, List
from dotenv import load_dotenv

import google.genai as genai
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

load_dotenv()

logger = logging.getLogger("api_key_manager")
logging.basicConfig(level=logging.INFO)

# Thời gian cooldown khi key bị rate limit (giây)
DEFAULT_COOLDOWN_SECONDS = 60


class GeminiKeyManager:
    """
    Quản lý nhiều Gemini API Key với cơ chế xoay vòng (Round-Robin).
    - Tự động chuyển key khi gặp lỗi Rate Limit (429 / ResourceExhausted).
    - Thread-safe cho môi trường FastAPI đa luồng.
    - Cooldown key bị rate limit, tự mở lại sau thời gian chờ.
    - Tương thích với google.genai.Client API.
    """

    def __init__(self, cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS):
        self._lock = threading.Lock()
        self._cooldown_seconds = cooldown_seconds

        # Ưu tiên đọc GEMINI_API_KEYS (nhiều key), fallback về GEMINI_API_KEY (key đơn)
        keys_str = os.getenv("GEMINI_API_KEYS", "")
        if keys_str.strip():
            self._keys: List[str] = [k.strip() for k in keys_str.split(",") if k.strip()]
        else:
            single_key = os.getenv("GEMINI_API_KEY", "")
            self._keys = [single_key] if single_key else []

        if not self._keys:
            raise ValueError(
                "Không tìm thấy API Key nào! "
                "Hãy đặt GEMINI_API_KEYS=key1,key2,... hoặc GEMINI_API_KEY=key trong file .env"
            )

        # Dict lưu thời điểm key bị rate limit: { key: timestamp }
        self._rate_limited: dict = {}
        # Chỉ số key hiện tại cho round-robin
        self._current_index = 0

        # Pool client persistent (KHÔNG tạo client tạm mỗi lần gọi → tránh bị GC khi stream)
        self._clients: dict = {}
        for key in self._keys:
            self._clients[key] = genai.Client(api_key=key)

        logger.info(f"[KeyManager] Đã nạp {len(self._keys)} API key(s).")

    # -------------------------------------------------------
    # LẤY KEY TIẾP THEO (ROUND-ROBIN)
    # -------------------------------------------------------
    def get_next_key(self) -> str:
        """Trả về API key khả dụng tiếp theo (bỏ qua key đang bị cooldown)."""
        with self._lock:
            now = time.time()
            total_keys = len(self._keys)

            for _ in range(total_keys):
                key = self._keys[self._current_index]
                self._current_index = (self._current_index + 1) % total_keys

                if key in self._rate_limited:
                    elapsed = now - self._rate_limited[key]
                    if elapsed >= self._cooldown_seconds:
                        del self._rate_limited[key]
                        logger.info(f"[KeyManager] Key ...{key[-6:]} đã hết cooldown, sử dụng lại.")
                        return key
                    else:
                        continue
                else:
                    return key

            # Tất cả key đều đang bị rate limit -> chờ key sớm nhất hết cooldown
            earliest_key = min(self._rate_limited, key=lambda k: self._rate_limited[k])
            wait_time = self._cooldown_seconds - (now - self._rate_limited[earliest_key])
            logger.warning(
                f"[KeyManager] TẤT CẢ {total_keys} key đều bị rate limit! "
                f"Chờ {wait_time:.0f}s..."
            )

        time.sleep(max(0.0, wait_time))
        with self._lock:
            if earliest_key in self._rate_limited:
                del self._rate_limited[earliest_key]
            return earliest_key

    # -------------------------------------------------------
    # ĐÁNH DẤU KEY BỊ RATE LIMIT
    # -------------------------------------------------------
    def mark_rate_limited(self, key: str):
        """Đánh dấu key bị rate limit, tạm nghỉ cooldown_seconds."""
        with self._lock:
            self._rate_limited[key] = time.time()
            logger.warning(f"[KeyManager] Key ...{key[-6:]} bị RATE LIMIT! Tạm nghỉ {self._cooldown_seconds}s.")

    # -------------------------------------------------------
    # TẠO CLIENT VỚI KEY XOAY VÒNG
    # -------------------------------------------------------
    def get_client(self) -> genai.Client:
        """Trả về genai.Client instance với API key xoay vòng (từ pool persistent)."""
        key = self.get_next_key()
        return self._clients[key]

    # -------------------------------------------------------
    # TẠO EMBEDDING MODEL VỚI KEY XOAY VÒNG
    # -------------------------------------------------------
    def get_embed_model(self) -> GoogleGenAIEmbedding:
        """Trả về GoogleGenAIEmbedding instance với API key xoay vòng."""
        key = self.get_next_key()
        return GoogleGenAIEmbedding(
            model_name="models/gemini-embedding-001",
            api_key=key,
            query_task_type="RETRIEVAL_QUERY",
            text_task_type="RETRIEVAL_DOCUMENT",
        )

    # -------------------------------------------------------
    # GỌI GEMINI VỚI AUTO-RETRY KHI BỊ RATE LIMIT
    # -------------------------------------------------------
    def generate_with_retry(
        self,
        model: str,
        contents: Any,
        config: Any = None,
        stream: bool = False,
        max_retries: Optional[int] = None,
    ):
        """
        Gọi Gemini generate_content với cơ chế tự xoay key khi bị rate limit.
        Sử dụng persistent client pool để tránh lỗi stream bị đóng.
        """
        if max_retries is None:
            max_retries = len(self._keys)

        last_error = None

        for attempt in range(max_retries):
            key = self.get_next_key()
            client = self._clients[key]  # Dùng client persistent, KHÔNG tạo mới

            try:
                if stream:
                    response = client.models.generate_content_stream(
                        model=model, contents=contents, config=config
                    )
                else:
                    response = client.models.generate_content(
                        model=model, contents=contents, config=config
                    )

                logger.info(f"[KeyManager] Gọi thành công với key ...{key[-6:]} (lần {attempt + 1})")
                return response

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = any(kw in error_str for kw in [
                    "429", "resource exhausted", "rate limit", 
                    "quota", "resourceexhausted", "too many requests"
                ])

                if is_rate_limit:
                    self.mark_rate_limited(key)
                    logger.warning(
                        f"[KeyManager] Lần {attempt + 1}/{max_retries}: "
                        f"Key ...{key[-6:]} bị rate limit. Xoay sang key khác..."
                    )
                    last_error = e
                    continue
                else:
                    raise e

        raise Exception(
            f"Tất cả {len(self._keys)} API key đều bị rate limit sau {max_retries} lần thử! "
            f"Lỗi cuối: {last_error}"
        )


# =========================================================
# SINGLETON INSTANCE - Import trực tiếp từ các file khác
# =========================================================
key_manager = GeminiKeyManager()
