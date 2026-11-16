# -*- coding: utf-8 -*-
"""
Pure-CPU Neural & Cloud Embedding Engine (Zero PyTorch / Zero CUDA).
Supports three execution backends in priority order:
1. Pure-CPU ONNX Runtime + HuggingFace Tokenizers (BAAI/bge-small-zh-v1.5, 512-dim, ~1ms/text)
2. Cloud Embedding API (DashScope text-embedding-v3 / OpenAI text-embedding-3-small, 512-dim)
3. Synonym-Enhanced N-Gram Semantic Fallback (0 download, <0.1ms/text for offline/CI mode)
"""

import hashlib
import math
import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import httpx
import jieba
import numpy as np
from configs.settings import Settings, get_settings
from src.core.logger import get_logger

logger = get_logger("embedding")

# 商业与无领导小组讨论高频同义词根归一表（强化离线兜底的跨表述语义对齐能力）
_SYNONYM_ROOTS: List[Tuple[Tuple[str, ...], str]] = [
    (("降本", "削减成本", "缩减开支", "压缩费用", "控制成本", "节约资金", "精简开支", "降低成本"), "成本控制"),
    (("营收", "营业收入", "利润", "盈利", "经济效益", "收益", "毛利", "现金流", "业绩"), "经济效益"),
    (("裁员", "精简人员", "优化人力", "缩编", "人员分流", "组织精简"), "人力优化"),
    (("客户", "用户", "消费者", "买家", "受众", "客群"), "客户群体"),
    (("营销", "推广", "宣传", "获客", "引流", "品牌曝光", "投放"), "营销推广"),
    (("风险", "隐患", "危机", "漏洞", "不确定性", "合规风险"), "风险管控"),
    (("创新", "研发", "新产品", "新功能", "技术突破", "智能化", "数字化"), "产品创新"),
    (("市占率", "市场占有率", "市场份额", "行业份额"), "市场份额"),
    (("士气", "积极性", "归属感", "凝聚力", "激励", "绩效考核"), "团队激励"),
    (("协同", "配合", "协作", "跨部门", "联动", "对齐"), "团队协同"),
    (("可行", "落地", "可操作", "执行力", "实施路径"), "方案落地"),
]


class EmbeddingEngine:
    """纯 CPU 高性能语义向量计算引擎 (支持真实 ONNX BGE / 云端 API / 离线兜底)"""

    def __init__(self, settings: Optional[Settings] = None, vector_dim: int = 512):
        self.settings = settings or get_settings()
        self.vector_dim = vector_dim
        self._cache: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._onnx_session = None
        self._onnx_input_names: List[str] = []
        self._tokenizer = None
        self._backend_mode: str = "native"

        provider = (self.settings.embedding_provider or "onnx").lower()
        if provider in ("onnx", "bge", "local", "auto"):
            self._init_onnx()

        if self._onnx_session is None and provider in ("api", "auto"):
            if self._has_valid_embedding_api_key():
                self._backend_mode = "api"
                logger.info("EmbeddingEngine using Cloud Embedding API backend (512-dim).")

        if self._backend_mode == "native":
            jieba.initialize()
            if provider in ("onnx", "bge", "local"):
                logger.debug(
                    "Local ONNX BGE weights not found yet; using synonym-enhanced CPU vectorizer "
                    "(run `python cli.py download-model` or set EMBEDDING_PROVIDER=api for neural BGE)."
                )

    def _has_valid_embedding_api_key(self) -> bool:
        ds_key = self.settings.dashscope_api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        oa_key = self.settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        for k in (ds_key, oa_key):
            if k and not k.startswith("mock") and "your_" not in k:
                return True
        return False

    def _resolve_onnx_paths(self) -> Optional[Tuple[Path, Path]]:
        """查找本地 ONNX 模型权重与 tokenizer.json"""
        candidates: List[Path] = []
        cfg_model = (self.settings.embedding_model or "").strip()
        if cfg_model:
            p = Path(cfg_model).expanduser()
            candidates.append(p)

        candidates.extend([
            self.settings.data_root_dir / "models" / "bge-small-zh-v1.5-onnx",
            Path.home() / ".cache" / "bge_models" / "bge-small-zh-v1.5-onnx",
            Path.home() / ".cache" / "bge_models" / "BAAI_bge-small-zh-v1.5",
        ])

        for cand in candidates:
            if not cand.exists():
                continue
            if cand.is_file() and cand.suffix == ".onnx":
                tok = cand.parent / "tokenizer.json"
                if not tok.exists():
                    tok = cand.parent.parent / "tokenizer.json"
                if tok.exists():
                    return cand, tok
            elif cand.is_dir():
                tok = cand / "tokenizer.json"
                if not tok.exists():
                    continue
                for onnx_rel in (
                    "model_quantized.onnx",
                    "onnx/model_quantized.onnx",
                    "model.onnx",
                    "onnx/model.onnx",
                ):
                    onnx_file = cand / onnx_rel
                    if onnx_file.exists():
                        return onnx_file, tok
        return None

    def _init_onnx(self) -> None:
        """初始化本地纯 CPU ONNX Runtime + HuggingFace Tokenizers 推理会话"""
        resolved = self._resolve_onnx_paths()
        if resolved is None:
            return

        onnx_path, tok_path = resolved
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = min(4, os.cpu_count() or 2)
            sess_opts.inter_op_num_threads = 1
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            session = ort.InferenceSession(
                str(onnx_path),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            tokenizer = Tokenizer.from_file(str(tok_path))
            tokenizer.enable_truncation(max_length=256)
            tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

            self._onnx_session = session
            self._onnx_input_names = [inp.name for inp in session.get_inputs()]
            self._tokenizer = tokenizer
            self._backend_mode = "onnx"
            logger.info(f"⚡ Pure-CPU ONNX BGE Engine loaded: {onnx_path.name} (inputs={self._onnx_input_names})")
        except Exception as exc:
            logger.warning(f"Failed to initialize ONNX BGE session ({onnx_path}), falling back: {exc}")
            self._onnx_session = None
            self._tokenizer = None

    @classmethod
    def download_onnx_model(cls, target_dir: Optional[Path] = None) -> Path:
        """
        一键下载 INT8 量化版 BAAI/bge-small-zh-v1.5 纯 CPU ONNX 权重 (~25MB，无需 PyTorch/GPU)。
        优先走国内可达镜像 hf-mirror.com，失败时自动回退官方源。
        """
        settings = get_settings()
        dest_dir = Path(target_dir or (settings.data_root_dir / "models" / "bge-small-zh-v1.5-onnx"))
        dest_dir.mkdir(parents=True, exist_ok=True)

        files_to_fetch = {
            "tokenizer.json": "Xenova/bge-small-zh-v1.5/resolve/main/tokenizer.json",
            "model_quantized.onnx": "Xenova/bge-small-zh-v1.5/resolve/main/onnx/model_quantized.onnx",
        }
        mirrors = ["https://hf-mirror.com", "https://huggingface.co"]

        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            for local_name, rel_url in files_to_fetch.items():
                out_file = dest_dir / local_name
                if out_file.exists() and out_file.stat().st_size > 1024:
                    continue
                last_err = None
                for base in mirrors:
                    url = f"{base}/{rel_url}"
                    try:
                        logger.info(f"Downloading {local_name} from {url} ...")
                        resp = client.get(url)
                        resp.raise_for_status()
                        out_file.write_bytes(resp.content)
                        logger.info(f"Saved {out_file} ({len(resp.content) / 1024 / 1024:.1f} MB)")
                        last_err = None
                        break
                    except Exception as exc:
                        last_err = exc
                if last_err is not None:
                    raise RuntimeError(f"Failed to download {local_name}: {last_err}") from last_err

        return dest_dir

    @property
    def backend_mode(self) -> str:
        """当前实际生效的向量后端: 'onnx' | 'api' | 'native'"""
        return self._backend_mode

    def _align_and_normalize(self, mat: np.ndarray) -> np.ndarray:
        """对齐维度至 vector_dim 并执行 L2 归一化"""
        mat = np.asarray(mat, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat[np.newaxis, :]

        cur_dim = mat.shape[1]
        if cur_dim > self.vector_dim:
            mat = mat[:, : self.vector_dim]
        elif cur_dim < self.vector_dim:
            pad = np.zeros((mat.shape[0], self.vector_dim - cur_dim), dtype=np.float32)
            mat = np.hstack([mat, pad])

        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return (mat / norms).astype(np.float32)

    def _embed_onnx_batch(self, texts: List[str]) -> np.ndarray:
        """执行纯 CPU ONNX Runtime 批量神经网络前向推理 (BGE CLS Pooling + L2 Norm)"""
        cleaned = [t.replace("\n", " ").strip() or " " for t in texts]
        batch_size = max(1, self.settings.embedding_batch_size)
        chunks_out: List[np.ndarray] = []

        for start in range(0, len(cleaned), batch_size):
            sub = cleaned[start : start + batch_size]
            encodings = self._tokenizer.encode_batch(sub)
            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

            feed: Dict[str, np.ndarray] = {}
            if "input_ids" in self._onnx_input_names:
                feed["input_ids"] = input_ids
            if "attention_mask" in self._onnx_input_names:
                feed["attention_mask"] = attention_mask
            if "token_type_ids" in self._onnx_input_names:
                feed["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

            outputs = self._onnx_session.run(None, feed)
            first_out = outputs[0]
            # BGE 标准池化：取 [CLS] token 向量 (index 0)
            if first_out.ndim == 3:
                cls_vecs = first_out[:, 0, :]
            else:
                cls_vecs = first_out
            chunks_out.append(self._align_and_normalize(cls_vecs))

        return np.vstack(chunks_out)

    def _embed_api_batch(self, texts: List[str]) -> np.ndarray:
        """调用阿里云百炼 (text-embedding-v3) 或 OpenAI 兼容接口批量获取 512 维句向量"""
        ds_key = self.settings.dashscope_api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        oa_key = self.settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")

        if ds_key and not ds_key.startswith("mock") and "your_" not in ds_key:
            api_key = ds_key
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            model = self.settings.dashscope_embedding_model
            max_batch = min(10, self.settings.embedding_batch_size)  # DashScope v3 单批上限 10 条
        elif oa_key and not oa_key.startswith("mock") and "your_" not in oa_key:
            api_key = oa_key
            base_url = self.settings.openai_base_url.rstrip("/")
            model = self.settings.openai_embedding_model
            max_batch = min(64, self.settings.embedding_batch_size)
        else:
            return np.vstack([self._compute_native_vector(t) for t in texts])

        cleaned = [t.replace("\n", " ").strip()[:2000] or "空" for t in texts]
        all_vecs: List[np.ndarray] = []

        try:
            with httpx.Client(timeout=20.0) as client:
                for start in range(0, len(cleaned), max_batch):
                    sub = cleaned[start : start + max_batch]
                    payload = {
                        "model": model,
                        "input": sub,
                        "dimensions": self.vector_dim,
                    }
                    resp = client.post(
                        f"{base_url}/embeddings",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json().get("data", [])
                    data_sorted = sorted(data, key=lambda item: item.get("index", 0))
                    batch_mat = np.array([item["embedding"] for item in data_sorted], dtype=np.float32)
                    all_vecs.append(self._align_and_normalize(batch_mat))
            return np.vstack(all_vecs)
        except Exception as exc:
            logger.warning(f"Cloud Embedding API call failed, falling back to CPU vectorizer: {exc}")
            return np.vstack([self._compute_native_vector(t) for t in texts])

    def embed_text(self, text: str) -> np.ndarray:
        """计算单个文本的归一化稠密向量 (1D float32 array, shape=(512,))"""
        if not text or not text.strip():
            return np.zeros(self.vector_dim, dtype=np.float32)

        cache_key = hashlib.md5(text.strip().encode("utf-8")).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        mat = self.embed_texts([text])
        return mat[0] if mat.shape[0] > 0 else np.zeros(self.vector_dim, dtype=np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """批量计算多条文本的稠密向量矩阵 (N x D float32)，带内存缓存复用"""
        if not texts:
            return np.empty((0, self.vector_dim), dtype=np.float32)

        results: List[Optional[np.ndarray]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_keys: List[str] = []
        uncached_texts: List[str] = []
        seen_uncached: Dict[str, int] = {}

        for idx, raw in enumerate(texts):
            if not raw or not raw.strip():
                results[idx] = np.zeros(self.vector_dim, dtype=np.float32)
                continue
            key = hashlib.md5(raw.strip().encode("utf-8")).hexdigest()
            cached = self._cache.get(key)
            if cached is not None:
                results[idx] = cached
            else:
                uncached_indices.append(idx)
                uncached_keys.append(key)
                if key not in seen_uncached:
                    seen_uncached[key] = len(uncached_texts)
                    uncached_texts.append(raw.strip())

        if uncached_texts:
            if self._backend_mode == "onnx" and self._onnx_session is not None:
                computed = self._embed_onnx_batch(uncached_texts)
            elif self._backend_mode == "api":
                computed = self._embed_api_batch(uncached_texts)
            else:
                computed = np.vstack([self._compute_native_vector(t) for t in uncached_texts])

            with self._lock:
                for u_key, pos in seen_uncached.items():
                    self._cache[u_key] = computed[pos]

            for idx, key in zip(uncached_indices, uncached_keys):
                results[idx] = self._cache[key]

        return np.vstack([r if r is not None else np.zeros(self.vector_dim, dtype=np.float32) for r in results])

    def _compute_native_vector(self, text: str) -> np.ndarray:
        """
        带商业同义词根归一 + 中文分词 + 字符子词 N-gram 带符号随机投影的纯 CPU 离线兜底向量器。
        单次耗时 < 0.15ms，供未下载 ONNX 且未配 API 时的单元测试与离线降级使用。
        """
        cleaned = re.sub(r"[^\w\s\u4e00-\u9fa5]", "", text).strip()
        if not cleaned:
            return np.zeros(self.vector_dim, dtype=np.float32)

        vec = np.zeros(self.vector_dim, dtype=np.float32)

        # 0. 商业同义词根概念对齐增强
        for variants, canonical in _SYNONYM_ROOTS:
            if any(v in cleaned for v in variants):
                digest = hashlib.md5(f"syn_{canonical}".encode("utf-8")).digest()
                idx1 = int.from_bytes(digest[:4], "little") % self.vector_dim
                idx2 = int.from_bytes(digest[4:8], "little") % self.vector_dim
                vec[idx1] += 3.2
                vec[idx2] += 2.4

        words = [w for w in jieba.lcut(cleaned) if w.strip()]
        chars = [c for c in cleaned if not c.isspace()]

        # 1. 词级特征与权重
        for word in words:
            weight = math.log1p(len(word)) * 2.0
            h = int.from_bytes(hashlib.md5(word.encode("utf-8")).digest()[:4], "little") % self.vector_dim
            vec[h] += weight

        # 2. 单字与字符 2-gram / 3-gram 特征
        for ch in chars:
            h = int.from_bytes(hashlib.md5(ch.encode("utf-8")).digest()[:4], "little") % self.vector_dim
            vec[h] += 0.8

        for i in range(len(chars) - 1):
            bi_char = chars[i] + chars[i + 1]
            h = int.from_bytes(hashlib.md5(bi_char.encode("utf-8")).digest()[:4], "little") % self.vector_dim
            vec[h] += 1.3

        for i in range(len(chars) - 2):
            tri_char = chars[i] + chars[i + 1] + chars[i + 2]
            h = int.from_bytes(hashlib.md5(tri_char.encode("utf-8")).digest()[:4], "little") % self.vector_dim
            vec[h] += 1.6

        # 3. 词序 2-gram
        for i in range(len(words) - 1):
            bi_word = f"{words[i]}_{words[i + 1]}"
            h = int.from_bytes(hashlib.md5(bi_word.encode("utf-8")).digest()[:4], "little") % self.vector_dim
            vec[h] += 1.8

        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec.astype(np.float32)

    def compute_sequence_jump_distance(self, texts: List[str]) -> float:
        """
        计算节点/语句序列的累计语义跳动欧氏距离 (100% 对齐 V5 EmbeddingManager.compute_total_distance)
        供 3_1 创造性评分器使用。
        """
        valid = [t for t in texts if t and t.strip()]
        if len(valid) < 2:
            return 0.0
        vecs = self.embed_texts(valid)
        if vecs.shape[0] < 2:
            return 0.0
        diffs = vecs[1:] - vecs[:-1]
        step_norms = np.linalg.norm(diffs, axis=1)
        return float(np.sum(step_norms))

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个 1D 向量的余弦相似度 (-1.0 ~ 1.0)"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        dot = float(np.dot(vec1, vec2))
        sim = dot / (norm1 * norm2)
        return max(-1.0, min(1.0, sim))

    @classmethod
    def batch_cosine_similarity(cls, query_vecs: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """
        批量矩阵点乘计算余弦相似度矩阵
        query_vecs: (M, D), doc_vecs: (N, D) -> (M, N) 相似度矩阵
        """
        if query_vecs.size == 0 or doc_vecs.size == 0:
            return np.empty((0, 0), dtype=np.float32)

        q_norm = np.linalg.norm(query_vecs, axis=1, keepdims=True)
        d_norm = np.linalg.norm(doc_vecs, axis=1, keepdims=True)

        q_norm = np.where(q_norm < 1e-6, 1.0, q_norm)
        d_norm = np.where(d_norm < 1e-6, 1.0, d_norm)

        q_normed = query_vecs / q_norm
        d_normed = doc_vecs / d_norm

        sim_matrix = np.dot(q_normed, d_normed.T)
        return np.clip(sim_matrix, -1.0, 1.0)

    def compute_similarity(
        self,
        query_vec: Union[np.ndarray, List[float]],
        doc_vecs: Union[np.ndarray, List[np.ndarray], List[List[float]]],
    ) -> np.ndarray:
        """计算单条 query 向量与多条 doc 向量之间的余弦相似度数组 (1D float32 array)"""
        q = np.asarray(query_vec, dtype=np.float32)
        if q.ndim == 1:
            q = q[np.newaxis, :]

        if isinstance(doc_vecs, list):
            if not doc_vecs:
                return np.empty((0,), dtype=np.float32)
            d = np.vstack([np.asarray(x, dtype=np.float32) for x in doc_vecs])
        else:
            d = np.asarray(doc_vecs, dtype=np.float32)
            if d.ndim == 1:
                d = d[np.newaxis, :]

        sim_matrix = self.batch_cosine_similarity(q, d)
        if sim_matrix.shape[0] > 0:
            return sim_matrix[0]
        return np.empty((0,), dtype=np.float32)

    def find_most_similar(
        self, query: str, candidates: List[str], top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """从候选列表中查询与 query 最相似的前 top_k 项 (index, similarity)"""
        if not candidates:
            return []

        q_vec = self.embed_text(query)
        cand_vecs = self.embed_texts(candidates)
        sims = self.batch_cosine_similarity(q_vec[np.newaxis, :], cand_vecs)[0]

        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(int(idx), float(sims[idx])) for idx in top_indices]


_global_embedding_engine: Optional[EmbeddingEngine] = None


def get_embedding_engine(settings: Optional[Settings] = None) -> EmbeddingEngine:
    """获取全局单例向量计算引擎"""
    global _global_embedding_engine
    if _global_embedding_engine is None:
        _global_embedding_engine = EmbeddingEngine(settings)
    return _global_embedding_engine
