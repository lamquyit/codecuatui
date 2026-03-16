import asyncio
import os
import sys
import threading
import time
from pathlib import Path
from dotenv import load_dotenv

# Ensure local RAG-Anything package is importable
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "RAG-Anything"))

from raganything import RAGAnything, RAGAnythingConfig
print(f"DEBUG: Loaded RAGAnything from {RAGAnything.__module__} at {sys.modules['raganything'].__file__}")
from lightrag.utils import EmbeddingFunc
from llm_provider import openai_complete_with_fallback, openai_embed_with_fallback, get_providers

# Metrics & Logging imports
from logging_util import setup_memory_logger, get_logs, parse_document_metrics, cleanup_logger
from processed_util import add_processed_file

load_dotenv()

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_rerank_tokenizer = None
_rerank_model = None

def init_reranker():
    global _rerank_tokenizer, _rerank_model
    if _rerank_tokenizer is None:
        print("Loading local Qwen3-Reranker-0.6B model...")
        model_name = "Qwen/Qwen3-Reranker-0.6B"
        _rerank_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if _rerank_tokenizer.pad_token is None:
            _rerank_tokenizer.pad_token = _rerank_tokenizer.eos_token
        _rerank_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        _rerank_model.eval()
        print(f"Reranker loaded on: {_rerank_model.device}")

async def local_qwen_rerank(query: str, documents: list[str], **kwargs) -> list[dict]:
    init_reranker()
    pairs = [[query, doc] for doc in documents]
    scores_list = []
    with torch.no_grad():
        for pair in pairs:
            inputs = _rerank_tokenizer([pair], padding=False, truncation=True, return_tensors='pt', max_length=1024).to(_rerank_model.device)
            score = _rerank_model(**inputs, return_dict=True).logits.view(-1,).float()
            scores_list.append(score)
        
        scores = torch.cat(scores_list)
        scores = torch.sigmoid(scores).cpu().numpy()
        
    results = []
    for idx, float_score in enumerate(scores):
        results.append({
            "index": idx,
            "relevance_score": float(float_score)
        })
    return results

class BackgroundLoop:
    """
    vai trò : chạy background thread có chứa asyncio event loop
    class này giúp chạy các tác vụ nặng ở dưới nền mà không làm treo giao diện UI
    """
    def __init__(self):
        self.loop = None
        self.thread = None

    def start(self):
        if self.loop is not None:
            return
        self.loop = asyncio.new_event_loop()
        def _run():
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def run_coro(self, coro):
        """Submit a coroutine to the background loop and wait for result."""
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return fut.result()


class RAGFactory:
    """
    Factory Pattern: Responsible for creating and configuring the RAGAnything instance.
    Decouples configuration/creation logic from usage.
    vai trò : tạo ra RAGAnything instance và cấu hình các tham số cần thiết
    """
    @staticmethod
    def create_rag_sync(root_path: Path) -> RAGAnything:
        # Validate: cần ít nhất 1 API key
        providers = get_providers()  # raises RuntimeError nếu không có key nào
        primary = providers[0]
        print(f"🔑 Primary LLM provider: {primary.name} (fallback: {', '.join(p.name for p in providers[1:])} )")

        import os
        config = RAGAnythingConfig(
            working_dir=str(root_path / "rag_storage"),
            parser=os.getenv("RAG_PARSER", "mineru"),
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        def llm_model_func(
            prompt, 
            system_prompt=None, 
            history_messages=None, 
            **kwargs
        ):
            # Increase max_tokens to prevent JSON truncation in large responses
            if "max_tokens" not in kwargs:
                kwargs["max_tokens"] = 4096
                
            history_messages = history_messages or []
            return openai_complete_with_fallback(
                model_type="llm",
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )

        def vision_model_func(
            prompt, 
            system_prompt=None, 
            history_messages=None, 
            image_data=None, 
            messages=None, 
            **kwargs
        ):
            # Increase max_tokens to prevent JSON truncation in large responses
            if "max_tokens" not in kwargs:
                kwargs["max_tokens"] = 4096
                
            history_messages = history_messages or []
            if messages:
                return openai_complete_with_fallback(
                    model_type="vision",
                    prompt="",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    **kwargs,
                )
            elif image_data:
                built_messages = []
                if system_prompt:
                    built_messages.append({"role": "system", "content": system_prompt})
                
                built_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                    ],
                })
                
                return openai_complete_with_fallback(
                    model_type="vision",
                    prompt="",
                    system_prompt=None,
                    history_messages=[],
                    messages=built_messages,
                    **kwargs,
                )
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        embedding_func = EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=lambda texts: openai_embed_with_fallback(texts),
        )

        return RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
            lightrag_kwargs={
                "rerank_model_func": local_qwen_rerank
            }
        )


class RAGService:
    """
    Service Pattern: Facade for interacting with the RAG system.
    Manages the background loop and async execution, exposing synchronous methods to the UI.
    """
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.bg_loop = BackgroundLoop()
        self.bg_loop.start()
        self.rag = None # Lazy initialization

    def initialize(self):
        """Initializes the RAG instance if not already done."""
        if self.rag is None:
            # Run initialization on background thread to be safe with async calls inside
            self.rag = self.bg_loop.run_coro(self._build_and_load())
        return self.rag

    async def _build_and_load(self):
        rag = RAGFactory.create_rag_sync(self.root_path)
        # Force RAGAnything to load existing LightRAG index
        await rag._ensure_lightrag_initialized()
        return rag

    def process_document(self, file_path: str, output_dir: str, role_ids: list[str] = None, company_id: str = None):
        """Sync wrapper for processing a document với thu thập stdout/stderr."""
        self.initialize()
        
        # 1. Setup stdout capture using TeeIO
        import sys
        from io import StringIO
        
        class TeeIO:
            """Captures stdout while also printing to console."""
            def __init__(self, original):
                self.original = original
                self.buffer = StringIO()
            
            def write(self, text):
                self.original.write(text)
                self.buffer.write(text)
            
            def flush(self):
                self.original.flush()
            
            def getvalue(self):
                return self.buffer.getvalue()
        
        tee_stdout = TeeIO(sys.stdout)
        tee_stderr = TeeIO(sys.stderr)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = tee_stdout, tee_stderr
        
        start_time = time.perf_counter()
        try:
            # 2. Run processing
            result = self.bg_loop.run_coro(
                self.rag.process_document_complete(
                    file_path=file_path,
                    output_dir=output_dir,
                    parse_method="auto",
                    role_ids=role_ids,
                    company_id=company_id,
                )
            )
        finally:
            # Restore stdout/stderr
            sys.stdout, sys.stderr = old_stdout, old_stderr
            elapsed = time.perf_counter() - start_time
            
            # 3. Get Stats directly from RAG instance (primary source)
            try:
                # Retrieve explicit stats tracked inside RAGAnything
                direct_stats = {}
                if hasattr(self.rag, "get_last_processing_stats"):
                    direct_stats = self.rag.get_last_processing_stats()
                    print(f"[DEBUG] Direct stats: {direct_stats}")

                logs = tee_stdout.getvalue() + "\n" + tee_stderr.getvalue()
                
                # Still parse logs for backup/verification
                parsed_metrics = parse_document_metrics(logs)
                
                # Merge: Direct stats override parsed stats if available and non-zero
                metrics = parsed_metrics.copy()
                
                if direct_stats:
                    if direct_stats.get("mineru_parsing_time"):
                        metrics["mineru_parsing_time"] = direct_stats["mineru_parsing_time"]
                    if direct_stats.get("content_analysis_time"):
                        metrics["content_analysis_time"] = direct_stats["content_analysis_time"]
                    if direct_stats.get("multimodal_processing_time"):
                        metrics["multimodal_processing_time"] = direct_stats["multimodal_processing_time"]
                    if direct_stats.get("multimodal_description_time"):
                        metrics["multimodal_description_time"] = direct_stats["multimodal_description_time"]
                    if direct_stats.get("multimodal_embedding_time"):
                        metrics["multimodal_embedding_time"] = direct_stats["multimodal_embedding_time"]
                    if direct_stats.get("multimodal_extraction_time"):
                        metrics["multimodal_extraction_time"] = direct_stats["multimodal_extraction_time"]
                    if direct_stats.get("crossmodal_graph_merging_time"):
                        metrics["crossmodal_graph_merging_time"] = direct_stats["crossmodal_graph_merging_time"]
                    if direct_stats.get("text_indexing_time"):
                        metrics["text_indexing_time"] = direct_stats["text_indexing_time"]
                    
                    # Merge RAG stats (chunks count is reliable from direct stats)
                    if direct_stats.get("rag_stats"):
                         if "rag_stats" not in metrics: metrics["rag_stats"] = {}
                         metrics["rag_stats"]["chunks"] = direct_stats["rag_stats"].get("chunks", metrics["rag_stats"].get("chunks", 0))

                
                # DEBUG: Dump captured logs to verify content
                print(f"[DEBUG] Captured output size: {len(logs)} chars")
                print(f"[DEBUG] Total Processing Time: {elapsed:.3f}s")
                with open("debug_last_log.txt", "w", encoding="utf-8") as f:
                    f.write(logs)
                    
                print(f"[DEBUG] Final metrics: {metrics}")
                
                # 4. Save metadata with metrics
                add_processed_file(file_path, elapsed, metrics)
                
                # 5. Print Formatted Report to Console
                from processed_util import format_single_file_report
                
                # Construct temporary file_data for report formatting
                temp_file_data = {
                    "filename": Path(file_path).name,
                    "processing_time_sec": round(elapsed, 2),
                    "phase_stats": {}
                }
                
                # Flatten metrics for report
                # (metrics structure: key -> {stage, duration_sec})
                # We need simple key -> duration in seconds
                t_stats = temp_file_data["phase_stats"]
                
                # Mapping known keys
                if metrics.get("mineru_parsing_time"): t_stats["mineru_parsing"] = metrics["mineru_parsing_time"].get("duration_sec", 0)
                if metrics.get("content_analysis_time"): t_stats["content_analysis"] = metrics["content_analysis_time"].get("duration_sec", 0)
                if metrics.get("multimodal_processing_time"): t_stats["multimodal_processing"] = metrics["multimodal_processing_time"].get("duration_sec", 0)
                if metrics.get("multimodal_description_time"): t_stats["multimodal_description"] = metrics["multimodal_description_time"].get("duration_sec", 0)
                if metrics.get("multimodal_embedding_time"): t_stats["multimodal_embedding"] = metrics["multimodal_embedding_time"].get("duration_sec", 0)
                if metrics.get("multimodal_extraction_time"): t_stats["multimodal_extraction"] = metrics["multimodal_extraction_time"].get("duration_sec", 0)
                if metrics.get("crossmodal_graph_merging_time"): t_stats["crossmodal_graph_merging"] = metrics["crossmodal_graph_merging_time"].get("duration_sec", 0)
                if metrics.get("text_indexing_time"): t_stats["text_indexing"] = metrics["text_indexing_time"].get("duration_sec", 0)
                # Calculate Overhead for report
                
                # Report formatting
                report_str = format_single_file_report(temp_file_data)
                print(f"\n{report_str}\n")
                
            except Exception as e:
                print(f"Error retrieving/parsing metrics: {e}")
                import traceback
                traceback.print_exc()
            
        return result

    async def aquery(self, question: str, mode: str = "hybrid", user_roles: list[str] = None, company_id: str = None):
        return await self.rag.aquery(question, mode=mode, user_roles=user_roles, company_id=company_id)

    def query(self, question: str, mode: str = "hybrid", user_roles: list[str] = None, company_id: str = None):
        """Sync wrapper for querying."""
        self.initialize()
        return self.bg_loop.run_coro(
            self.rag.aquery(question, mode=mode, user_roles=user_roles, company_id=company_id)
        )

    def delete_document(self, doc_id: str) -> dict:
        """
        Xóa toàn bộ dữ liệu liên quan đến một document khỏi RAG storage.

        Args:
            doc_id: MD5 hash của file (được trả về bởi /index-doc).

        Returns:
            dict với các stats về số lượng records đã xóa.
        """
        import json
        import shutil
        import xml.etree.ElementTree as ET

        storage_dir = self.root_path / "rag_storage"
        processed_path = self.root_path / "processed_files.json"

        def load_json(p):
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

        def save_json(p, d):
            p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

        # ── Tìm full_doc_id trong storage ────────────────────────────────────
        chunks_path = storage_dir / "kv_store_text_chunks.json"
        chunks_data = load_json(chunks_path) or {}

        # full_doc_id có thể là "doc-<md5>" hoặc khớp trực tiếp
        full_doc_ids_to_remove = set()
        for _, chunk in chunks_data.items():
            if not isinstance(chunk, dict):
                continue
            fdid = chunk.get("full_doc_id", "")
            fp = chunk.get("file_path", "")
            # Match by doc_id embedded in full_doc_id, or match by file lookup
            if doc_id in fdid or fdid == f"doc-{doc_id}":
                full_doc_ids_to_remove.add(fdid)

        if not full_doc_ids_to_remove:
            return {
                "doc_id": doc_id,
                "status": "not_found",
                "message": f"doc_id '{doc_id}' not found in storage.",
            }

        stats = {
            "chunks_removed": 0,
            "vdb_chunks_removed": 0,
            "entities_removed": 0,
            "entities_updated": 0,
            "relations_removed": 0,
            "relations_updated": 0,
            "vdb_entities_removed": 0,
            "vdb_relations_removed": 0,
            "graph_nodes_removed": 0,
            "graph_edges_removed": 0,
        }

        # ── 1. Text chunks ────────────────────────────────────────────────────
        chunk_ids = {
            cid for cid, c in chunks_data.items()
            if isinstance(c, dict) and c.get("full_doc_id") in full_doc_ids_to_remove
        }
        for cid in chunk_ids:
            chunks_data.pop(cid)
        save_json(chunks_path, chunks_data)
        stats["chunks_removed"] = len(chunk_ids)

        # ── 2. vdb_chunks (matrix-aware) ──────────────────────────────────────
        def _remove_vdb_entries(vdb_path, ids_to_remove):
            """Remove entries from a NanoVectorDB file, updating both data and matrix."""
            import numpy as np
            import base64
            d = load_json(vdb_path)
            if not d:
                return 0
            old_data = d.get("data", [])
            old_ids = [e.get("__id__") for e in old_data]
            keep_mask = [eid not in ids_to_remove for eid in old_ids]
            new_data = [e for e, keep in zip(old_data, keep_mask) if keep]
            removed = len(old_data) - len(new_data)
            if removed == 0:
                return 0
            d["data"] = new_data
            # Rebuild matrix by slicing rows
            emb_dim = d.get("embedding_dim", 3072)
            mat_b64 = d.get("matrix", "")
            if mat_b64:
                try:
                    raw = base64.b64decode(mat_b64)
                    n_total = len(raw) // (emb_dim * 4)
                    if n_total == len(old_data):  # matrix is in sync with old data
                        mat = np.frombuffer(raw, dtype=np.float32).reshape(n_total, emb_dim)
                        keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
                        new_mat = mat[keep_indices]
                        d["matrix"] = base64.b64encode(new_mat.tobytes()).decode("ascii")
                    else:
                        print(f"[WARN] {vdb_path.name}: matrix rows ({n_total}) != data len ({len(old_data)}), skipping matrix update")
                except Exception as ex:
                    print(f"[WARN] {vdb_path.name}: matrix update failed: {ex}")
            save_json(vdb_path, d)
            return removed

        p = storage_dir / "vdb_chunks.json"
        stats["vdb_chunks_removed"] = _remove_vdb_entries(p, chunk_ids)

        # ── 3. Entities ───────────────────────────────────────────────────────
        p = storage_dir / "kv_store_full_entities.json"
        d = load_json(p) or {}
        removed_ent = set()
        for eid, e in list(d.items()):
            if not isinstance(e, dict):
                continue
            srcs = set(e.get("source_id", "").split("<SEP>"))
            remaining = srcs - chunk_ids
            if not remaining or remaining == {""}:
                d.pop(eid)
                removed_ent.add(eid)
                stats["entities_removed"] += 1
            elif srcs & chunk_ids:
                e["source_id"] = "<SEP>".join(remaining)
                stats["entities_updated"] += 1
        save_json(p, d)

        # ── 4. Relations ──────────────────────────────────────────────────────
        p = storage_dir / "kv_store_full_relations.json"
        d = load_json(p) or {}
        removed_rel = set()
        for rid, r in list(d.items()):
            if not isinstance(r, dict):
                continue
            srcs = set(r.get("source_id", "").split("<SEP>"))
            remaining = srcs - chunk_ids
            if not remaining or remaining == {""}:
                d.pop(rid)
                removed_rel.add(rid)
                stats["relations_removed"] += 1
            elif srcs & chunk_ids:
                r["source_id"] = "<SEP>".join(remaining)
                stats["relations_updated"] += 1
        save_json(p, d)

        # ── 5. vdb_entities ───────────────────────────────────────────────────
        if removed_ent:
            p = storage_dir / "vdb_entities.json"
            stats["vdb_entities_removed"] = _remove_vdb_entries(p, removed_ent)

        # ── 6. vdb_relationships ──────────────────────────────────────────────
        if removed_rel:
            p = storage_dir / "vdb_relationships.json"
            stats["vdb_relations_removed"] = _remove_vdb_entries(p, removed_rel)

        # ── 7. GraphML ────────────────────────────────────────────────────────
        gp = storage_dir / "graph_chunk_entity_relation.graphml"
        if gp.exists():
            try:
                ET.register_namespace("", "http://graphml.graphdrawing.org/graphml")
                tree = ET.parse(str(gp))
                root = tree.getroot()
                # Find graph element regardless of namespace
                graph = None
                for child in root:
                    if child.tag.endswith("}graph") or child.tag == "graph":
                        graph = child
                        break
                if graph is not None:
                    bad_ids = chunk_ids | removed_ent | removed_rel
                    nodes = [n for n in list(graph) if n.tag.endswith("}node") or n.tag == "node"
                             if n.get("id", "") in bad_ids]
                    edges = [e for e in list(graph) if e.tag.endswith("}edge") or e.tag == "edge"
                             if e.get("source", "") in bad_ids or e.get("target", "") in bad_ids]
                    for n in nodes:
                        graph.remove(n)
                    for e in edges:
                        graph.remove(e)
                    tree.write(str(gp), encoding="utf-8", xml_declaration=True)
                    stats["graph_nodes_removed"] = len(nodes)
                    stats["graph_edges_removed"] = len(edges)
            except Exception as ex:
                print(f"[WARN] GraphML update failed: {ex}")

        # ── 8. processed_files.json ───────────────────────────────────────────
        pf = load_json(processed_path)
        if pf and isinstance(pf, dict):
            before = len(pf.get("files", []))
            pf["files"] = [f for f in pf.get("files", []) if f.get("doc_id") != doc_id]
            if len(pf["files"]) < before:
                save_json(processed_path, pf)

        # ── Reload RAG instance để đồng bộ in-memory state ───────────────────
        # Reset rag instance để lần query tiếp theo sẽ load lại từ storage đã sạch
        self.rag = None

        return {
            "doc_id": doc_id,
            "status": "deleted",
            "full_doc_ids": list(full_doc_ids_to_remove),
            "stats": stats,
        }

