"""
Index evaluation dataset subsets directly into LightRAG's knowledge graph.
Bypasses the document parser (mineru/docling) by using LightRAG's insert() API.

Usage:
    python index_eval_subset.py [--hotpot N] [--mmlongbench N]
    
Examples:
    python index_eval_subset.py --hotpot 100 --mmlongbench 100
    python index_eval_subset.py --hotpot 50
"""

import json
import os
import argparse
import time
import asyncio
from tqdm import tqdm


def get_lightrag_instance():
    """Get the inner LightRAG instance from RAGService -> RAGAnything -> LightRAG."""
    from rag_bridge import get_rag_service
    svc = get_rag_service()
    svc.initialize()
    # Access: RAGService.rag (RAGAnything) -> RAGAnything.lightrag (LightRAG)
    lightrag = svc.rag.lightrag
    return lightrag, svc


def index_hotpot(lightrag, num_samples=100):
    """Extract context passages from HotpotQA and insert them as text into LightRAG."""
    input_file = os.path.join(os.path.dirname(__file__), "data", "hotpot", "hotpot_dev_distractor_20.json")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n📚 Loaded {len(data)} items from Hotpot. Indexing first {num_samples}...")
    data = data[:num_samples]

    # Collect all unique context passages across all questions
    all_texts = []
    all_ids = []
    seen_titles = set()

    for item in data:
        for title, sentences in item["context"]:
            if title in seen_titles:
                continue
            seen_titles.add(title)

            text_content = f"# {title}\n\n" + " ".join(sentences)
            all_texts.append(text_content)
            all_ids.append(f"hotpot_{title.replace(' ', '_')[:80]}")

    print(f"   Found {len(all_texts)} unique context passages to index.")

    # Insert in batches for efficiency
    batch_size = 20
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Indexing Hotpot"):
        batch_texts = all_texts[i:i+batch_size]
        batch_ids = all_ids[i:i+batch_size]
        try:
            lightrag.insert(batch_texts, ids=batch_ids)
        except Exception as e:
            print(f"\n⚠️ Error indexing batch {i//batch_size + 1}: {e}")
            # Try one-by-one for failed batch
            for j, (text, doc_id) in enumerate(zip(batch_texts, batch_ids)):
                try:
                    lightrag.insert(text, ids=doc_id)
                except Exception as e2:
                    print(f"  ❌ Skipping doc {doc_id}: {e2}")

    print(f"✅ Hotpot indexing complete. {len(all_texts)} passages indexed.")


def index_mmlongbench(lightrag, num_samples=100):
    """Extract text from MMLongBench PDFs using PyPDF2 and insert into LightRAG."""
    input_file = os.path.join(os.path.dirname(__file__), "data", "MMLongBench-Doc", "data", "samples_20.json")
    doc_dir = os.path.join(os.path.dirname(__file__), "data", "MMLongBench-Doc", "data", "documents")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n📚 Loaded {len(data)} items from MMLongBench. Processing first {num_samples}...")
    data = data[:num_samples]

    # Get unique PDF documents
    unique_docs = list(dict.fromkeys(item['doc_id'] for item in data))
    print(f"   Found {len(unique_docs)} unique PDF documents to index.")

    # Try to import a PDF reader
    try:
        import PyPDF2
        pdf_reader_available = True
    except ImportError:
        try:
            import fitz  # PyMuPDF
            pdf_reader_available = "pymupdf"
        except ImportError:
            pdf_reader_available = False
            print("⚠️ Neither PyPDF2 nor PyMuPDF installed. Trying pdfplumber...")
            try:
                import pdfplumber
                pdf_reader_available = "pdfplumber"
            except ImportError:
                print("❌ No PDF reader available. Install one: pip install PyPDF2")
                return

    indexed_count = 0
    for doc_name in tqdm(unique_docs, desc="Indexing MMLongBench"):
        pdf_path = os.path.join(doc_dir, doc_name)
        if not os.path.exists(pdf_path):
            print(f"\n⚠️ PDF not found: {doc_name}")
            continue

        try:
            text = ""
            if pdf_reader_available == True:  # PyPDF2
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
            elif pdf_reader_available == "pymupdf":
                doc = fitz.open(pdf_path)
                for page in doc:
                    text += page.get_text() + "\n\n"
                doc.close()
            elif pdf_reader_available == "pdfplumber":
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"

            if not text.strip():
                print(f"\n⚠️ No text extracted from {doc_name} (might be image-only PDF)")
                continue

            # Truncate very long documents to avoid token limits
            max_chars = 50000
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[... document truncated ...]"

            doc_id = f"mmlb_{doc_name.replace('.pdf', '').replace(' ', '_')[:80]}"
            lightrag.insert(text, ids=doc_id, file_paths=doc_name)
            indexed_count += 1

        except Exception as e:
            print(f"\n⚠️ Error processing {doc_name}: {e}")

    print(f"✅ MMLongBench indexing complete. {indexed_count}/{len(unique_docs)} PDFs indexed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index evaluation dataset subsets into LightRAG")
    parser.add_argument("--hotpot", type=int, default=0, help="Number of Hotpot samples to index (0=skip)")
    parser.add_argument("--mmlongbench", type=int, default=0, help="Number of MMLongBench samples to index (0=skip)")
    args = parser.parse_args()

    if args.hotpot == 0 and args.mmlongbench == 0:
        print("No dataset specified. Use --hotpot N and/or --mmlongbench N")
        print("Example: python index_eval_subset.py --hotpot 100 --mmlongbench 100")
        exit(1)

    start = time.time()
    lightrag, svc = get_lightrag_instance()

    if args.hotpot > 0:
        index_hotpot(lightrag, args.hotpot)

    if args.mmlongbench > 0:
        index_mmlongbench(lightrag, args.mmlongbench)

    elapsed = time.time() - start
    print(f"\n⏱️  Total indexing time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("🎉 Done! You can now run: python baseline_eval.py --dataset hotpot")
