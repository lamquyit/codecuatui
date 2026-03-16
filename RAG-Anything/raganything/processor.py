"""
Document processing functionality for RAGAnything

Contains methods for parsing documents and processing multimodal content
"""

import os
import time
import hashlib
import json
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from raganything.base import DocStatus
from raganything.parser import MineruParser, DoclingParser, MineruExecutionError
from raganything.utils import (
    separate_content,
    insert_text_content,
    insert_text_content_with_multimodal_content,
    get_processor_for_type,
    convert_heic_to_png,
)
import asyncio
from lightrag.utils import compute_mdhash_id
from raganything.handwriting_router import HandwritingRouter


class ProcessorMixin:
    """ProcessorMixin class containing document processing functionality for RAGAnything"""

    def _get_file_reference(self, file_path: str) -> str:
        """
        Get file reference based on use_full_path configuration.
        """
        # (Original code remains)
        if self.config.use_full_path:
            return str(file_path)
        else:
            return os.path.basename(file_path)

    def _process_handwriting_detection(self, content_list: List[Dict[str, Any]], file_path: Path):
        """
        Scan text blocks and reclassify handwriting using HandwritingRouter.
        """
        if not getattr(self.config, "enable_handwriting_processing", False):
            return

        self.logger.info("Starting handwriting detection for text blocks...")
        router = HandwritingRouter(self.config)
        
        # Load original image once for cropping
        from PIL import Image
        try:
            original_image = Image.open(file_path)
        except Exception as e:
            self.logger.warning(f"Could not open image for handwriting detection: {e}")
            return

        count_reclassified = 0
        for block in content_list:
            if block.get("type") == "text" and "bbox" in block:
                try:
                    # Mineru bbox is [x0, y0, x1, y1] (usually normalized 1000 or absolute pixels depending on version)
                    # Assuming Mineru 2.0 returns coordinates relative to original image size or needs scaling
                    # For safety, let's wrap this in a try-catch and assume pixel coordinates if integers > 1
                    
                    bbox = block["bbox"]
                    # Simple validation
                    if len(bbox) == 4:
                        crop = original_image.crop(bbox)
                        if router.is_handwriting(crop):
                            block["type"] = "handwriting" # Change type to handwriting
                            # Optionally, overwrite text content if Vintern already OCR-ed it in some flow
                            # But here we just reclassify so the correct ModalProcessor picks it up later
                            count_reclassified += 1
                except Exception as e:
                    self.logger.debug(f"Failed to process block for handwriting: {e}")
                    continue
                    
        self.logger.info(f"Handwriting detection complete. Reclassified {count_reclassified} blocks.")

    def _generate_cache_key(
        self, file_path: Path, parse_method: str = None, **kwargs
    ) -> str:
        """
        Generate cache key based on file path and parsing configuration

        Args:
            file_path: Path to the file
            parse_method: Parse method used
            **kwargs: Additional parser parameters

        Returns:
            str: Cache key for the file and configuration
        """

        # Get file modification time
        mtime = file_path.stat().st_mtime

        # Create configuration dict for cache key
        config_dict = {
            "file_path": str(file_path.absolute()),
            "mtime": mtime,
            "parser": self.config.parser,
            "parse_method": parse_method or self.config.parse_method,
        }

        # Add relevant kwargs to config
        relevant_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "lang",
                "device",
                "start_page",
                "end_page",
                "formula",
                "table",
                "backend",
                "source",
            ]
        }
        config_dict.update(relevant_kwargs)

        # Generate hash from config
        config_str = json.dumps(config_dict, sort_keys=True)
        cache_key = hashlib.md5(config_str.encode()).hexdigest()

        return cache_key

    def _generate_content_based_doc_id(self, content_list: List[Dict[str, Any]]) -> str:
        """
        Generate doc_id based on document content

        Args:
            content_list: Parsed content list

        Returns:
            str: Content-based document ID with doc- prefix
        """
        from lightrag.utils import compute_mdhash_id

        # Extract key content for ID generation
        content_hash_data = []

        for item in content_list:
            if isinstance(item, dict):
                # For text content, use the text
                if item.get("type") == "text" and item.get("text"):
                    content_hash_data.append(item["text"].strip())
                # For other content types, use key identifiers
                elif item.get("type") == "image" and item.get("img_path"):
                    content_hash_data.append(f"image:{item['img_path']}")
                elif item.get("type") == "table" and item.get("table_body"):
                    content_hash_data.append(f"table:{item['table_body']}")
                elif item.get("type") == "equation" and item.get("text"):
                    content_hash_data.append(f"equation:{item['text']}")
                else:
                    # For other types, use string representation
                    content_hash_data.append(str(item))

        # Create a content signature
        content_signature = "\n".join(content_hash_data)

        # Generate doc_id from content
        doc_id = compute_mdhash_id(content_signature, prefix="doc-")

        return doc_id

    async def _get_cached_result(
        self, cache_key: str, file_path: Path, parse_method: str = None, **kwargs
    ) -> tuple[List[Dict[str, Any]], str] | None:
        """
        Get cached parsing result if available and valid

        Args:
            cache_key: Cache key to look up
            file_path: Path to the file for mtime check
            parse_method: Parse method used
            **kwargs: Additional parser parameters

        Returns:
            tuple[List[Dict[str, Any]], str] | None: (content_list, doc_id) or None if not found/invalid
        """
        if not hasattr(self, "parse_cache") or self.parse_cache is None:
            return None

        try:
            cached_data = await self.parse_cache.get_by_id(cache_key)
            if not cached_data:
                return None

            # Check file modification time
            current_mtime = file_path.stat().st_mtime
            cached_mtime = cached_data.get("mtime", 0)

            if current_mtime != cached_mtime:
                self.logger.debug(f"Cache invalid - file modified: {cache_key}")
                return None

            # Check parsing configuration
            cached_config = cached_data.get("parse_config", {})
            current_config = {
                "parser": self.config.parser,
                "parse_method": parse_method or self.config.parse_method,
            }

            # Add relevant kwargs to current config
            relevant_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in [
                    "lang",
                    "device",
                    "start_page",
                    "end_page",
                    "formula",
                    "table",
                    "backend",
                    "source",
                ]
            }
            current_config.update(relevant_kwargs)

            if cached_config != current_config:
                self.logger.debug(f"Cache invalid - config changed: {cache_key}")
                return None

            content_list = cached_data.get("content_list", [])
            doc_id = cached_data.get("doc_id")

            if content_list and doc_id:
                self.logger.debug(
                    f"Found valid cached parsing result for key: {cache_key}"
                )
                return content_list, doc_id
            else:
                self.logger.debug(
                    f"Cache incomplete - missing content or doc_id: {cache_key}"
                )
                return None

        except Exception as e:
            self.logger.warning(f"Error accessing parse cache: {e}")

        return None

    async def _store_cached_result(
        self,
        cache_key: str,
        content_list: List[Dict[str, Any]],
        doc_id: str,
        file_path: Path,
        parse_method: str = None,
        **kwargs,
    ) -> None:
        """
        Store parsing result in cache

        Args:
            cache_key: Cache key to store under
            content_list: Content list to cache
            doc_id: Content-based document ID
            file_path: Path to the file for mtime storage
            parse_method: Parse method used
            **kwargs: Additional parser parameters
        """
        if not hasattr(self, "parse_cache") or self.parse_cache is None:
            return

        try:
            # Get file modification time
            file_mtime = file_path.stat().st_mtime

            # Create parsing configuration
            parse_config = {
                "parser": self.config.parser,
                "parse_method": parse_method or self.config.parse_method,
            }

            # Add relevant kwargs to config
            relevant_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in [
                    "lang",
                    "device",
                    "start_page",
                    "end_page",
                    "formula",
                    "table",
                    "backend",
                    "source",
                ]
            }
            parse_config.update(relevant_kwargs)

            cache_data = {
                cache_key: {
                    "content_list": content_list,
                    "doc_id": doc_id,
                    "mtime": file_mtime,
                    "parse_config": parse_config,
                    "cached_at": time.time(),
                    "cache_version": "1.0",
                }
            }
            await self.parse_cache.upsert(cache_data)
            # Ensure data is persisted to disk
            await self.parse_cache.index_done_callback()
            self.logger.info(f"Stored parsing result in cache: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Error storing to parse cache: {e}")

    async def parse_document(
        self,
        file_path: str,
        output_dir: str = None,
        parse_method: str = None,
        display_stats: bool = None,
        **kwargs,
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        Parse document with caching support

        Args:
            file_path: Path to the file to parse
            output_dir: Output directory (defaults to config.parser_output_dir)
            parse_method: Parse method (defaults to config.parse_method)
            display_stats: Whether to display content statistics (defaults to config.display_content_stats)
            **kwargs: Additional parameters for parser (e.g., lang, device, start_page, end_page, formula, table, backend, source)

        Returns:
            tuple[List[Dict[str, Any]], str]: (content_list, doc_id)
        """
        # Use config defaults if not provided
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if display_stats is None:
            display_stats = self.config.display_content_stats

        self.logger.info(f"Starting document parsing: {file_path}")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Generate cache key based on file and configuration
        cache_key = self._generate_cache_key(file_path, parse_method, **kwargs)

        # Check cache first
        cached_result = await self._get_cached_result(
            cache_key, file_path, parse_method, **kwargs
        )
        if cached_result is not None:
            content_list, doc_id = cached_result
            self.logger.info(f"Using cached parsing result for: {file_path}")
            if display_stats:
                self.logger.info(
                    f"* Total blocks in cached content_list: {len(content_list)}"
                )
            return content_list, doc_id

        # Choose appropriate parsing method based on file extension
        ext = file_path.suffix.lower()

        try:
            doc_parser = (
                DoclingParser() if self.config.parser == "docling" else MineruParser()
            )

            # Log parser and method information
            self.logger.info(
                f"Using {self.config.parser} parser with method: {parse_method}"
            )

            if ext in [".pdf"]:
                self.logger.info("Detected PDF file, using parser for PDF...")
                content_list = await asyncio.to_thread(
                    doc_parser.parse_pdf,
                    pdf_path=file_path,
                    output_dir=output_dir,
                    method=parse_method,
                    **kwargs,
                )
            elif ext in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tiff",
                ".tif",
                ".gif",
                ".webp",
                ".heic",
            ]:
                self.logger.info("Detected image file, using parser for images...")
                
                # Handle HEIC conversion if needed
                if ext == ".heic":
                    self.logger.info("Converting HEIC to PNG for processing...")
                    png_path = convert_heic_to_png(str(file_path))
                    if png_path:
                        self.logger.info(f"Successfully converted HEIC to: {png_path}")
                        # Update file_path to the converted file
                        file_path = Path(png_path)
                        # Note: we don't update ext here as we've already entered the image block
                    else:
                        raise ValueError(f"Failed to convert HEIC file: {file_path}")

                # Use the selected parser's image parsing capability
                if hasattr(doc_parser, "parse_image"):
                    content_list = await asyncio.to_thread(
                        doc_parser.parse_image,
                        image_path=file_path,
                        output_dir=output_dir,
                        **kwargs,
                    )
                else:
                    # Fallback to MinerU for image parsing if current parser doesn't support it
                    self.logger.warning(
                        f"{self.config.parser} parser doesn't support image parsing, falling back to MinerU"
                    )
                    content_list = MineruParser().parse_image(
                        image_path=file_path, output_dir=output_dir, **kwargs
                    )
            elif ext in [
                ".doc",
                ".docx",
                ".ppt",
                ".pptx",
                ".xls",
                ".xlsx",
                ".html",
                ".htm",
                ".xhtml",
            ]:
                self.logger.info(
                    "Detected Office or HTML document, using parser for Office/HTML..."
                )
                content_list = await asyncio.to_thread(
                    doc_parser.parse_office_doc,
                    doc_path=file_path,
                    output_dir=output_dir,
                    **kwargs,
                )
            else:
                # For other or unknown formats, use generic parser
                self.logger.info(
                    f"Using generic parser for {ext} file (method={parse_method})..."
                )
                content_list = await asyncio.to_thread(
                    doc_parser.parse_document,
                    file_path=file_path,
                    method=parse_method,
                    output_dir=output_dir,
                    **kwargs,
                )

        except MineruExecutionError as e:
            self.logger.error(f"Mineru command failed: {e}")
            raise
        except Exception as e:
            self.logger.error(
                f"Error during parsing with {self.config.parser} parser: {str(e)}"
            )
            raise e

        msg = f"Parsing {file_path} complete! Extracted {len(content_list)} content blocks"
        self.logger.info(msg)

        if len(content_list) == 0:
            raise ValueError("Parsing failed: No content was extracted")

        # [NEW] Post-processing: Handwriting Detection
        # Only meaningful for image-based files where we can crop from source
        if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".png", ".webp"]:
             self._process_handwriting_detection(content_list, file_path)

             # [NEW] Vintern Text Recognition Correction
             # Re-recognize text in bounding boxes using Vintern-1B
             from raganything.vintern_recognizer import VinternRecognizer
             if VinternRecognizer.is_available():
                 content_list, corrected_count = VinternRecognizer.process_content_list(content_list, str(file_path))
                 if corrected_count > 0:
                     self.logger.info(f"Vintern corrected {corrected_count} text regions found by MinerU")

        # Generate doc_id based on content
        doc_id = self._generate_content_based_doc_id(content_list)

        # Store result in cache
        await self._store_cached_result(
            cache_key, content_list, doc_id, file_path, parse_method, **kwargs
        )

        # Display content statistics if requested
        if display_stats:
            self.logger.info("\nContent Information:")
            self.logger.info(f"* Total blocks in content_list: {len(content_list)}")

            # Count elements by type
            block_types: Dict[str, int] = {}
            for block in content_list:
                if isinstance(block, dict):
                    block_type = block.get("type", "unknown")
                    if isinstance(block_type, str):
                        block_types[block_type] = block_types.get(block_type, 0) + 1

            self.logger.info("* Content block types:")
            for block_type, count in block_types.items():
                self.logger.info(f"  - {block_type}: {count}")

        return content_list, doc_id

    async def _process_multimodal_content(
        self,
        multimodal_items: List[Dict[str, Any]],
        file_path: str,
        doc_id: str,
        pipeline_status: Optional[Any] = None,
        pipeline_status_lock: Optional[Any] = None,
        role_ids: List[str] = None,
        company_id: str = None,
        **kwargs,
    ):
        """
        Process multimodal content (using specialized processors)

        Args:
            multimodal_items: List of multimodal items
            file_path: File path (for reference)
            doc_id: Document ID for proper chunk association
            pipeline_status: Pipeline status object
            pipeline_status_lock: Pipeline status lock
            role_ids: List of role IDs for access control
            company_id: Company ID for tenant isolation
        """
        self.logger.info(f"DEBUG: Entering _process_multimodal_content with role_ids: {role_ids}, company_id: {company_id}")

        if not multimodal_items:
            self.logger.debug("No multimodal content to process")
            return

        # Check multimodal processing status - handle LightRAG's early DocStatus.PROCESSED marking
        try:
            existing_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if existing_doc_status:
                # Check if multimodal content is already processed
                multimodal_processed = existing_doc_status.get(
                    "multimodal_processed", False
                )

                if multimodal_processed:
                    self.logger.info(
                        f"Document {doc_id} multimodal content is already processed"
                    )
                    return

                # Even if status is DocStatus.PROCESSED (text processing done),
                # we still need to process multimodal content if not yet done
                doc_status = existing_doc_status.get("status", "")
                if doc_status == DocStatus.PROCESSED and not multimodal_processed:
                    self.logger.info(
                        f"Document {doc_id} text processing is complete, but multimodal content still needs processing"
                    )
                    # Continue with multimodal processing
                elif doc_status == DocStatus.PROCESSED and multimodal_processed:
                    self.logger.info(
                        f"Document {doc_id} is fully processed (text + multimodal)"
                    )
                    return

        except Exception as e:
            self.logger.debug(f"Error checking document status for {doc_id}: {e}")
            # Continue with processing if cache check fails

        # Use ProcessorMixin's own batch processing that can handle multiple content types
        log_message = "Starting multimodal content processing..."
        self.logger.info(log_message)
        if pipeline_status_lock and pipeline_status:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        try:
            # Ensure LightRAG is initialized
            await self._ensure_lightrag_initialized()

            await self._process_multimodal_content_batch_type_aware(
                multimodal_items=multimodal_items, 
                file_path=file_path, 
                doc_id=doc_id, 
                role_ids=role_ids,
                company_id=company_id,
                **kwargs
            )

            # Mark multimodal content as processed and update final status
            await self._mark_multimodal_processing_complete(doc_id)

            log_message = "Multimodal content processing complete"
            self.logger.info(log_message)
            if pipeline_status_lock and pipeline_status:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)

        except Exception as e:
            self.logger.error(f"Error in multimodal processing: {e}")
            # Fallback to individual processing if batch processing fails
            self.logger.warning("Falling back to individual multimodal processing")
            await self._process_multimodal_content_individual(
                multimodal_items=multimodal_items, 
                file_path=file_path, 
                doc_id=doc_id, 
                role_ids=role_ids,
                company_id=company_id,
                **kwargs
            )

            # Mark multimodal content as processed even after fallback
            await self._mark_multimodal_processing_complete(doc_id)

    async def _process_multimodal_content_individual(
        self, multimodal_items: List[Dict[str, Any]], file_path: str, doc_id: str, **kwargs
    ):
        """
        Process multimodal content individually (fallback method)

        Args:
        """
        # Initialize timing accumulators if not already present
        self._last_stats.setdefault("multimodal_description_time", {"stage": "multimodal_description", "duration_sec": 0})
        self._last_stats.setdefault("multimodal_processing_time", {"stage": "multimodal_processing", "duration_sec": 0})
        self._last_stats.setdefault("multimodal_embedding_time", {"stage": "multimodal_embedding", "duration_sec": 0})
        self._last_stats.setdefault("multimodal_extraction_time", {"stage": "multimodal_extraction", "duration_sec": 0})
        # Start total multimodal processing timer
        total_start = time.perf_counter()

        # Use full path or basename based on config
        file_name = self._get_file_reference(file_path)

        # Collect all chunk results for batch processing (similar to text content processing)
        all_chunk_results = []
        multimodal_chunk_ids = []

        # Get current text chunks count to set proper order indexes for multimodal chunks
        existing_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
        existing_chunks_count = (
            existing_doc_status.get("chunks_count", 0) if existing_doc_status else 0
        )

        for i, item in enumerate(multimodal_items):
            try:
                content_type = item.get("type", "unknown")
                self.logger.info(
                    f"Processing item {i+1}/{len(multimodal_items)}: {content_type} content"
                )

                # Select appropriate processor
                processor = get_processor_for_type(self.modal_processors, content_type)

                if processor:
                    # Prepare item info for context extraction
                    item_info = {
                        "page_idx": item.get("page_idx", 0),
                        "index": i,
                        "type": content_type,
                    }

                    # Process content and get chunk results instead of immediately merging
                    # Measure description generation time (includes all sub‑steps inside processor)
                    desc_start = time.perf_counter()
                    (enhanced_caption,
                        entity_info,
                        chunk_results,
                    ) = await processor.process_multimodal_content(
                        modal_content=item,
                        content_type=content_type,
                        file_path=file_name,
                        item_info=item_info,  # Pass item info for context extraction
                        batch_mode=True,
                        doc_id=doc_id,  # Pass doc_id for proper association
                        chunk_order_index=existing_chunks_count + i,  # Proper order index
                        role_ids=kwargs.get("role_ids"),
                        company_id=kwargs.get("company_id"),
                    )
                    desc_duration = time.perf_counter() - desc_start
                    # Accumulate description time
                    self._last_stats["multimodal_description_time"]["duration_sec"] += desc_duration
                    # Approximate embedding and extraction times as same as description (since they are done inside the same call)
                    # Embedding & extraction times are not separately measured in fallback; keeping them at 0.

                    # Collect chunk results for batch processing
                    all_chunk_results.extend(chunk_results)

                    # Extract chunk ID from the entity_info (actual chunk_id created by processor)
                    if entity_info and "chunk_id" in entity_info:
                        chunk_id = entity_info["chunk_id"]
                        multimodal_chunk_ids.append(chunk_id)

                    self.logger.info(
                        f"{content_type} processing complete: {entity_info.get('entity_name', 'Unknown')}"
                    )
                else:
                    self.logger.warning(
                        f"No suitable processor found for {content_type} type content"
                    )

            except Exception as e:
                self.logger.error(f"Error processing multimodal content: {str(e)}")
                self.logger.debug("Exception details:", exc_info=True)
                continue

        # Update doc_status to include multimodal chunks in the standard chunks_list
        if multimodal_chunk_ids:
            try:
                # Get current document status
                current_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)

                if current_doc_status:
                    existing_chunks_list = current_doc_status.get("chunks_list", [])
                    existing_chunks_count = current_doc_status.get("chunks_count", 0)

                    # Add multimodal chunks to the standard chunks_list
                    updated_chunks_list = existing_chunks_list + multimodal_chunk_ids
                    updated_chunks_count = existing_chunks_count + len(
                        multimodal_chunk_ids
                    )

                    # Update document status with integrated chunk list
                    await self.lightrag.doc_status.upsert(
                        {
                            doc_id: {
                                **current_doc_status,  # Keep existing fields
                                "chunks_list": updated_chunks_list,  # Integrated chunks list
                                "chunks_count": updated_chunks_count,  # Updated total count
                                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                            }
                        }
                    )

                    # Ensure doc_status update is persisted to disk
                    await self.lightrag.doc_status.index_done_callback()

                    self.logger.info(
                        f"Updated doc_status with {len(multimodal_chunk_ids)} multimodal chunks integrated into chunks_list"
                    )

            except Exception as e:
                self.logger.warning(
                    f"Error updating doc_status with multimodal chunks: {e}"
                )

        # Batch merge all multimodal content results (similar to text content processing)
        if all_chunk_results:
            from lightrag.operate import merge_nodes_and_edges
            from lightrag.kg.shared_storage import (
                get_namespace_data,
                get_pipeline_status_lock,
            )

            # Get pipeline status and lock from shared storage
            pipeline_status = await get_namespace_data("pipeline_status")
            pipeline_status_lock = get_pipeline_status_lock()

            await merge_nodes_and_edges(
                chunk_results=all_chunk_results,
                knowledge_graph_inst=self.lightrag.chunk_entity_relation_graph,
                entity_vdb=self.lightrag.entities_vdb,
                relationships_vdb=self.lightrag.relationships_vdb,
                global_config=self.lightrag.__dict__,
                full_entities_storage=self.lightrag.full_entities,
                full_relations_storage=self.lightrag.full_relations,
                doc_id=doc_id,
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.lightrag.llm_response_cache,
                current_file_number=1,
                total_files=1,
                file_path=file_name,
            )

            await self.lightrag._insert_done()

        self.logger.info("Individual multimodal content processing complete")

        # Compute total multimodal processing duration and store
        total_duration = time.perf_counter() - total_start
        self._last_stats["multimodal_processing_time"]["duration_sec"] = total_duration
        total_duration = time.perf_counter() - total_start
        self._last_stats["multimodal_processing_time"]["duration_sec"] = total_duration

        # Mark multimodal content as processed
        await self._mark_multimodal_processing_complete(doc_id)

    async def _process_multimodal_content_batch_type_aware(
        self, multimodal_items: List[Dict[str, Any]], file_path: str, doc_id: str, **kwargs
    ):
        """
        Type-aware batch processing that selects correct processors based on content type.
        This is the corrected implementation that handles different modality types properly.

        Args:
            multimodal_items: List of multimodal items with different types
            file_path: File path for citation
            doc_id: Document ID for proper association
        """
        if not multimodal_items:
            self.logger.debug("No multimodal content to process")
            return

        # Get existing chunks count for proper order indexing
        try:
            existing_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            existing_chunks_count = (
                existing_doc_status.get("chunks_count", 0) if existing_doc_status else 0
            )
        except Exception:
            existing_chunks_count = 0

        # Use LightRAG's concurrency control
        semaphore = asyncio.Semaphore(getattr(self.lightrag, "max_parallel_insert", 2))

        # Progress tracking variables
        total_items = len(multimodal_items)
        completed_count = 0
        progress_lock = asyncio.Lock()

        # Log processing start
        self.logger.info(f"Starting to process {total_items} multimodal content items")

        # Stage 1: Concurrent generation of descriptions using correct processors for each type
        async def process_single_item_with_correct_processor(
            item: Dict[str, Any], index: int, file_path: str
        ):
            """Process single item using the correct processor for its type"""
            nonlocal completed_count
            async with semaphore:
                try:
                    content_type = item.get("type", "unknown")

                    # Select the correct processor based on content type
                    processor = get_processor_for_type(
                        self.modal_processors, content_type
                    )

                    if not processor:
                        self.logger.warning(
                            f"No processor found for type: {content_type}"
                        )
                        return None

                    item_info = {
                        "page_idx": item.get("page_idx", 0),
                        "index": index,
                        "type": content_type,
                    }

                    # Call the correct processor's description generation method
                    (
                        description,
                        entity_info,
                    ) = await processor.generate_description_only(
                        modal_content=item,
                        content_type=content_type,
                        item_info=item_info,
                        entity_name=None,  # Let LLM auto-generate
                    )

                    # Update progress (non-blocking)
                    async with progress_lock:
                        completed_count += 1
                        if (
                            completed_count % max(1, total_items // 10) == 0
                            or completed_count == total_items
                        ):
                            progress_percent = (completed_count / total_items) * 100
                            self.logger.info(
                                f"Multimodal chunk generation progress: {completed_count}/{total_items} ({progress_percent:.1f}%)"
                            )

                    return {
                        "index": index,
                        "content_type": content_type,
                        "description": description,
                        "entity_info": entity_info,
                        "original_item": item,
                        "item_info": item_info,
                        "chunk_order_index": existing_chunks_count + index,
                        "processor": processor,  # Keep reference to the processor used
                        "file_path": file_path,  # Add file_path to the result
                    }

                except Exception as e:
                    # Update progress even on error (non-blocking)
                    async with progress_lock:
                        completed_count += 1
                        if (
                            completed_count % max(1, total_items // 10) == 0
                            or completed_count == total_items
                        ):
                            progress_percent = (completed_count / total_items) * 100
                            self.logger.info(
                                f"Multimodal chunk generation progress: {completed_count}/{total_items} ({progress_percent:.1f}%)"
                            )

                    self.logger.error(
                        f"Error generating description for {content_type} item {index}: {e}"
                    )
                    return None

        # Stage 1: Concurrent generation of descriptions using correct processors for each type
        t_desc_start = time.perf_counter()
        tasks = [
            asyncio.create_task(
                process_single_item_with_correct_processor(item, i, file_path)
            )
            for i, item in enumerate(multimodal_items)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        t_desc_end = time.perf_counter()

        if hasattr(self, "_last_stats"):
            self._last_stats["multimodal_description_time"]["duration_sec"] = round(t_desc_end - t_desc_start, 3)
            self.logger.info(f"[Phase] Multimodal Description Generation took {self._last_stats['multimodal_description_time']['duration_sec']}s")

        # Filter successful results
        multimodal_data_list = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Task failed: {result}")
                continue
            if result is not None:
                multimodal_data_list.append(result)

        if not multimodal_data_list:
            self.logger.warning("No valid multimodal descriptions generated")
            return

        self.logger.info(
            f"Generated descriptions for {len(multimodal_data_list)}/{len(multimodal_items)} multimodal items using correct processors"
        )

        # Stage 2: Convert to LightRAG chunks format
        
        t_embed_start = time.perf_counter()

        lightrag_chunks = self._convert_to_lightrag_chunks_type_aware(
            multimodal_data_list, file_path, doc_id, role_ids=kwargs.get("role_ids"), company_id=kwargs.get("company_id")
        )

        # Stage 3: Store chunks to LightRAG storage
        await self._store_chunks_to_lightrag_storage_type_aware(lightrag_chunks)

        # Stage 3.5: Store multimodal main entities to entities_vdb and full_entities
        await self._store_multimodal_main_entities(
            multimodal_data_list, lightrag_chunks, file_path, doc_id
        )
        t_embed_end = time.perf_counter()
        
        if hasattr(self, "_last_stats"):
            self._last_stats["multimodal_embedding_time"]["duration_sec"] = round(t_embed_end - t_embed_start, 3)
            self.logger.info(f"[Phase] Multimodal Embedding & Storage took {self._last_stats['multimodal_embedding_time']['duration_sec']}s")

        # Track chunk IDs for doc_status update
        chunk_ids = list(lightrag_chunks.keys())

        t_extract_start = time.perf_counter()
        # Stage 4: Use LightRAG's batch entity relation extraction
        chunk_results = await self._batch_extract_entities_lightrag_style_type_aware(
            lightrag_chunks
        )

        # Stage 5: Add belongs_to relations (multimodal-specific)
        enhanced_chunk_results = await self._batch_add_belongs_to_relations_type_aware(
            chunk_results, multimodal_data_list
        )
        t_extract_end = time.perf_counter()

        if hasattr(self, "_last_stats"):
            self._last_stats["multimodal_extraction_time"]["duration_sec"] = round(t_extract_end - t_extract_start, 3)
            self.logger.info(f"[Phase] Multimodal Entity Extraction took {self._last_stats['multimodal_extraction_time']['duration_sec']}s")

        # Stage 6: Use LightRAG's batch merge
        t_merge_start = time.perf_counter()
        await self._batch_merge_lightrag_style_type_aware(
            enhanced_chunk_results, file_path, doc_id
        )
        t_merge_end = time.perf_counter()
        
        # Capture stats
        if hasattr(self, "_last_stats"):
             self._last_stats["crossmodal_graph_merging_time"]["duration_sec"] = round(t_merge_end - t_merge_start, 3)
             self.logger.info(f"[Phase] Cross-modal Graph Merging took {self._last_stats['crossmodal_graph_merging_time']['duration_sec']}s")

        # Stage 7: Update doc_status with integrated chunks_list
        await self._update_doc_status_with_chunks_type_aware(doc_id, chunk_ids)

    def _convert_to_lightrag_chunks_type_aware(
        self, multimodal_data_list: List[Dict[str, Any]], file_path: str, doc_id: str, role_ids: List[str] = None, company_id: str = None
    ) -> Dict[str, Any]:
        """Convert multimodal data to LightRAG standard chunks format"""

        chunks = {}

        for data in multimodal_data_list:
            description = data["description"]
            entity_info = data["entity_info"]
            chunk_order_index = data["chunk_order_index"]
            content_type = data["content_type"]
            original_item = data["original_item"]

            # Apply the appropriate chunk template based on content type
            formatted_chunk_content = self._apply_chunk_template(
                content_type, original_item, description
            )

            # Generate chunk_id
            chunk_id = compute_mdhash_id(formatted_chunk_content, prefix="chunk-")

            # Calculate tokens
            tokens = len(self.lightrag.tokenizer.encode(formatted_chunk_content))

            # Use full path or basename based on config
            file_ref = self._get_file_reference(file_path)

            # Build LightRAG standard chunk format
            chunks[chunk_id] = {
                "content": formatted_chunk_content,  # Now uses the templated content
                "tokens": tokens,
                "full_doc_id": doc_id,
                "chunk_order_index": chunk_order_index,
                "file_path": file_ref,
                "llm_cache_list": [],  # LightRAG will populate this field
                # Multimodal-specific metadata
                "is_multimodal": True,
                "modal_entity_name": entity_info["entity_name"],
                "original_type": data["content_type"],
                "page_idx": data["item_info"].get("page_idx", 0),
                # [RBAC] Inject role_ids metadata
                "role_ids": role_ids or ["public"],
                "company_id": company_id,
            }

        self.logger.debug(
            f"Converted {len(chunks)} multimodal items to multimodal chunks format"
        )
        return chunks

    def _apply_chunk_template(
        self, content_type: str, original_item: Dict[str, Any], description: str
    ) -> str:
        """
        Apply the appropriate chunk template based on content type

        Args:
            content_type: Type of content (image, table, equation, generic)
            original_item: Original multimodal item data
            description: Enhanced description generated by the processor

        Returns:
            Formatted chunk content using the appropriate template
        """
        from raganything.prompt import PROMPTS

        try:
            if content_type == "image":
                image_path = original_item.get("img_path", "")
                captions = original_item.get(
                    "image_caption", original_item.get("img_caption", [])
                )
                footnotes = original_item.get(
                    "image_footnote", original_item.get("img_footnote", [])
                )

                return PROMPTS["image_chunk"].format(
                    image_path=image_path,
                    captions=", ".join(captions) if captions else "None",
                    footnotes=", ".join(footnotes) if footnotes else "None",
                    enhanced_caption=description,
                )

            elif content_type == "table":
                table_img_path = original_item.get("img_path", "")
                table_caption = original_item.get("table_caption", [])
                table_body = original_item.get("table_body", "")
                table_footnote = original_item.get("table_footnote", [])

                return PROMPTS["table_chunk"].format(
                    table_img_path=table_img_path,
                    table_caption=", ".join(table_caption) if table_caption else "None",
                    table_body=table_body,
                    table_footnote=", ".join(table_footnote)
                    if table_footnote
                    else "None",
                    enhanced_caption=description,
                )

            elif content_type == "equation":
                equation_text = original_item.get("text", "")
                equation_format = original_item.get("text_format", "")

                return PROMPTS["equation_chunk"].format(
                    equation_text=equation_text,
                    equation_format=equation_format,
                    enhanced_caption=description,
                )

            else:  # generic or unknown types
                content = str(original_item.get("content", original_item))

                return PROMPTS["generic_chunk"].format(
                    content_type=content_type.title(),
                    content=content,
                    enhanced_caption=description,
                )

        except Exception as e:
            self.logger.warning(
                f"Error applying chunk template for {content_type}: {e}"
            )
            # Fallback to just the description if template fails
            return description

    async def _store_chunks_to_lightrag_storage_type_aware(
        self, chunks: Dict[str, Any]
    ):
        """Store chunks to storage"""
        try:
            # Store in text_chunks storage (required for extract_entities)
            await self.lightrag.text_chunks.upsert(chunks)

            # Store in chunks vector database for retrieval
            await self.lightrag.chunks_vdb.upsert(chunks)

            self.logger.debug(f"Stored {len(chunks)} multimodal chunks to storage")

        except Exception as e:
            self.logger.error(f"Error storing chunks to storage: {e}")
            raise

    async def _store_multimodal_main_entities(
        self,
        multimodal_data_list: List[Dict[str, Any]],
        lightrag_chunks: Dict[str, Any],
        file_path: str,
        doc_id: str = None,
    ):
        """
        Store multimodal main entities to entities_vdb and full_entities.
        This ensures that entities like "TableName (table)" are properly indexed.

        Args:
            multimodal_data_list: List of processed multimodal data with entity info
            lightrag_chunks: Chunks in LightRAG format (already formatted with templates)
            file_path: File path for the entities
            doc_id: Document ID for full_entities storage
        """
        if not multimodal_data_list:
            return

        # Create entities_vdb entries for all multimodal main entities
        entities_to_store = {}

        # Use full path or basename based on config
        file_ref = self._get_file_reference(file_path)

        for data in multimodal_data_list:
            entity_info = data["entity_info"]
            entity_name = entity_info["entity_name"]
            description = data["description"]
            content_type = data["content_type"]
            original_item = data["original_item"]

            # Apply the same chunk template to get the formatted content
            formatted_chunk_content = self._apply_chunk_template(
                content_type, original_item, description
            )

            # Generate chunk_id using the formatted content (same as in _convert_to_lightrag_chunks)
            chunk_id = compute_mdhash_id(formatted_chunk_content, prefix="chunk-")

            # Generate entity_id using LightRAG's standard format
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")

            # Create entity data in LightRAG format
            entity_data = {
                "entity_name": entity_name,
                "entity_type": entity_info.get("entity_type", content_type),
                "content": entity_info.get("summary", description),
                "source_id": chunk_id,
                "file_path": file_ref,
            }

            entities_to_store[entity_id] = entity_data

        if entities_to_store:
            try:
                # Store entities in knowledge graph
                for entity_id, entity_data in entities_to_store.items():
                    entity_name = entity_data["entity_name"]

                    # Create node data for knowledge graph
                    node_data = {
                        "entity_id": entity_name,
                        "entity_type": entity_data["entity_type"],
                        "description": entity_data["content"],
                        "source_id": entity_data["source_id"],
                        "file_path": entity_data["file_path"],
                        "created_at": int(time.time()),
                    }

                    # Store in knowledge graph
                    await self.lightrag.chunk_entity_relation_graph.upsert_node(
                        entity_name, node_data
                    )

                # Store in entities_vdb
                await self.lightrag.entities_vdb.upsert(entities_to_store)
                await self.lightrag.entities_vdb.index_done_callback()

                # NEW: Store multimodal main entities in full_entities storage
                if doc_id and self.lightrag.full_entities:
                    await self._store_multimodal_entities_to_full_entities(
                        entities_to_store, doc_id
                    )

                self.logger.debug(
                    f"Stored {len(entities_to_store)} multimodal main entities to knowledge graph, entities_vdb, and full_entities"
                )

            except Exception as e:
                self.logger.error(f"Error storing multimodal main entities: {e}")
                raise

    async def _store_multimodal_entities_to_full_entities(
        self, entities_to_store: Dict[str, Any], doc_id: str
    ):
        """
        Store multimodal main entities to full_entities storage.

        Args:
            entities_to_store: Dictionary of entities to store
            doc_id: Document ID for grouping entities
        """
        try:
            # Get current full_entities data for this document
            current_doc_entities = await self.lightrag.full_entities.get_by_id(doc_id)

            if current_doc_entities is None:
                # Create new document entry
                entity_names = list(
                    entity_data["entity_name"]
                    for entity_data in entities_to_store.values()
                )
                doc_entities_data = {
                    "entity_names": entity_names,
                    "count": len(entity_names),
                    "update_time": int(time.time()),
                }
            else:
                # Update existing document entry
                existing_entity_names = set(
                    current_doc_entities.get("entity_names", [])
                )
                new_entity_names = [
                    entity_data["entity_name"]
                    for entity_data in entities_to_store.values()
                ]

                # Add new multimodal entities to the list (avoid duplicates)
                for entity_name in new_entity_names:
                    existing_entity_names.add(entity_name)

                doc_entities_data = {
                    "entity_names": list(existing_entity_names),
                    "count": len(existing_entity_names),
                    "update_time": int(time.time()),
                }

            # Store updated data
            await self.lightrag.full_entities.upsert({doc_id: doc_entities_data})
            await self.lightrag.full_entities.index_done_callback()

            self.logger.debug(
                f"Added {len(entities_to_store)} multimodal main entities to full_entities for doc {doc_id}"
            )

        except Exception as e:
            self.logger.error(
                f"Error storing multimodal entities to full_entities: {e}"
            )
            raise

    async def _batch_extract_entities_lightrag_style_type_aware(
        self, lightrag_chunks: Dict[str, Any]
    ) -> List[Tuple]:
        """Use LightRAG's extract_entities for batch entity relation extraction"""
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_pipeline_status_lock,
        )
        from lightrag.operate import extract_entities

        # Get pipeline status (consistent with LightRAG)
        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

        # Directly use LightRAG's extract_entities
        chunk_results = await extract_entities(
            chunks=lightrag_chunks,
            global_config=self.lightrag.__dict__,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=self.lightrag.llm_response_cache,
            text_chunks_storage=self.lightrag.text_chunks,
        )

        self.logger.info(
            f"Extracted entities from {len(lightrag_chunks)} multimodal chunks"
        )
        return chunk_results

    async def _batch_add_belongs_to_relations_type_aware(
        self, chunk_results: List[Tuple], multimodal_data_list: List[Dict[str, Any]]
    ) -> List[Tuple]:
        """Add belongs_to relations for multimodal entities"""
        # Create mapping from chunk_id to modal_entity_name
        chunk_to_modal_entity = {}
        chunk_to_file_path = {}

        for data in multimodal_data_list:
            description = data["description"]
            content_type = data["content_type"]
            original_item = data["original_item"]

            # Use the same template formatting as in _convert_to_lightrag_chunks_type_aware
            formatted_chunk_content = self._apply_chunk_template(
                content_type, original_item, description
            )
            chunk_id = compute_mdhash_id(formatted_chunk_content, prefix="chunk-")

            chunk_to_modal_entity[chunk_id] = data["entity_info"]["entity_name"]
            chunk_to_file_path[chunk_id] = data.get("file_path", "multimodal_content")

        enhanced_chunk_results = []
        belongs_to_count = 0

        for maybe_nodes, maybe_edges in chunk_results:
            # Find corresponding modal_entity_name for this chunk
            chunk_id = None
            for nodes_dict in maybe_nodes.values():
                if nodes_dict:
                    chunk_id = nodes_dict[0].get("source_id")
                    break

            if chunk_id and chunk_id in chunk_to_modal_entity:
                modal_entity_name = chunk_to_modal_entity[chunk_id]
                file_path = chunk_to_file_path.get(chunk_id, "multimodal_content")

                # Add belongs_to relations for all extracted entities
                for entity_name in maybe_nodes.keys():
                    if entity_name != modal_entity_name:  # Avoid self-relation
                        belongs_to_relation = {
                            "src_id": entity_name,
                            "tgt_id": modal_entity_name,
                            "description": f"Entity {entity_name} belongs to {modal_entity_name}",
                            "keywords": "belongs_to,part_of,contained_in",
                            "source_id": chunk_id,
                            "weight": 10.0,
                            "file_path": file_path,
                        }

                        # Add to maybe_edges
                        edge_key = (entity_name, modal_entity_name)
                        if edge_key not in maybe_edges:
                            maybe_edges[edge_key] = []
                        maybe_edges[edge_key].append(belongs_to_relation)
                        belongs_to_count += 1

            enhanced_chunk_results.append((maybe_nodes, maybe_edges))

        self.logger.info(
            f"Added {belongs_to_count} belongs_to relations for multimodal entities"
        )
        return enhanced_chunk_results

    async def _batch_merge_lightrag_style_type_aware(
        self, enhanced_chunk_results: List[Tuple], file_path: str, doc_id: str = None
    ):
        """Use LightRAG's merge_nodes_and_edges for batch merge"""
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_pipeline_status_lock,
        )
        from lightrag.operate import merge_nodes_and_edges

        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

        # Use full path or basename based on config
        file_ref = self._get_file_reference(file_path)

        await merge_nodes_and_edges(
            chunk_results=enhanced_chunk_results,
            knowledge_graph_inst=self.lightrag.chunk_entity_relation_graph,
            entity_vdb=self.lightrag.entities_vdb,
            relationships_vdb=self.lightrag.relationships_vdb,
            global_config=self.lightrag.__dict__,
            full_entities_storage=self.lightrag.full_entities,
            full_relations_storage=self.lightrag.full_relations,
            doc_id=doc_id,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=self.lightrag.llm_response_cache,
            current_file_number=1,
            total_files=1,
            file_path=file_ref,
        )

        await self.lightrag._insert_done()

    async def _update_doc_status_with_chunks_type_aware(
        self, doc_id: str, chunk_ids: List[str]
    ):
        """Update document status with multimodal chunks"""
        try:
            # Get current document status
            current_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)

            if current_doc_status:
                existing_chunks_list = current_doc_status.get("chunks_list", [])
                existing_chunks_count = current_doc_status.get("chunks_count", 0)

                # Add multimodal chunks to the standard chunks_list
                updated_chunks_list = existing_chunks_list + chunk_ids
                updated_chunks_count = existing_chunks_count + len(chunk_ids)

                # Update document status with integrated chunk list
                await self.lightrag.doc_status.upsert(
                    {
                        doc_id: {
                            **current_doc_status,  # Keep existing fields
                            "chunks_list": updated_chunks_list,  # Integrated chunks list
                            "chunks_count": updated_chunks_count,  # Updated total count
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                        }
                    }
                )

                # Ensure doc_status update is persisted to disk
                await self.lightrag.doc_status.index_done_callback()

                self.logger.info(
                    f"Updated doc_status: added {len(chunk_ids)} multimodal chunks to standard chunks_list "
                    f"(total chunks: {updated_chunks_count})"
                )

        except Exception as e:
            self.logger.warning(
                f"Error updating doc_status with multimodal chunks: {e}"
            )

    async def _mark_multimodal_processing_complete(self, doc_id: str):
        """Mark multimodal content processing as complete in the document status."""
        try:
            current_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if current_doc_status:
                await self.lightrag.doc_status.upsert(
                    {
                        doc_id: {
                            **current_doc_status,
                            "multimodal_processed": True,
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                        }
                    }
                )
                await self.lightrag.doc_status.index_done_callback()
                self.logger.debug(
                    f"Marked multimodal content processing as complete for document {doc_id}"
                )
        except Exception as e:
            self.logger.warning(
                f"Error marking multimodal processing as complete for document {doc_id}: {e}"
            )

    async def is_document_fully_processed(self, doc_id: str) -> bool:
        """
        Check if a document is fully processed (both text and multimodal content).

        Args:
            doc_id: Document ID to check

        Returns:
            bool: True if both text and multimodal content are processed
        """
        try:
            doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if not doc_status:
                return False

            text_processed = doc_status.get("status") == DocStatus.PROCESSED
            multimodal_processed = doc_status.get("multimodal_processed", False)

            return text_processed and multimodal_processed

        except Exception as e:
            self.logger.error(
                f"Error checking document processing status for {doc_id}: {e}"
            )
            return False

    async def get_document_processing_status(self, doc_id: str) -> Dict[str, Any]:
        """
        Get detailed processing status for a document.

        Args:
            doc_id: Document ID to check

        Returns:
            Dict with processing status details
        """
        try:
            doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if not doc_status:
                return {
                    "exists": False,
                    "text_processed": False,
                    "multimodal_processed": False,
                    "fully_processed": False,
                    "chunks_count": 0,
                }

            text_processed = doc_status.get("status") == DocStatus.PROCESSED
            multimodal_processed = doc_status.get("multimodal_processed", False)
            fully_processed = text_processed and multimodal_processed

            return {
                "exists": True,
                "text_processed": text_processed,
                "multimodal_processed": multimodal_processed,
                "fully_processed": fully_processed,
                "chunks_count": doc_status.get("chunks_count", 0),
                "chunks_list": doc_status.get("chunks_list", []),
                "status": doc_status.get("status", ""),
                "updated_at": doc_status.get("updated_at", ""),
                "raw_status": doc_status,
            }

        except Exception as e:
            self.logger.error(
                f"Error getting document processing status for {doc_id}: {e}"
            )
            return {
                "exists": False,
                "error": str(e),
                "text_processed": False,
                "multimodal_processed": False,
                "fully_processed": False,
                "chunks_count": 0,
            }

    def get_last_processing_stats(self) -> Dict[str, Any]:
        """Return statistics from the last processed document."""
        return getattr(self, "_last_stats", {})

    async def process_document_complete(
        self,
        file_path: str,
        output_dir: str = None,
        parse_method: str = None,
        display_stats: bool = None,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        doc_id: str | None = None,
        file_name: str | None = None,
        **kwargs,
    ):
        """
        Complete document processing workflow
        
        Args:
            file_path: Path to the file to process
            output_dir: output directory (defaults to config.parser_output_dir)
            parse_method: Parse method (defaults to config.parse_method)
            display_stats: Whether to display content statistics (defaults to config.display_content_stats)
            split_by_character: Optional character to split the text by
            split_by_character_only: If True, split only by the specified character
            doc_id: Optional document ID, if not provided will be generated from content
            **kwargs: Additional parameters for parser (e.g., lang, device, start_page, end_page, formula, table, backend, source)
        """
        # Ensure LightRAG is initialized
        await self._ensure_lightrag_initialized()

        # Initialize global stats container
        self._last_stats = {
            "mineru_parsing_time": {"stage": "mineru_parsing", "duration_sec": 0.0},
            "content_analysis_time": {"stage": "content_analysis", "duration_sec": 0.0},
            "multimodal_description_time": {"stage": "multimodal_description", "duration_sec": 0.0},
            "multimodal_embedding_time": {"stage": "multimodal_embedding", "duration_sec": 0.0},
            "multimodal_extraction_time": {"stage": "multimodal_extraction", "duration_sec": 0.0},
            "crossmodal_graph_merging_time": {"stage": "crossmodal_graph_merging", "duration_sec": 0.0},
            "rag_stats": {"chunks": 0, "entities": 0, "relations": 0}
        }

        # Use config defaults if not provided
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if display_stats is None:
            display_stats = self.config.display_content_stats
        
        self.logger.info(f"Starting complete document processing: {file_path}")
        
        # Extract metadata keys that should NOT be passed to the parser
        _metadata_keys = {"role_ids", "company_id"}
        parser_kwargs = {k: v for k, v in kwargs.items() if k not in _metadata_keys}
        
        # Step 1: Parse document (MinerU Parsing Phase)
        t0 = time.perf_counter()
        content_list, content_based_doc_id = await self.parse_document(
            file_path, output_dir, parse_method, display_stats, **parser_kwargs
        )
        t1 = time.perf_counter()
        self._last_stats["mineru_parsing_time"]["duration_sec"] = round(t1 - t0, 3)
        self.logger.info(f"[Phase] MinerU Parsing took {self._last_stats['mineru_parsing_time']['duration_sec']}s")
        
        # Use provided doc_id or fall back to content-based doc_id
        if doc_id is None:
            doc_id = content_based_doc_id
            
        # Step 2: Separate text and multimodal content (Content Analysis Phase Start)
        t2 = time.perf_counter()
        text_content, multimodal_items = separate_content(content_list)
        
        # Step 2.5: Set content source for context extraction
        if hasattr(self, "set_content_source_for_context") and multimodal_items:
            self.logger.info(
                "Setting content source for context-aware multimodal processing..."
            )
            self.set_content_source_for_context(
                content_list, self.config.content_format
            )
        
        # Content Analysis technically includes separation + maybe setting context, 
        # but let's count it up to here for separation logic
        t3 = time.perf_counter()
        self._last_stats["content_analysis_time"]["duration_sec"] = round(t3 - t2, 3)
        self.logger.info(f"[Phase] Content Analysis took {self._last_stats['content_analysis_time']['duration_sec']}s")

        # Step 3: Insert pure text content 
        t_text_start = time.perf_counter()
        if text_content.strip():
            if file_name is None:
                file_name = self._get_file_reference(file_path)
            
            # [NEW] Manual Text Indexing with Metadata Injection
            # Strategy: Use LightRAG's chunker manually to inject metadata at creation time
            try:
                role_ids = kwargs.get("role_ids", ["public"])
                company_id = kwargs.get("company_id")
                self.logger.info(f"DEBUG: Processing text content with role_ids: {role_ids}, company_id: {company_id}")
                
                # Use LightRAG's native chunking from correct module
                from lightrag.operate import chunking_by_token_size
                from lightrag.utils import compute_mdhash_id
                
                token_size = self.lightrag.chunk_token_size
                overlap_size = self.lightrag.chunk_overlap_token_size
                self.logger.info(f"DEBUG TEXT INDEXING: token_size={token_size}, overlap={overlap_size}, text_len={len(text_content)}, split_by_char={split_by_character}")
                
                # Chunk the text using LightRAG's chunker
                # Note: chunking_by_token_size arguments: tokenizer, content, split_by_char, split_by_char_only, chunk_token_size, chunk_overlap_token_size
                text_chunks = chunking_by_token_size(
                    self.lightrag.tokenizer,
                    text_content,
                    split_by_character,
                    split_by_character_only,
                    token_size,
                    overlap_size
                )
                self.logger.info(f"DEBUG TEXT INDEXING: chunker returned {len(text_chunks)} chunks")
                
                # Fallback: if chunker returns 0 chunks (text smaller than chunk_token_size), create single chunk
                if not text_chunks and text_content.strip():
                    tokens = len(self.lightrag.tokenizer.encode(text_content))
                    text_chunks = [{"content": text_content.strip(), "tokens": tokens}]
                    self.logger.info(f"DEBUG TEXT INDEXING: fallback created 1 chunk ({tokens} tokens)")
                
                chunks_to_upsert = {}
                chunk_order_index = 0
                
                for chunk_data in text_chunks:
                     chunk_text = chunk_data["content"]
                     chunk_id = compute_mdhash_id(chunk_text, prefix="chunk-")
                     tokens = chunk_data["tokens"]
                     
                     chunks_to_upsert[chunk_id] = {
                         "content": chunk_text,
                         "tokens": tokens,
                         "full_doc_id": doc_id,
                         "chunk_order_index": chunk_order_index,
                         "file_path": file_name,
                         "role_ids": role_ids, # [NEW] Metadata
                         "company_id": company_id,
                         "is_multimodal": False,
                     }
                     chunk_order_index += 1
                
                self.logger.info(f"DEBUG TEXT INDEXING: created {len(chunks_to_upsert)} chunks to upsert")
                
                # Upsert Chunks
                if chunks_to_upsert:
                    await self.lightrag.chunks_vdb.upsert(chunks_to_upsert)
                    await self.lightrag.text_chunks.upsert(chunks_to_upsert)
                    
                    # Extract Entities (Text)
                    text_chunk_results = await self._batch_extract_entities_lightrag_style_type_aware(chunks_to_upsert)
                    
                    # Merge Graph (Text)
                    await self._batch_merge_lightrag_style_type_aware(text_chunk_results, file_path, doc_id)
                    
                    # Update Doc Status
                    await self._update_doc_status_with_chunks_type_aware(doc_id, list(chunks_to_upsert.keys()))
                    
                    self.logger.info(f"Manually indexed {len(chunks_to_upsert)} text chunks with roles: {role_ids}, company_id: {company_id}")
                else:
                    self.logger.warning("DEBUG TEXT INDEXING: No chunks created from text content!")
                
            except Exception as e:
                self.logger.error(f"Manual text indexing failed: {e}")
                import traceback
                traceback.print_exc()
                raise e
        else:
             if file_name is None:
                file_name = self._get_file_reference(file_path)
        
        t_text_end = time.perf_counter()
        if "rag_stats" not in self._last_stats: self._last_stats["rag_stats"] = {}
        # We store this as a distinct phase, but it contributes to "Rag KB Construction"
        self._last_stats["text_indexing_time"] = {"stage": "text_indexing", "duration_sec": round(t_text_end - t_text_start, 3)}
        self.logger.info(f"[Phase] Text Indexing took {self._last_stats['text_indexing_time']['duration_sec']}s")

        # Step 4: Process multimodal content
        t4 = time.perf_counter()
        if multimodal_items:
            if "multimodal_processing_time" not in self._last_stats:
                self._last_stats["multimodal_processing_time"] = {"stage": "multimodal_processing", "duration_sec": 0.0}
            
            await self._process_multimodal_content(
                multimodal_items, 
                file_name, 
                doc_id,
                role_ids=kwargs.get("role_ids", ["public"]),
                company_id=kwargs.get("company_id")
            )
        else:
            await self._mark_multimodal_processing_complete(doc_id)
            self.logger.debug(
                f"No multimodal content found in document {doc_id}, marked multimodal processing as complete"
            )
            
        t5 = time.perf_counter()
        if multimodal_items:
            self._last_stats["multimodal_processing_time"]["duration_sec"] = round(t5 - t4, 3)
            self.logger.info(f"[Phase] Multimodal Processing took {self._last_stats['multimodal_processing_time']['duration_sec']}s")
            
        # Try to gather stats (approximation)
        try:
             doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
             if doc_status:
                 self._last_stats["rag_stats"]["chunks"] = doc_status.get("chunks_count", 0)
                 # Entities/Relations count is harder to get exactly for *this* document without query
                 # but we can try to estimate or leave as 0 (logging_util parsing does this via regex)
        except:
             pass
             
        self.logger.info(f"Document {file_path} processing complete!")

    async def process_document_complete_lightrag_api(
        self,
        file_path: str,
        output_dir: str = None,
        parse_method: str = None,
        display_stats: bool = None,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        doc_id: str | None = None,
        scheme_name: str | None = None,
        parser: str | None = None,
        **kwargs,
    ):
        """
        API exclusively for LightRAG calls: Complete document processing workflow

        Args:
            file_path: Path to the file to process
            output_dir: output directory (defaults to config.parser_output_dir)
            parse_method: Parse method (defaults to config.parse_method)
            display_stats: Whether to display content statistics (defaults to config.display_content_stats)
            split_by_character: Optional character to split the text by
            split_by_character_only: If True, split only by the specified character
            doc_id: Optional document ID, if not provided will be generated from content
            **kwargs: Additional parameters for parser (e.g., lang, device, start_page, end_page, formula, table, backend, source)
        """
        # Use full path or basename based on config
        file_name = self._get_file_reference(file_path)
        doc_pre_id = f"doc-pre-{file_name}"
        pipeline_status = None
        pipeline_status_lock = None

        if parser:
            self.config.parser = parser

        current_doc_status = await self.lightrag.doc_status.get_by_id(doc_pre_id)

        try:
            # Ensure LightRAG is initialized
            result = await self._ensure_lightrag_initialized()
            if not result["success"]:
                await self.lightrag.doc_status.upsert(
                    {
                        doc_pre_id: {
                            **current_doc_status,
                            "status": DocStatus.FAILED,
                            "error_msg": result["error"],
                        }
                    }
                )
                return False

            # Use config defaults if not provided
            if output_dir is None:
                output_dir = self.config.parser_output_dir
            if parse_method is None:
                parse_method = self.config.parse_method
            if display_stats is None:
                display_stats = self.config.display_content_stats

            self.logger.info(f"Starting complete document processing: {file_path}")

            # Initialize doc status
            current_doc_status = await self.lightrag.doc_status.get_by_id(doc_pre_id)
            if not current_doc_status:
                await self.lightrag.doc_status.upsert(
                    {
                        doc_pre_id: {
                            "status": DocStatus.READY,
                            "content": "",
                            "error_msg": "",
                            "content_summary": "",
                            "multimodal_content": [],
                            "scheme_name": scheme_name,
                            "content_length": 0,
                            "created_at": "",
                            "updated_at": "",
                            "file_path": file_name,
                        }
                    }
                )
                current_doc_status = await self.lightrag.doc_status.get_by_id(
                    doc_pre_id
                )

            from lightrag.kg.shared_storage import (
                get_namespace_data,
                get_pipeline_status_lock,
            )

            pipeline_status = await get_namespace_data("pipeline_status")
            pipeline_status_lock = get_pipeline_status_lock()

            # Set processing status
            async with pipeline_status_lock:
                pipeline_status.update({"scan_disabled": True})
                pipeline_status["history_messages"].append("Now is not allowed to scan")

            await self.lightrag.doc_status.upsert(
                {
                    doc_pre_id: {
                        **current_doc_status,
                        "status": DocStatus.HANDLING,
                        "error_msg": "",
                    }
                }
            )

            content_list = []
            content_based_doc_id = ""

            try:
                # Step 1: Parse document
                content_list, content_based_doc_id = await self.parse_document(
                    file_path, output_dir, parse_method, display_stats, **kwargs
                )
            except MineruExecutionError as e:
                error_message = e.error_msg
                if isinstance(e.error_msg, list):
                    error_message = "\n".join(e.error_msg)
                await self.lightrag.doc_status.upsert(
                    {
                        doc_pre_id: {
                            **current_doc_status,
                            "status": DocStatus.FAILED,
                            "error_msg": error_message,
                        }
                    }
                )
                self.logger.info(
                    f"Error processing document {file_path}: MineruExecutionError"
                )
                return False
            except Exception as e:
                await self.lightrag.doc_status.upsert(
                    {
                        doc_pre_id: {
                            **current_doc_status,
                            "status": DocStatus.FAILED,
                            "error_msg": str(e),
                        }
                    }
                )
                self.logger.info(f"Error processing document {file_path}: {str(e)}")
                return False

            # Use provided doc_id or fall back to content-based doc_id
            if doc_id is None:
                doc_id = content_based_doc_id

            # Step 2: Separate text and multimodal content
            text_content, multimodal_items = separate_content(content_list)

            # Step 2.5: Set content source for context extraction in multimodal processing
            if hasattr(self, "set_content_source_for_context") and multimodal_items:
                self.logger.info(
                    "Setting content source for context-aware multimodal processing..."
                )
                self.set_content_source_for_context(
                    content_list, self.config.content_format
                )

            # Step 3: Insert pure text content and multimodal content with all parameters
            if text_content.strip():
                await insert_text_content_with_multimodal_content(
                    self.lightrag,
                    input=text_content,
                    multimodal_content=multimodal_items,
                    file_paths=file_name,
                    split_by_character=split_by_character,
                    split_by_character_only=split_by_character_only,
                    ids=doc_id,
                    scheme_name=scheme_name,
                )

            self.logger.info(f"Document {file_path} processing completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {str(e)}")
            self.logger.debug("Exception details:", exc_info=True)

            # Update doc status to Failed
            await self.lightrag.doc_status.upsert(
                {
                    doc_pre_id: {
                        **current_doc_status,
                        "status": DocStatus.FAILED,
                        "error_msg": str(e),
                    }
                }
            )
            await self.lightrag.doc_status.index_done_callback()

            # Update pipeline status
            if pipeline_status_lock and pipeline_status:
                try:
                    async with pipeline_status_lock:
                        pipeline_status.update({"scan_disabled": False})
                        error_msg = (
                            f"RAGAnything processing failed for {file_name}: {str(e)}"
                        )
                        pipeline_status["latest_message"] = error_msg
                        pipeline_status["history_messages"].append(error_msg)
                        pipeline_status["history_messages"].append(
                            "Now is allowed to scan"
                        )
                except Exception as pipeline_update_error:
                    self.logger.error(
                        f"Failed to update pipeline status: {pipeline_update_error}"
                    )

            return False

        finally:
            async with pipeline_status_lock:
                pipeline_status.update({"scan_disabled": False})
                pipeline_status["latest_message"] = (
                    f"RAGAnything processing completed for {file_name}"
                )
                pipeline_status["history_messages"].append(
                    f"RAGAnything processing completed for {file_name}"
                )
                pipeline_status["history_messages"].append("Now is allowed to scan")

    async def insert_content_list(
        self,
        content_list: List[Dict[str, Any]],
        file_path: str = "unknown_document",
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        doc_id: str | None = None,
        display_stats: bool = None,
    ):
        """
        Insert content list directly without document parsing

        Args:
            content_list: Pre-parsed content list containing text and multimodal items.
                         Each item should be a dictionary with the following structure:
                         - Text: {"type": "text", "text": "content", "page_idx": 0}
                         - Image: {"type": "image", "img_path": "/absolute/path/to/image.jpg",
                                  "image_caption": ["caption"], "image_footnote": ["note"], "page_idx": 1}
                         - Table: {"type": "table", "table_body": "markdown table",
                                  "table_caption": ["caption"], "table_footnote": ["note"], "page_idx": 2}
                         - Equation: {"type": "equation", "latex": "LaTeX formula",
                                     "text": "description", "page_idx": 3}
                         - Generic: {"type": "custom_type", "content": "any content", "page_idx": 4}
            file_path: Reference file path/name for citation (defaults to "unknown_document")
            split_by_character: Optional character to split the text by
            split_by_character_only: If True, split only by the specified character
            doc_id: Optional document ID, if not provided will be generated from content
            display_stats: Whether to display content statistics (defaults to config.display_content_stats)

        Note:
            - img_path must be an absolute path to the image file
            - page_idx represents the page number where the content appears (0-based indexing)
            - Items are processed in the order they appear in the list
        """
        # Ensure LightRAG is initialized
        await self._ensure_lightrag_initialized()

        # Use config defaults if not provided
        if display_stats is None:
            display_stats = self.config.display_content_stats

        self.logger.info(
            f"Starting direct content list insertion for: {file_path} ({len(content_list)} items)"
        )

        # Generate doc_id based on content if not provided
        if doc_id is None:
            doc_id = self._generate_content_based_doc_id(content_list)

        # Display content statistics if requested
        if display_stats:
            self.logger.info("\nContent Information:")
            self.logger.info(f"* Total blocks in content_list: {len(content_list)}")

            # Count elements by type
            block_types: Dict[str, int] = {}
            for block in content_list:
                if isinstance(block, dict):
                    block_type = block.get("type", "unknown")
                    if isinstance(block_type, str):
                        block_types[block_type] = block_types.get(block_type, 0) + 1

            self.logger.info("* Content block types:")
            for block_type, count in block_types.items():
                self.logger.info(f"  - {block_type}: {count}")

        # Step 1: Separate text and multimodal content
        text_content, multimodal_items = separate_content(content_list)

        # Step 1.5: Set content source for context extraction in multimodal processing
        if hasattr(self, "set_content_source_for_context") and multimodal_items:
            self.logger.info(
                "Setting content source for context-aware multimodal processing..."
            )
            self.set_content_source_for_context(
                content_list, self.config.content_format
            )

        # Step 2: Insert pure text content with all parameters
        if text_content.strip():
            # Use full path or basename based on config
            file_ref = self._get_file_reference(file_path)
            await insert_text_content(
                self.lightrag,
                input=text_content,
                file_paths=file_ref,
                split_by_character=split_by_character,
                split_by_character_only=split_by_character_only,
                ids=doc_id,
            )
        else:
            # Determine file reference even if no text content
            file_ref = self._get_file_reference(file_path)

        # Step 3: Process multimodal content (using specialized processors)
        import time # Added import for time.perf_counter()
        t4 = time.perf_counter()
        if multimodal_items:
            # Initialize stat entry if not exists
            if "multimodal_processing_time" not in self._last_stats:
                self._last_stats["multimodal_processing_time"] = {"stage": "multimodal_processing", "duration_sec": 0.0}
            
            await self._process_multimodal_content(multimodal_items, file_ref, doc_id)
        else:
            # If no multimodal content, mark multimodal processing as complete
            # This ensures the document status properly reflects completion of all processing
            await self._mark_multimodal_processing_complete(doc_id)
            self.logger.debug(
                f"No multimodal content found in document {doc_id}, marked multimodal processing as complete"
            )
        
        t5 = time.perf_counter()
        if multimodal_items:
            self._last_stats["multimodal_processing_time"]["duration_sec"] = round(t5 - t4, 3)
            self.logger.info(f"[Phase] Multimodal Processing took {self._last_stats['multimodal_processing_time']['duration_sec']}s")
            
        # Try to gather stats (approximation) for: {file_path}")

        self.logger.info(f"Content list insertion complete for: {file_path}")
