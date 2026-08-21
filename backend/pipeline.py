import os
import time
import asyncio
import re
import requests
import numpy as np
from groq import AsyncGroq
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

class RAGPipeline:
    def __init__(self):
        # Read API keys and configs from environment
        self.sarvam_api_key = os.environ.get("SARVAM_API_KEY", "")
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "")
        self.groq_model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
        self.chroma_db_dir = os.environ.get("CHROMA_DB_DIR", "d:/RAG/backend/chroma_db")
        self.similarity_threshold = 0.30  # Adjust slightly for multilingual vector space

        # Initialize Groq client
        self.groq_client = AsyncGroq(api_key=self.groq_api_key)
        # Use a dedicated model for guard_out that reliably returns yes/no after thinking
        self.guard_out_model = "qwen/qwen3.6-27b"

        # Initialize local multilingual SentenceTransformer model
        self.model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        self.embed_model = SentenceTransformer(self.model_name)
        self.tokenizer = self.embed_model.tokenizer

        # Define supported languages mapping
        self.LANGUAGE_NAMES = {
            "en": "English",
            "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi",
            "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali",
            "or": "Odia", "pa": "Punjabi", "sa": "Sanskrit", "ta": "Tamil",
            "te": "Telugu", "ur": "Urdu",
        }

        # Initialize Chroma DB client
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_dir)
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.model_name)
        self.collection_name = os.environ.get("CHROMA_COLLECTION_NAME", "msmarco_rag")
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=embedding_func
        )

    def clean_thinking(self, text: str) -> str:
        """Strips <think>...</think> reasoning blocks from LLM responses."""
        if not text:
            return ""
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
        return cleaned.strip()

    def get_token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _sync_transcribe(self, file_path: str) -> dict:
        """Synchronous wrapper for transcription, executed in a thread pool."""
        url = "https://api.sarvam.ai/speech-to-text"
        headers = {
            "api-subscription-key": self.sarvam_api_key
        }
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        with open(file_path, "rb") as f:
            files = {
                "file": (os.path.basename(file_path), f, "audio/wav")
            }
            data = {
                "model": "saaras:v3"
            }
            response = requests.post(url, headers=headers, files=files, data=data, timeout=15)
            
        if response.status_code != 200:
            raise RuntimeError(f"Sarvam API STT failed with status {response.status_code}: {response.text}")
            
        resp_json = response.json()
        return {
            "transcript": resp_json.get("transcript", ""),
            "language_code": resp_json.get("language_code", None)
        }

    async def transcribe_audio(self, file_path: str) -> dict:
        """Transcribe audio with up to 2 retries on failure."""
        max_retries = 2
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                result = await asyncio.to_thread(self._sync_transcribe, file_path)
                if result and result.get("transcript"):
                    return result
            except Exception as e:
                last_err = e
                print(f"STT Attempt {attempt+1} failed: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(0.5)
        raise last_err or RuntimeError("STT transcription failed.")

    def detect_language(self, text: str, sarvam_lang_code: str = None) -> str:
        if sarvam_lang_code and sarvam_lang_code in self.LANGUAGE_NAMES:
            return sarvam_lang_code
        
        try:
            from langdetect import detect
            code = detect(text)
            if code in self.LANGUAGE_NAMES:
                return code
        except Exception:
            pass
        return "hi"  # Default fallback

    def guard_input(self, text: str, query_embedding: list[float], detected_lang_code: str, threshold=0.30) -> tuple[bool, str]:
        """Input guardrail: checks keyword triggers and vector database similarity score."""
        unsafe_keywords = ["bomb", "explod", "terroris", "kill", "hack", "bypass", "suicid", "self-harm"]
        for kw in unsafe_keywords:
            if re.search(r'\b' + kw, text, re.IGNORECASE):
                return False, "unsafe keyword detected"

        try:
            # Query matching the specific language first
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=1,
                where={"language": detected_lang_code},
                include=["embeddings", "documents"]
            )
            # Fallback to unfiltered if language-specific query returns nothing
            if not result or "embeddings" not in result or result["embeddings"] is None or len(result["embeddings"]) == 0 or len(result["embeddings"][0]) == 0:
                result = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=1,
                    include=["embeddings", "documents"]
                )
                
            if not result or "embeddings" not in result or result["embeddings"] is None or len(result["embeddings"]) == 0 or len(result["embeddings"][0]) == 0:
                return True, ""
            
            chunk_emb = result["embeddings"][0][0]
            q_arr = np.array(query_embedding)
            c_arr = np.array(chunk_emb)
            
            similarity = np.dot(q_arr, c_arr) / (np.linalg.norm(q_arr) * np.linalg.norm(c_arr))
            
            if similarity < threshold:
                return False, f"off_topic (similarity {similarity:.4f} below threshold {threshold})"
            return True, ""
        except Exception as e:
            print(f"Input guardrail embedding similarity check failed: {e}")
            return True, ""

    def embed_query(self, text: str) -> list[float]:
        """Embed query locally using sentence-transformers."""
        embedding = self.embed_model.encode(text)
        return embedding.tolist()

    def retrieve(self, query_vector: list[float], detected_lang_code: str, top_k=5) -> tuple[list[dict], bool]:
        """Retrieves top-k candidates across all strategies, filters by language, and re-ranks."""
        strategies = ["fixed", "semantic", "boundary"]
        candidates = {}
        language_fallback = False

        # Query only the matching language
        for strategy in strategies:
            try:
                results = self.collection.query(
                    query_embeddings=[query_vector],
                    n_results=top_k * 2,
                    where={"language": detected_lang_code},
                    include=["documents", "metadatas", "embeddings"]
                )
                
                if results and results["ids"] and results["ids"][0]:
                    ids = results["ids"][0]
                    docs = results["documents"][0]
                    metadatas = results["metadatas"][0]
                    embeddings = results["embeddings"][0]

                    for idx in range(len(ids)):
                        c_id = ids[idx]
                        doc_text = docs[idx]
                        metadata = metadatas[idx]
                        vector = embeddings[idx]

                        q_arr = np.array(query_vector)
                        d_arr = np.array(vector)
                        similarity = np.dot(q_arr, d_arr) / (np.linalg.norm(q_arr) * np.linalg.norm(d_arr))
                        
                        if doc_text not in candidates or similarity > candidates[doc_text]["similarity"]:
                            candidates[doc_text] = {
                                "chunk_id": metadata.get("chunk_id", c_id),
                                "strategy": metadata.get("strategy", strategy),
                                "source_passage_id": metadata.get("source_passage_id", ""),
                                "text": doc_text,
                                "similarity": float(similarity)
                            }
            except Exception as e:
                print(f"Retrieval from strategy {strategy} for {detected_lang_code} failed: {e}")

        # Fallback to unfiltered if zero results found
        if len(candidates) == 0:
            print(f"No chunks found for {detected_lang_code}. Falling back to unfiltered search.")
            language_fallback = True
            for strategy in strategies:
                try:
                    results = self.collection.query(
                        query_embeddings=[query_vector],
                        n_results=top_k * 2,
                        include=["documents", "metadatas", "embeddings"]
                    )
                    
                    if results and results["ids"] and results["ids"][0]:
                        ids = results["ids"][0]
                        docs = results["documents"][0]
                        metadatas = results["metadatas"][0]
                        embeddings = results["embeddings"][0]

                        for idx in range(len(ids)):
                            c_id = ids[idx]
                            doc_text = docs[idx]
                            metadata = metadatas[idx]
                            vector = embeddings[idx]

                            q_arr = np.array(query_vector)
                            d_arr = np.array(vector)
                            similarity = np.dot(q_arr, d_arr) / (np.linalg.norm(q_arr) * np.linalg.norm(d_arr))
                            
                            if doc_text not in candidates or similarity > candidates[doc_text]["similarity"]:
                                candidates[doc_text] = {
                                    "chunk_id": metadata.get("chunk_id", c_id),
                                    "strategy": metadata.get("strategy", strategy),
                                    "source_passage_id": metadata.get("source_passage_id", ""),
                                    "text": doc_text,
                                    "similarity": float(similarity)
                                }
                except Exception as e:
                    print(f"Unfiltered retrieval failed for strategy {strategy}: {e}")

        sorted_candidates = sorted(candidates.values(), key=lambda x: x["similarity"], reverse=True)
        return sorted_candidates[:top_k], language_fallback

    async def generate_answer(self, query: str, chunks: list[dict], detected_lang_code: str) -> str:
        """Generate answer from context using Groq LLM in the user's detected language."""
        lang_name = self.LANGUAGE_NAMES.get(detected_lang_code, "Hindi")
        if not chunks:
            system_prompt = (
                f"You are a helpful QA system. Output ONLY the translation of the phrase "
                f"'I don't have enough information on this in the dataset' in the language {lang_name}."
            )
            user_prompt = f"Translate the phrase to {lang_name}."
            try:
                response = await self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=self.groq_model,
                    temperature=0.0,
                    max_tokens=128,
                    timeout=5.0
                )
                return response.choices[0].message.content.strip()
            except Exception:
                return "I don't have enough information on this in the dataset."

        context = "\n\n".join([f"Source {idx+1} (ID: {c['chunk_id']}):\n{c['text']}" for idx, c in enumerate(chunks)])
        system_prompt = (
            f"You are a helpful and factual QA system. Answer the user's question relying ONLY on the provided source contexts. "
            f"The user's question is in the language {lang_name}. You MUST write your answer in {lang_name} as well, using its correct script. "
            f"If the contexts do not contain enough information to answer the question, say 'I don't have enough information on this in the dataset' in {lang_name}. "
            f"Keep your answer concise, factual, and strictly grounded in the context. Do not make up facts."
        )
        user_prompt = f"Contexts:\n{context}\n\nQuestion: {query}\nAnswer:"

        response = await self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=self.groq_model,
            temperature=0.0,
            max_tokens=1024,
            timeout=5.0
        )
        return response.choices[0].message.content.strip()

    async def guard_output(self, answer: str, chunks: list[dict]) -> tuple[bool, str, str]:
        """Groundedness / Hallucination check on Groq using the main generation model."""
        if not chunks:
            return True, "yes", "N/A"

        context_texts = "\n\n".join([f"Source {idx+1}:\n{c['text']}" for idx, c in enumerate(chunks)])
        user_content = (
            f"Context:\n{context_texts}\n\n"
            f"Answer: {answer}\n\n"
            "Is this answer reasonably supported by the context above? Partial support "
            "based on the context still counts as supported. Only answer \"no\" if the "
            "answer contains claims that directly contradict the context or introduces "
            "facts completely absent from it. Respond with only \"yes\" or \"no\"."
        )

        try:
            response = await self.groq_client.chat.completions.create(
                messages=[
                    {"role": "user", "content": user_content}
                ],
                model=self.guard_out_model,
                temperature=0.0,
                max_tokens=1024,
                timeout=12.0
            )
            raw_verdict = response.choices[0].message.content
            verdict = self.clean_thinking(raw_verdict).strip().lower()
            # If clean_thinking stripped everything (thinking-only model response),
            # fall back to checking the raw content for a yes/no signal
            if not verdict:
                verdict = raw_verdict.strip().lower()
            grounded = "yes" in verdict or ("no" not in verdict and len(verdict) > 0)
            return grounded, verdict, raw_verdict
        except Exception as e:
            print(f"Output guardrail LLM call failed: {e}. Falling back to default Pass.")
            return True, "yes (error fallback)", str(e)

    async def run(self, audio_file_path: str = None, text_query: str = None) -> dict:
        """Orchestrates the entire RAG pipeline stage-by-stage with precise timing."""
        latencies = {"stt": 0, "guard_in": 0, "retrieval": 0, "generation": 0, "guard_out": 0, "total": 0}
        start_pipeline = time.time()
        
        transcript = ""
        query_text = ""
        sarvam_lang_code = None

        # 1. Transcribe Stage
        if audio_file_path:
            t_start = time.time()
            try:
                stt_res = await self.transcribe_audio(audio_file_path)
                transcript = stt_res.get("transcript", "")
                sarvam_lang_code = stt_res.get("language_code", None)
                query_text = transcript
                latencies["stt"] = int((time.time() - t_start) * 1000)
            except Exception as e:
                latencies["total"] = int((time.time() - start_pipeline) * 1000)
                return {
                    "error": f"STT transcription failed: {str(e)}",
                    "flagged": True,
                    "grounded": False,
                    "latency_ms": latencies,
                    "detected_language": "hi",
                    "detected_language_name": "Hindi",
                    "language_fallback": False
                }
        elif text_query:
            query_text = text_query.strip()
            
        if not query_text:
            latencies["total"] = int((time.time() - start_pipeline) * 1000)
            return {
                "error": "Empty query",
                "flagged": True,
                "grounded": False,
                "latency_ms": latencies,
                "detected_language": "hi",
                "detected_language_name": "Hindi",
                "language_fallback": False
            }

        # Detect query language
        detected_lang_code = self.detect_language(query_text, sarvam_lang_code)

        # 2. Local Query Embedding & Input Guardrail
        t_guard_in_start = time.time()
        query_vector = self.embed_query(query_text)
        guard_ok, guard_reason = self.guard_input(query_text, query_vector, detected_lang_code)
        latencies["guard_in"] = int((time.time() - t_guard_in_start) * 1000)

        # If Input Guardrail blocks, stop execution
        if not guard_ok:
            latencies["total"] = int((time.time() - start_pipeline) * 1000)
            return {
                "transcript": query_text,
                "answer": f"Request blocked: {guard_reason}",
                "sources": [],
                "flagged": True,
                "grounded": False,
                "latency_ms": latencies,
                "detected_language": detected_lang_code,
                "detected_language_name": self.LANGUAGE_NAMES.get(detected_lang_code, "Unknown"),
                "language_fallback": False
            }

        # 3. Retrieval Stage
        t_ret_start = time.time()
        retrieved_chunks, language_fallback = self.retrieve(query_vector, detected_lang_code)
        latencies["retrieval"] = int((time.time() - t_ret_start) * 1000)

        # Off-Topic Similarity Check
        max_score = max([c["similarity"] for c in retrieved_chunks]) if retrieved_chunks else 0.0
        if max_score < self.similarity_threshold:
            latencies["total"] = int((time.time() - start_pipeline) * 1000)
            
            # Generate translated off-topic message
            lang_name = self.LANGUAGE_NAMES.get(detected_lang_code, "Hindi")
            system_prompt = (
                f"You are a helpful QA system. Output ONLY the translation of the phrase "
                f"'I don't have enough information on this in the dataset' in the language {lang_name}."
            )
            user_prompt = f"Translate the phrase to {lang_name}."
            try:
                response = await self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=self.groq_model,
                    temperature=0.0,
                    max_tokens=128,
                    timeout=3.0
                )
                answer = response.choices[0].message.content.strip()
            except Exception:
                answer = "I don't have enough information on this in the dataset."

            return {
                "transcript": query_text,
                "answer": answer,
                "sources": retrieved_chunks,
                "flagged": False,
                "grounded": True,
                "latency_ms": latencies,
                "detected_language": detected_lang_code,
                "detected_language_name": self.LANGUAGE_NAMES.get(detected_lang_code, "Unknown"),
                "language_fallback": language_fallback
            }

        # 4. Generation Stage
        t_gen_start = time.time()
        try:
            raw_answer = await self.generate_answer(query_text, retrieved_chunks, detected_lang_code)
            answer = self.clean_thinking(raw_answer)
            if not answer:
                answer = "I don't have enough information on this in the dataset."
            latencies["generation"] = int((time.time() - t_gen_start) * 1000)
        except Exception as e:
            latencies["total"] = int((time.time() - start_pipeline) * 1000)
            return {
                "error": f"Answer generation failed: {str(e)}",
                "flagged": True,
                "grounded": False,
                "latency_ms": latencies,
                "detected_language": detected_lang_code,
                "detected_language_name": self.LANGUAGE_NAMES.get(detected_lang_code, "Unknown"),
                "language_fallback": language_fallback
            }

        # 5. Output Guardrail Stage
        t_guard_out_start = time.time()
        raw_verdict = "N/A"
        verdict = "N/A"
        try:
            grounded, verdict, raw_verdict = await self.guard_output(answer, retrieved_chunks)
            latencies["guard_out"] = int((time.time() - t_guard_out_start) * 1000)
            
            if not grounded:
                # Replace with fallback text in user's matching language
                lang_name = self.LANGUAGE_NAMES.get(detected_lang_code, "Hindi")
                system_prompt = (
                    f"Translate the phrase 'insufficient grounded context' into the language {lang_name}. "
                    f"Output ONLY the translated text in its correct script."
                )
                user_prompt = f"Translate to {lang_name}."
                try:
                    response = await self.groq_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        model=self.groq_model,
                        temperature=0.0,
                        max_tokens=128,
                        timeout=3.0
                    )
                    answer = response.choices[0].message.content.strip()
                except Exception:
                    answer = "insufficient grounded context"
        except Exception as e:
            print(f"Output guardrail failed: {e}")
            grounded = True
            latencies["guard_out"] = int((time.time() - t_guard_out_start) * 1000)

        latencies["total"] = int((time.time() - start_pipeline) * 1000)

        # Debug logging
        try:
            print("\n=== DEBUG RAG PIPELINE RUN ===")
            print(f"QUERY: '{query_text.encode('ascii', errors='ignore').decode('ascii')}'")
            print(f"LANGUAGE DETECTED: {detected_lang_code} ({self.LANGUAGE_NAMES.get(detected_lang_code)})")
            print("RETRIEVED CHUNKS:")
            for idx, c in enumerate(retrieved_chunks):
                print(f"  [{idx}] (ID: {c['chunk_id']}) Similarity: {c['similarity']:.4f}")
                safe_chunk = c['text'][:120].encode('ascii', errors='ignore').decode('ascii')
                print(f"      Text: {safe_chunk}...")
            safe_ans = answer.encode('ascii', errors='ignore').decode('ascii')
            print(f"GENERATED ANSWER: '{safe_ans}'")
            print(f"GROUNDEDNESS VERDICT: '{verdict}' (Grounded: {grounded})")
            print("==============================\n")
        except Exception as e:
            print(f"Debug logging printed with error: {e}")

        return {
            "transcript": query_text,
            "answer": answer,
            "sources": retrieved_chunks,
            "flagged": False,
            "grounded": grounded,
            "latency_ms": latencies,
            "detected_language": detected_lang_code,
            "detected_language_name": self.LANGUAGE_NAMES.get(detected_lang_code, "Unknown"),
            "language_fallback": language_fallback
        }





