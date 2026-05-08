# -*- coding: utf-8 -*-
"""
LLM service for final answer generation.
All responses go through Gemini — no hardcoded replies.
"""

import json
from urllib import request, error
from backend.config import get_settings

settings = get_settings()


class LLMService:
    def __init__(self):
        self.provider = (settings.LLM_PROVIDER or "groq").lower().strip()
        
        # Gemini setup
        self.gemini_key = (settings.GEMINI_API_KEY or "").strip()
        self.gemini_model = (settings.GEMINI_MODEL or "gemini-2.0-flash").strip()
        
        # Groq setup
        self.groq_key = (settings.GROQ_API_KEY or "").strip()
        self.groq_model = (settings.GROQ_MODEL or "llama-3.3-70b-versatile").strip()

    def _build_prompt(self, query: str, context: str, route: str) -> str:
        """Build a natural prompt that lets the LLM handle everything:
        greetings, small talk, out-of-scope, and document questions."""

        base_persona = (
            "You are an intelligent research assistant called Agentic RAG. "
            "You must answer concisely, directly, and without fluff. "
            "CRITICAL RULE: You must ONLY use the provided context to answer the question. "
            "DO NOT hallucinate or use outside knowledge. If the exact answer is not explicitly "
            "stated in the context, you must clearly state that the information is missing. "
            "Keep your answers brief, ideally 2-4 sentences.\n\n"
        )

        # Add route-specific guidance when there's actual context
        route_guidance = ""
        if context and context.strip():
            if route == "graph":
                route_guidance = (
                    "You retrieved this information from a knowledge graph (Neo4j). "
                    "The context includes graph relationships and evidence chunks. "
                    "Reference the relationships when relevant.\n\n"
                )
            elif route == "hybrid":
                route_guidance = (
                    "You used both vector search and knowledge graph retrieval. "
                    "Synthesize information from both sources.\n\n"
                )
            else:
                route_guidance = (
                    "You retrieved this information via semantic vector search. "
                    "Base your answer on the provided context.\n\n"
                )

        # Build the full prompt
        if context and context.strip():
            return (
                f"{base_persona}"
                f"{route_guidance}"
                f"Retrieved context:\n{context[:3000]}\n\n"
                f"User: {query}\n\n"
                "Respond naturally and precisely:"
            )
        else:
            return (
                f"{base_persona}"
                f"User: {query}\n\n"
                "Respond naturally:"
            )

    def _call_gemini(self, prompt: str) -> str:
        """Make the actual Gemini API call."""
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.gemini_model}:generateContent?key={self.gemini_key}"
        )
        print(f"DEBUG: Gemini API Call: {self.gemini_model}")

        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }, method="POST")

        with request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            candidates = parsed.get("candidates", [])
            if not candidates: return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            return "\n".join([p.get("text", "") for p in parts if p.get("text")]).strip()

    def _call_groq(self, prompt: str) -> str:
        """Make the Groq API call (OpenAI-compatible)."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        print(f"DEBUG: Groq API Call: {self.groq_model}")

        payload = {
            "model": self.groq_model,
            "messages": [
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url, 
            data=data, 
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.groq_key}",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }, 
            method="POST"
        )

        with request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            choices = parsed.get("choices", [])
            if not choices: return ""
            return choices[0].get("message", {}).get("content", "").strip()

    def generate_answer(self, query: str, context: str, route: str) -> str:
        """Generate a natural LLM response using the configured provider."""
        prompt = self._build_prompt(query, context, route)

        # Check for API keys
        if self.provider == "gemini" and not self.gemini_key:
            return "Gemini API key is missing. Please configure GEMINI_API_KEY in .env."
        if self.provider == "groq" and not self.groq_key:
            return "Groq API key is missing. Please configure GROQ_API_KEY in .env."

        try:
            if self.provider == "groq":
                answer = self._call_groq(prompt)
            else:
                answer = self._call_gemini(prompt)
                
            if answer:
                return answer
            
            # Fallback for empty responses
            fallback_prompt = f"Answer this naturally: {query}"
            return self._call_groq(fallback_prompt) if self.provider == "groq" else self._call_gemini(fallback_prompt)

        except error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            provider_name = self.provider.upper()
            print(f"{provider_name} API ERROR: {e.code} {e.reason} - Body: {error_body}")
            
            try:
                parsed_error = json.loads(error_body)
                # Handle different error formats
                if self.provider == "gemini":
                    msg = parsed_error.get("error", {}).get("message", e.reason)
                else: # Groq/OpenAI format
                    msg = parsed_error.get("error", {}).get("message", e.reason)
            except:
                msg = e.reason

            if e.code == 429:
                return f"{provider_name} Rate Limit Reached (429). Please wait a few seconds. (Detail: {msg})"
            elif e.code == 401:
                return f"{provider_name} Authentication Error (401). Check your API key. (Detail: {msg})"
            else:
                return f"{provider_name} API error ({e.code}: {msg})."
        except Exception as e:
            print(f"LLM SYSTEM ERROR: {str(e)}")
            return f"I encountered a system error: {str(e)}"
