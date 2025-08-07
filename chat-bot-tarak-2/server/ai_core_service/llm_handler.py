# FusedChatbot/server/ai_core_service/llm_handler.py

import os
import logging
import json
from abc import ABC, abstractmethod

# --- SDK Imports (from the more advanced branch) ---
try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore
    print("Import successful!")
except ImportError:
    genai = None
    ChatGoogleGenerativeAI = None
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None
try:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    # Added new imports as requested
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    ChatOllama, HumanMessage, SystemMessage, AIMessage = None, None, None, None
    PromptTemplate, StrOutputParser = None, None # Ensure they are None if import fails

# --- Local Imports ---
try:
    from . import config as service_config
except ImportError:
    import config as service_config

logger = logging.getLogger(__name__)

# In llm_handler.py, right after the logger is defined
def _build_langchain_history(history: list) -> list:
    """Converts a list of chat messages from the frontend into LangChain Message objects."""
    messages = []
    for msg in history:
        role = msg.get('role')
        content = msg.get('parts', [{}])[0].get('text', '')
        if not content: continue # Skip empty messages
        if role == 'model':
            messages.append(AIMessage(content=content))
        elif role == 'user':
            messages.append(HumanMessage(content=content))
    return messages


# --- Prompt Templates (A complete collection from both branches) ---
# In server/ai_core_service/llm_handler.py

# In server/ai_core_service/llm_handler.py

_ANALYSIS_PROMPT_TEMPLATES = {
    "faq": """You are a data processing machine. Your only function is to extract questions and answers from the provided text.
**CRITICAL RULES:**
1.  **FORMAT:** Your output MUST strictly follow the `Q: [Question]\nA: [Answer]` format for each item.
2.  **NO PREAMBLE:** Your entire response MUST begin directly with `Q:`. Do not output any other text.
3.  **DATA SOURCE:** Base all questions and answers ONLY on the provided document text.
4.  **QUANTITY:** Generate approximately {num_items} questions.
--- START DOCUMENT TEXT ---
{doc_text_for_llm}
--- END DOCUMENT TEXT ---
EXECUTE NOW.""",

    "topics": """You are a document analysis specialist. Your task is to identify the main topics from the provided text and give a brief explanation for each. From the context below, identify the top {num_items} most important topics. For each topic, provide a single-sentence explanation.
Context:
---
{doc_text_for_llm}
---
Format the output as a numbered list. Example:
1. **Topic Name:** A brief, one-sentence explanation.
""",

    # --- CORRECTED MINDMAP PROMPT ---
    "mindmap": """You are a machine that converts text into valid Mermaid.js mind map syntax.
**CRITICAL RULES:**
1.  **Output ONLY Mermaid Syntax:** Your entire response must begin with `mindmap` and contain ONLY valid Mermaid syntax. Do not add any explanation or markdown code blocks like ```mermaid.
2.  **NO NESTED QUOTES:** A node's text, which is enclosed in double quotes, CANNOT contain other double quotes. Replace any internal double quotes with single quotes or remove them.
3.  **HIERARCHY OVER COMBINATION:** For items with a label and a value (e.g., "Phone: 12345"), create a parent-child relationship.
4.  **Conciseness:** Keep the text for each node brief and to the point.
**EXAMPLE OF PERFECT SYNTAX:**
mindmap
  root("Personal Information")
    sub("Contact")
      sub("Phone")
        sub("+91 9949954325")
--- START DOCUMENT TEXT TO CONVERT ---
{doc_text_for_llm}
--- END DOCUMENT TEXT TO CONVERT ---
Generate the valid Mermaid mind map syntax now.
""",

    # --- CORRECTED MCQ PROMPT (with escaped braces) ---
    "mcq": """You are an expert quiz designer. Your task is to generate {num_items} challenging multiple-choice questions based ONLY on the provided document text.
**CRITICAL RULES:**
1.  **Strict JSONL Format:** Your entire output must be a series of valid JSON objects, one per line (JSONL format). Do not include any other text.
2.  **JSON Structure:** Each JSON object MUST have these exact keys: "question", "options" (a list of 4 strings), and "answer" (the correct string from the options).
3.  **Plausible Distractors:** The incorrect options must be plausible and related to the topic.
4.  **Base on Context:** All questions and answers must be directly derived from the provided text.
**EXAMPLE (one line of output):**
{{"question": "What is the primary purpose of RAG?", "options": ["To train larger models", "To reduce AI hallucinations", "To improve AI speed", "To generate images"], "answer": "To reduce AI hallucinations"}}
--- START DOCUMENT TEXT ---
{doc_text_for_llm}
--- END DOCUMENT TEXT ---
Generate the {num_items} quiz questions in the specified JSONL format now.
""",
    
    # --- (Optional but Recommended) ADD A DEDICATED FLASHCARDS PROMPT ---
    "flashcards": """You are a learning assistant that creates concise, effective flashcards from a given text.
**CRITICAL RULES:**
1.  **Strict JSONL Format:** Your entire output must be a series of valid JSON objects, one per line (JSONL format). Do not include any other text.
2.  **JSON Structure:** Each JSON object MUST have these exact keys: "front" (for the question or term) and "back" (for the answer or definition).
3.  **Base on Context:** All flashcards must be directly derived from the provided text.
**EXAMPLE (one line of output):**
{{"front": "What does RAG stand for?", "back": "Retrieval-Augmented Generation"}}
--- START DOCUMENT TEXT ---
{doc_text_for_llm}
--- END DOCUMENT TEXT ---
Generate {num_items} flashcards in the specified JSONL format now.
"""
}

_SYNTHESIS_PROMPT_TEMPLATE = """You are a helpful AI assistant. Your behavior depends entirely on whether 'CONTEXT' is provided.
**RULE 1: ANSWER FROM CONTEXT**
If the 'CONTEXT' section below is NOT empty, you MUST base your answer *only* on the information within that context.
- Your response MUST begin with a "**Chain of Thought:**" section explaining which parts of the context you used.
- Following the Chain of Thought, provide the final answer under an "**Answer:**" section.
**RULE 2: ANSWER FROM GENERAL KNOWLEDGE**
If the 'CONTEXT' section below IS empty, you MUST act as a general knowledge assistant.
- Answer the user's 'QUERY' directly and conversationally.
- Do NOT mention context.
- Do NOT include a "Chain of Thought" or "Answer" section.
---
**CONTEXT:**
{context_text}
---
**QUERY:**
{query}
---
EXECUTE NOW based on the rules.
"""

_RELEVANCE_CHECK_PROMPT_TEMPLATE = """You are a meticulous relevance-checking AI. Your task is to determine if the provided 'CONTEXT' contains information that is semantically related to, or could help answer, the 'USER QUERY'.
**CRITICAL RULES:**
1.  The context does NOT need to contain a direct, complete answer. It only needs to contain related keywords, concepts, or partial information.
2.  Your entire response MUST be a single, valid JSON object.
3.  The JSON object must have two keys:
    - "is_relevant": a boolean (true or false).
    - "reason": a brief, one-sentence explanation for your decision.
---
USER QUERY: "{query}"
---
CONTEXT:
{context}
---
Provide your JSON response now.
"""

_REPORT_GENERATION_PROMPT_TEMPLATE = """You are a professional research analyst and technical writer. Your sole task is to generate a comprehensive, well-structured report on a given topic. You must base your report *exclusively* on the provided context from web search results.
**CRITICAL RULES:**
1.  **Strictly Use Context:** You MUST base your entire report on the information found in the "SEARCH RESULTS CONTEXT" section below. Do not use any external or prior knowledge.
2.  **Markdown Formatting:** The entire output MUST be in valid, clean Markdown format.
3.  **Report Structure:** The report must follow this exact structure:
    - A main title: `# Report: {topic}`
    - `## 1. Executive Summary`: A brief, high-level paragraph summarizing the most critical aspects.
    - `## 2. Key Findings`: A bulleted list of the most important points (3-5 bullets).
    - `## 3. Detailed Analysis`: The longest section, expanding on the key findings with subheadings.
    - `## 4. Conclusion`: A concluding paragraph summarizing the analysis.
    - `## 5. Sources Used`: A numbered list of the sources from the context, with citations like [1], [2] in the analysis section.
---
**SEARCH RESULTS CONTEXT:**
{context_text}
---
**TOPIC TO REPORT ON:**
{topic}
---
GENERATE THE MARKDOWN REPORT NOW.
"""

_URL_SELECTION_PROMPT_TEMPLATE = """You are an expert research assistant. Your task is to select the {num_to_select} most relevant and high-quality URLs from the provided list to help answer a user's research query.
**CRITICAL RULES:**
1.  **Analyze Relevance:** Based on the URL, title, and snippet, determine which links are most likely to contain detailed, factual information.
2.  **Prioritize Quality:** Prefer articles, official documentation, and established news sources. Avoid forums or social media.
3.  **Strict Output Format:** Your entire response MUST consist of only the selected URLs, each on a new line. Do NOT include any preamble, explanation, or numbering.
---
**USER'S RESEARCH TOPIC:** "{topic}"
---
**SEARCH RESULTS LIST:**
{search_results_text}
---
Select the top {num_to_select} URLs and provide them now, one per line.
"""

_PROMPT_REFINEMENT_TEMPLATE = """You are an AI assistant specializing in prompt engineering. Your task is to take a user's raw, potentially vague query and rewrite it into a clear, detailed, and effective prompt that will yield the best possible response from another AI.

**CRITICAL RULES:**
1.  **Do NOT answer the query.** Your only job is to improve the user's prompt.
2.  **Add Detail and Context:** Enhance the prompt with specifics. For example, if the user asks "what is machine learning", a better prompt would be "Explain the concept of machine learning, including its main types (supervised, unsupervised, reinforcement learning), and provide a simple real-world example for each type."
3.  **Maintain Intent:** Do not change the user's core goal.
4.  **Output ONLY the Refined Prompt:** Your entire response must be only the new, improved prompt text. Do not include any preamble, explanation, or quotation marks.

**RAW USER QUERY:**
"{raw_query}"

**REFINED PROMPT:**
"""


# Utility to parse "Chain of Thought" responses
def _parse_thinking_and_answer(full_llm_response: str):
    response_text = full_llm_response.strip()
    cot_start_tag = "**Chain of Thought:**"
    answer_start_tag = "**Answer:**"
    cot_index = response_text.find(cot_start_tag)
    answer_index = response_text.find(answer_start_tag)
    if cot_index != -1 and answer_index != -1:
        thinking = response_text[cot_index + len(cot_start_tag):answer_index].strip()
        answer = response_text[answer_index + len(answer_start_tag):].strip()
        return answer, thinking
    return response_text, None


# Base Handler (LangChain-native)
class BaseLLMHandler(ABC):
    def __init__(self, api_keys, model_name=None, **kwargs):
        self.api_keys = api_keys
        self.model_name = model_name
        self.kwargs = kwargs
        self.client = None
        self._validate_sdk()
        self._configure_client()

    @abstractmethod
    def _validate_sdk(self): pass
    @abstractmethod
    def _configure_client(self): pass
    @abstractmethod
    def generate_response(self, prompt, is_chat=True): pass
    @abstractmethod
    def generate_response_stream(self, prompt, is_chat=True): pass

    def analyze_document(self, document_text: str, analysis_type: str) -> str:
        prompt_template = _ANALYSIS_PROMPT_TEMPLATES.get(analysis_type)
        if not prompt_template: raise ValueError(f"Invalid analysis type: {analysis_type}")
        doc_text_for_llm = document_text[:service_config.ANALYSIS_MAX_CONTEXT_LENGTH]
        num_items = min(5 + (len(doc_text_for_llm) // 4000), 20)
        final_prompt = prompt_template.format(doc_text_for_llm=doc_text_for_llm, num_items=num_items)
        return self.generate_response(final_prompt, is_chat=False)

class GeminiHandler(BaseLLMHandler):
    def _validate_sdk(self):
        if not ChatGoogleGenerativeAI: raise ConnectionError("LangChain Google GenAI SDK missing.")
    def _configure_client(self):
        gemini_key = self.api_keys.get('gemini')
        if not gemini_key: raise ValueError("Gemini API key not found.")
        self.client = ChatGoogleGenerativeAI(
            model=self.model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash"),
            google_api_key=gemini_key,
            temperature=0.7,
            # convert_system_message_to_human=True # For models that don't support system prompts
        )
    # In server/ai_core_service/llm_handler.py -> class GeminiHandler

    def generate_response(self, prompt: str, is_chat: bool = True) -> str:
        messages = []
        
        # ================== THIS IS THE CORRECTED LOGIC ==================
        if is_chat and (system_prompt := self.kwargs.get('system_prompt')):
            # Use the correct SystemMessage type
            messages.append(SystemMessage(content=system_prompt))
        # =================================================================

        history = self.kwargs.get('chat_history', [])
        messages.extend(_build_langchain_history(history))
        messages.append(HumanMessage(content=prompt))
        
        response = self.client.invoke(messages)
        return response.content
        
    def generate_response_stream(self, prompt: str, is_chat: bool = True):
        messages = []
        if is_chat and (system_prompt := self.kwargs.get('system_prompt')):
            messages.append(SystemMessage(content=system_prompt))
    
        history = self.kwargs.get('chat_history', [])
        messages.extend(_build_langchain_history(history))
        messages.append(HumanMessage(content=prompt))
        
        for chunk in self.client.stream(messages):
            yield chunk.content

class GroqHandler(BaseLLMHandler):
    def _validate_sdk(self):
        if not ChatGroq: raise ConnectionError("LangChain Groq SDK missing.")
    def _configure_client(self):
        groq_key = self.api_keys.get('groq') or self.api_keys.get('grok') # Allow both keys
        if not groq_key: raise ValueError("Groq API key not found.")
        self.client = ChatGroq(
            temperature=0.7,
            groq_api_key=groq_key,
            model_name=self.model_name or os.getenv("DEFAULT_GROQ_LLAMA3_MODEL", "llama3-8b-8192")
        )
    def generate_response(self, prompt: str, is_chat: bool = True) -> str:
        messages = []
        if is_chat and (system_prompt := self.kwargs.get('system_prompt')):
            messages.append(SystemMessage(content=system_prompt))

        history = self.kwargs.get('chat_history', [])
        messages.extend(_build_langchain_history(history))
        messages.append(HumanMessage(content=prompt))
        response = self.client.invoke(messages)
        return response.content
        
    def generate_response_stream(self, prompt: str, is_chat: bool = True):
        messages = []
        if is_chat and (system_prompt := self.kwargs.get('system_prompt')):
            messages.append(SystemMessage(content=system_prompt))
    
        history = self.kwargs.get('chat_history', [])
        messages.extend(_build_langchain_history(history))
        messages.append(HumanMessage(content=prompt))
    
        for chunk in self.client.stream(messages):
            yield chunk.content

class OllamaHandler(BaseLLMHandler):
    def _validate_sdk(self):
        if not ChatOllama: raise ConnectionError("LangChain Ollama SDK missing.")
    def _configure_client(self):
        host = self.kwargs.get('ollama_host') or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = ChatOllama(base_url=host, model=self.model_name or os.getenv("DEFAULT_OLLAMA_MODEL", "llama3"))
    def generate_response(self, prompt: str, is_chat: bool = True) -> str:
        messages = []
        if is_chat and (system_prompt := self.kwargs.get('system_prompt')):
            messages.append(SystemMessage(content=system_prompt))

        history = self.kwargs.get('chat_history', [])
        messages.extend(_build_langchain_history(history))
        messages.append(HumanMessage(content=prompt))
        response = self.client.invoke(messages)
        return response.content
        
    def generate_response_stream(self, prompt: str, is_chat: bool = True):
        messages = []
        if is_chat and (system_prompt := self.kwargs.get('system_prompt')):
            messages.append(SystemMessage(content=system_prompt))
    
        history = self.kwargs.get('chat_history', [])
        messages.extend(_build_langchain_history(history))
        messages.append(HumanMessage(content=prompt))
    
        for chunk in self.client.stream(messages):
            yield chunk.content

# --- Core Functionality ---
PROVIDER_MAP = {"gemini": GeminiHandler, "groq": GroqHandler, "ollama": OllamaHandler}

def get_handler(provider_name: str, **kwargs) -> BaseLLMHandler:
    handler_class = next((handler for key, handler in PROVIDER_MAP.items() if provider_name.startswith(key)), None)
    if not handler_class: raise ValueError(f"Unsupported LLM provider: {provider_name}")
    return handler_class(**kwargs)

def check_context_relevance(query: str, context: str, **kwargs) -> bool:
    logger.info("Performing JSON-based relevance check on retrieved context...")
    try:
        handler = get_handler(provider_name="groq", api_keys=kwargs.get('api_keys', {}), model_name="llama3-8b-8192")
        prompt = _RELEVANCE_CHECK_PROMPT_TEMPLATE.format(query=query, context=context)
        raw_response = handler.generate_response(prompt, is_chat=False)
        response_json = json.loads(raw_response.strip())
        is_relevant = response_json.get("is_relevant", False)
        logger.info(f"Relevance check decision: {is_relevant}. Reason: {response_json.get('reason', 'N/A')}")
        return is_relevant
    except Exception as e:
        logger.error(f"Context relevance check failed: {e}. Defaulting to 'relevant' for safety.")
        return True

def generate_sub_queries(original_query: str, llm_provider: str, num_queries: int = 3, **kwargs) -> list[str]:
    logger.info(f"Generating sub-queries for: '{original_query[:50]}...'")
    try:
        handler = get_handler(provider_name=llm_provider, **kwargs)
        prompt = _SUB_QUERY_TEMPLATE.format(original_query=original_query, num_queries=num_queries)
        raw_response = handler.generate_response(prompt, is_chat=False)
        return [q.strip() for q in raw_response.strip().split('\n') if q.strip()][:num_queries]
    except Exception as e:
        logger.error(f"Failed to generate sub-queries: {e}", exc_info=True)
        return []

def generate_response(llm_provider: str, query: str, context_text: str, **kwargs) -> tuple[str, str | None]:
    logger.info(f"Generating RAG response with provider: {llm_provider}.")
    final_prompt = _SYNTHESIS_PROMPT_TEMPLATE.format(query=query, context_text=context_text)
    handler = get_handler(provider_name=llm_provider, **kwargs)
    raw_response = handler.generate_response(final_prompt, is_chat=True)
    return _parse_thinking_and_answer(raw_response)
    
def generate_chat_response(llm_provider: str, query: str, **kwargs) -> tuple[str, str | None]:
    logger.info(f"Generating conversational (non-RAG) response with provider: {llm_provider}.")
    handler = get_handler(provider_name=llm_provider, **kwargs)
    raw_response = handler.generate_response(query, is_chat=True)
    return raw_response, None

def generate_chat_response_stream(llm_provider: str, query: str, **kwargs):
    logger.info(f"Generating STREAMING conversational response with provider: {llm_provider}.")
    handler = get_handler(provider_name=llm_provider, **kwargs)
    yield from handler.generate_response_stream(query, is_chat=True)

def perform_document_analysis(document_text: str, analysis_type: str, llm_provider: str, **kwargs) -> tuple[str | None, str | None]:
    logger.info(f"Performing '{analysis_type}' analysis with {llm_provider}.")
    handler = get_handler(provider_name=llm_provider, **kwargs)
    analysis_result = handler.analyze_document(document_text, analysis_type)
    return analysis_result, None

def generate_report_from_text(topic: str, context_text: str, llm_provider: str, **kwargs) -> str | None:
    logger.info(f"Generating structured report for topic '{topic}' using provider: {llm_provider}.")
    try:
        handler = get_handler(provider_name=llm_provider or 'gemini', **kwargs)
        final_prompt = _REPORT_GENERATION_PROMPT_TEMPLATE.format(topic=topic, context_text=context_text)
        report_markdown = handler.generate_response(final_prompt, is_chat=False)
        return report_markdown
    except Exception as e:
        logger.error(f"Failed to generate report for topic '{topic}': {e}", exc_info=True)
        return None

def refine_user_prompt(raw_query: str, api_keys: dict) -> str | None:
    logger.info(f"Attempting to refine user prompt: '{raw_query}'")
    try:
        handler = get_handler(provider_name="groq", api_keys=api_keys, model_name="llama3-70b-8192")
        prompt = _PROMPT_REFINEMENT_TEMPLATE.format(raw_query=raw_query)
        refined_prompt = handler.generate_response(prompt, is_chat=False)
        return refined_prompt.strip()
    except Exception as e:
        logger.error(f"Failed to refine prompt: {e}", exc_info=True)
        return raw_query

def build_system_prompt(
    original_prompt: str, 
    history_summary: str | None = None, 
    kg_facts: str | None = None
) -> str:
    prompt_parts = [original_prompt]
    if history_summary:
        prompt_parts.append(f"---\nCONTEXT FROM RECENT CONVERSATION HISTORY:\n{history_summary}\n---")
    if kg_facts:
        prompt_parts.append(f"---\nRELEVANT FACTS FROM KNOWLEDGE BASE:\n{kg_facts}\n---")
    final_prompt = "\n\n".join(prompt_parts)
    logger.debug(f"Constructed final system prompt: {final_prompt}")
    return final_prompt

def get_podcast_qa_answer(context: str, question: str, api_keys: dict):
    """
    Handles the specific Q&A logic for the podcast feature using a fallback chain.
    """
    logger.info(f"Initiating Podcast Q&A for question: '{question[:50]}...'")

    qa_prompt_template = """INSTRUCTION:
Based ONLY on the context provided below, answer the user's question.
Your answer must be concise, accurate, and directly derived from the text.
If the context does not contain the information needed to answer the question, state ONLY: "The provided context does not contain enough information to answer this question."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    
    qa_prompt = PromptTemplate(
        template=qa_prompt_template,
        input_variables=["context", "question"]
    )

    llms_in_chain = []
    
    # --- Attempt 1: Ollama ---
    try:
        # Assumes Ollama is running at the default localhost:11434
        ollama_llm = ChatOllama(model="llama3", temperature=0)
        llms_in_chain.append(ollama_llm)
        logger.info("Ollama added to fallback chain as primary.")
    except Exception as e:
        logger.warning(f"Could not initialize Ollama. Is it running? Error: {e}")

    # --- Attempt 2: Gemini ---
    gemini_key = api_keys.get('gemini')
    if gemini_key and ChatGoogleGenerativeAI:
        try:
            # ================== FIX 1: USE A MORE COMPATIBLE MODEL NAME ==================
            gemini_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest", # More reliable than "gemini-pro"
                google_api_key=gemini_key, 
                temperature=0, 
                # convert_system_message_to_human=True
            )
            # ===========================================================================
            llms_in_chain.append(gemini_llm)
            logger.info("Gemini added to fallback chain as secondary.")
        except Exception as e:
            logger.warning(f"Could not initialize Gemini. Error: {e}")

    # --- Attempt 3: Groq ---
    groq_key = api_keys.get('groq')
    if groq_key and ChatGroq:
        try:
            groq_llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=groq_key, temperature=0)
            llms_in_chain.append(groq_llm)
            logger.info("Groq added to fallback chain as tertiary.")
        except Exception as e:
            logger.warning(f"Could not initialize Groq. Error: {e}")

    if not llms_in_chain:
        logger.error("No LLMs could be initialized for the fallback chain.")
        return "Error: No AI models are available. Please check your API keys and ensure Ollama is running."

    # The langchain .with_fallbacks() is the correct way to chain them
    chain_with_fallbacks = llms_in_chain[0].with_fallbacks(llms_in_chain[1:])
    full_chain = qa_prompt | chain_with_fallbacks | StrOutputParser()

    try:
        logger.info("Invoking LLM chain for Podcast Q&A...")
        response = full_chain.invoke({
            "context": context,
            "question": question
        })
        logger.info("Successfully received response from LLM chain.")
        return response
    except Exception as e:
        logger.error(f"The entire Podcast Q&A fallback chain failed. Final Error: {e}", exc_info=True)
        return "An unexpected error occurred while processing the question. Please try again later."


# --- EXPORTED ALIASES FOR APP-LEVEL CHECKS ---
Groq = ChatGroq
Gemini = ChatGoogleGenerativeAI
Ollama = ChatOllama

# --- CENTRALIZED AVAILABILITY CHECKER ---
AVAILABLE_PROVIDERS = {
    "groq": bool(ChatGroq),
    "gemini": bool(ChatGoogleGenerativeAI),
    "ollama": bool(ChatOllama)
}

def generate_podcast_script(source_text: str, api_keys: dict):
    """
    Uses an LLM to transform source text into a two-person podcast script.
    This version has a robust, sequential fallback mechanism.
    """
    # This is our diagnostic line from before, let's keep it.
    logger.info(f"Received API keys for script generation: {api_keys}")
    logger.info("Generating conversational podcast script from source text...")

    script_generation_prompt_template = """You are an expert scriptwriter for educational podcasts. Your task is to transform the following source text into a natural, engaging, two-person conversational podcast script.

The podcast features two hosts:
- Alex: The curious host, who asks clarifying questions and guides the conversation.
- Ben: The knowledgeable expert, who explains the concepts from the source text in detail.

CRITICAL INSTRUCTIONS:
1.  **Enrich the Content:** Do not just read the source text. Expand on the ideas, add real-world analogies, and make the concepts easier to understand.
2.  **Create a Conversational Flow:** The script must sound like a genuine conversation. Alex should prompt Ben, and Ben should elaborate.
3.  **Strict Formatting:** Every single line of dialogue MUST begin with either `Alex:` or `Ben:`. Do NOT include any other text, preambles, explanations, or scene directions outside of the dialogue lines.

Here is an example of the required format:
Alex: Welcome back to 'AI Unpacked'! Today, we're diving into a fascinating topic. Ben, can you start us off?
Ben: Absolutely, Alex. The core idea is Retrieval-Augmented Generation, or RAG. Think of it as giving an AI an open-book test.
Alex: That's a great analogy! So it's about providing context rather than relying on pure memorization?
Ben: Exactly. It dramatically helps reduce errors and what we call 'hallucinations'.

Now, transform the following source text into a full podcast script following all the rules above.

--- SOURCE TEXT ---
{source_text}
--- END SOURCE TEXT ---

PODCAST SCRIPT:
"""
    
    prompt = PromptTemplate(
        template=script_generation_prompt_template,
        input_variables=["source_text"]
    )
    max_length = 20000 # Limit input to avoid excessive API costs/time

    # --- Attempt 1: Gemini ---
    gemini_key = api_keys.get('gemini')
    if gemini_key and ChatGoogleGenerativeAI:
        try:
            logger.info("Attempting script generation with Gemini...")
            # Use a powerful model for this creative task
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=gemini_key, temperature=0.7)
            chain = prompt | llm | StrOutputParser()
            script_text = chain.invoke({"source_text": source_text[:max_length]})
            logger.info("Successfully generated podcast script with Gemini.")
            return script_text # If successful, exit the function here.
        except Exception as e:
            logger.warning(f"Gemini failed during script generation: {e}. Trying next available provider.")

    # --- Attempt 2: Groq (Fallback) ---
    groq_key = api_keys.get('groq')
    if groq_key and ChatGroq:
        try:
            logger.info("Attempting script generation with Groq as a fallback...")
            # Use a powerful model for this creative task
            llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=groq_key, temperature=0.7)
            chain = prompt | llm | StrOutputParser()
            script_text = chain.invoke({"source_text": source_text[:max_length]})
            logger.info("Successfully generated podcast script with Groq.")
            return script_text # If successful, exit the function here.
        except Exception as e:
            logger.error(f"Groq also failed during script generation: {e}", exc_info=True)

    # --- Final Fallback ---
    logger.warning("All powerful LLMs failed or were unavailable. Falling back to plain text.")
    return source_text