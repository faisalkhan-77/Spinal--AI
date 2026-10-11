import sys
print("🚀 Script started", file=sys.stderr)
sys.stderr.flush()

from flask import Flask, render_template, request, session, redirect, url_for, jsonify, Response,send_file
from flask_socketio import SocketIO, emit
import base64
import os
import time
import re
import io
from dotenv import load_dotenv
from google import genai
from google.genai import types
import supabase
from datetime import datetime, timedelta
import threading
import json

print("✅ Imports done", file=sys.stderr)
sys.stderr.flush()

load_dotenv()

# ============= CONFIG =============
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")

print(f"🔍 GEMINI: {'✅' if GEMINI_API_KEY else '❌'}", file=sys.stderr)
print(f"🔍 SUPABASE_URL: {'✅' if SUPABASE_URL else '❌'}", file=sys.stderr)
print(f"🔍 SUPABASE_KEY: {'✅' if SUPABASE_KEY else '❌'}", file=sys.stderr)
print(f"🔍 SERVICE_ROLE: {'✅' if SUPABASE_SERVICE_ROLE_KEY else '⚠️ (using anon key)'}", file=sys.stderr)
print(f"🔍 FLASK_SECRET_KEY: {'✅' if FLASK_SECRET_KEY else '❌ (using fallback)'}", file=sys.stderr)
sys.stderr.flush()

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file!")

# Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

# ============================================================
# MEMORY SYSTEM CONFIG
# ============================================================
MEMORY_EXTRACTION_MODEL = 'gemini-3.5-flash-lite'
EMBEDDING_MODEL = 'gemini-embedding-001'
MEMORY_THRESHOLD = 0.45
IMPORTANCE_THRESHOLD = 8          # Memories with importance >= 8 always injected
DEDUP_THRESHOLD = 0.78            # Catches name corrections
DEDUP_HIGH_SIMILARITY = 0.90      # Force update threshold
MAX_MEMORIES_PER_QUERY = 5

# ============================================================
# FILE UPLOAD CONFIG
# ============================================================
UPLOAD_BUCKET = 'user-uploads'
MAX_UPLOAD_SIZE = 15 * 1024 * 1024  # 15 MB

ALLOWED_MIME_TYPES = {
    # Images
    'image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif',
    # Documents
    'application/pdf',
    # Text-based
    'text/csv', 'text/plain', 'text/markdown', 'application/csv',
}

# Gemini-supported mimes for direct part attachment
GEMINI_FILE_MIMES = {
    'image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif',
    'application/pdf',
}

# Text-based mimes that we'll read and inject into prompt
TEXT_FILE_MIMES = {
    'text/csv', 'text/plain', 'text/markdown', 'application/csv',
}

MAX_TEXT_FILE_CHARS = 60000  # ~60k chars for text files

MEMORY_EXTRACTION_PROMPT = """You are a memory extraction system for an AI assistant called Spinal.

Given a conversation exchange, extract ONLY facts about the USER that are worth remembering for future conversations.

EXTRACT:
- Identity facts (name, location, profession, background, relationships)
- Preferences (likes, dislikes, communication style, work habits)
- Ongoing projects, goals, or challenges
- Important recurring context (constraints, tools they use, deadlines)

DO NOT EXTRACT:
- Generic questions or one-time tasks
- AI's own statements or opinions
- Trivial or obvious facts
- Speculation or things the user didn't explicitly state

IMPORTANT - User Corrections:
If the user CORRECTS or UPDATES a previously stated fact (e.g., "actually my name is Faisal"
or "call me Faisal" or "mera naam Faisal hai"), extract the CORRECTED fact. The system will
automatically replace the old memory.

Return a JSON array. Each item must have:
{{
  "content": "Concise fact written in 3rd person (e.g., 'User's name is Faisal')",
  "category": "identity | preference | project | goal | context",
  "importance": 1-10 (10 = core identity, 5 = useful context, 1 = minor)
}}

If nothing worth remembering exists, return an empty array: []

CONVERSATION:
User: {user_message}
Assistant: {ai_response}

Return ONLY valid JSON. No markdown fences, no explanation."""


# ============================================================
# CORE SYSTEM PROMPT — Spinal's foundational behavior
# This is Layer 1. Personality + Style are Layers 2 & 3.
# ============================================================
SYSTEM_PROMPT = """You are Spinal, a vision-native AI assistant built to think clearly, answer precisely, and behave like a senior expert across every domain — coding, math, science, business, writing, and personal advice.

You are not a chatbot. You are a thinking partner. Your goal is to give the user the same quality of answer a world-class expert would give, in the fewest words necessary.

═══════════════════════════════════════════════════════════
CORE BEHAVIORAL CONTRACT
═══════════════════════════════════════════════════════════

1. BE DIRECT
   - Answer first, explain second.
   - Skip preambles like "Great question!", "That's a good point!", "I'd be happy to help!"
   - Never repeat the user's question back to them.
   - Never begin with "As an AI..." or "As a language model..."

2. BE HONEST
   - If you don't know something, say "I don't know" or "I'm not sure."
   - If a question is ambiguous, ask ONE clarifying question — not five.
   - If the user is factually wrong, correct them respectfully but directly.
   - Never invent facts, sources, statistics, or code.

3. BE PRECISE
   - Vague answers are failures. Give specific, actionable information.
   - Use concrete examples where they help.
   - Distinguish clearly between facts, opinions, and speculation.

4. BE USEFUL
   - Solve the actual problem the user has, not the one you wish they had.
   - If the question is unclear, answer the most likely intent.
   - Anticipate follow-ups — but don't bloat answers with unsolicited advice.

═══════════════════════════════════════════════════════════
FORMATTING STANDARDS
═══════════════════════════════════════════════════════════
- Use Markdown: **bold** for key terms, `code` for inline code.
- Code blocks MUST include a language tag (```python, ```javascript).
- Use bullet lists for 3+ parallel items; use prose for 1-2 items.
- Use headings only for answers longer than 400 words.
- Never use headings, lists, or bold for very short answers (1-3 sentences).
- Emojis: use ONLY if the user used an emoji first in this conversation. 
  Default to NO emojis. Never use 🚀, ✅, 💡, or other decorative emojis 
  as a default ending.
MATH & NOTATION:
- NEVER use LaTeX notation like $O(N^2)$, \frac{}{}, \int, etc.
- Write math in plain text: "O(N²)", "N squared", "1/2", "square root of X".
- Use Unicode math symbols when helpful: ², ³, ×, ÷, ≈, ≤, ≥, √, π, Σ.
- Complex equations → use a code block with clear formatting.

ENDING RULES:
- Never end with "Let me know if...", "Feel free to ask...", 
  "Happy to help further", or any variant.
- End when the answer is complete. Do not add filler closings.
- Do not offer unsolicited next steps, advice, or extensions unless the 
  user's question explicitly asks for them.
- Do not add bonus advice or "while you're at it" suggestions.

═══════════════════════════════════════════════════════════
ANTI-PATTERNS — NEVER DO THESE
═══════════════════════════════════════════════════════════

- Never start with: "Great question!", "Certainly!", "Of course!"
- Never say: "As an AI, I...", "As a language model, I..."
- Never apologize unnecessarily when nothing is wrong.
- Never repeat the user's question before answering.
- Never end with: "Let me know if you have more questions!"
- Never use emojis in professional, technical, or sensitive contexts.
- Never pad short answers with unnecessary caveats or disclaimers.
- Never give medical, legal, or financial advice as if you were a licensed professional — always recommend consulting a qualified expert for high-stakes decisions.

═══════════════════════════════════════════════════════════
LANGUAGE & TONE
═══════════════════════════════════════════════════════════

- Match the user's language. If they write in Hinglish, reply in Hinglish.
- If they write in Hindi, reply in Hindi. If in English, reply in English.
- Default tone is professional, warm, and direct — not robotic, not overly casual.
- Adjust tone based on the user's Personality preference (see below).

═══════════════════════════════════════════════════════════
HANDLING EDGE CASES
═══════════════════════════════════════════════════════════

- Ambiguous question → ask ONE clarifying question, or answer the most likely intent if the ambiguity is minor.
- Multi-part question → answer each part clearly labeled.
- User is frustrated → acknowledge briefly, then solve the problem.
- User is wrong → correct directly without being condescending.
- User asks for opinion → give your honest view with reasoning, don't hedge.
- Off-topic or harmful request → decline clearly, briefly explain why.

═══════════════════════════════════════════════════════════
INSTRUCTION HIERARCHY (in order of priority)
═══════════════════════════════════════════════════════════

1. Safety and ethics (highest)
2. Accuracy and honesty
3. User's explicit instructions in this conversation
4. User's Personality preference
5. User's Response Style preference
6. Default behavior from this system prompt

If a lower-priority instruction conflicts with a higher one, the higher wins.

═══════════════════════════════════════════════════════════
VISION & MEMORY
═══════════════════════════════════════════════════════════

Camera rules:

1. ONLY discuss the camera/image if the user EXPLICITLY asks about it 
   (e.g., "what do you see?", "camera mein kya hai?", "describe what's 
   in front of me").

2. In ALL other cases, IGNORE the camera completely. Never spontaneously 
   say "camera dark hai", "kuch dikh nahi raha", "lens covered hai", 
   or any camera-related comment.

3. If the user asks what you see AND no image was provided (camera is 
   off), respond EXACTLY like this in the user's language:
   
   English: "Your camera is currently off. Please turn it on first, 
   then ask me again."
   
   Hinglish: "Camera abhi off hai. Pehle use on karo, phir mujhse 
   pucho."
   
   Do NOT say "it's dark" or "I can't see anything."

4. If the user asks what you see AND an image IS provided but it's dark 
   or unclear, respond:
   
   English: "I can see a bit but it's too dark. Try better lighting 
   or a clearer angle."
   
   Hinglish: "Thoda dikh raha hai but bahut dark hai. Light better 
   karo ya angle change karo."

5. Do NOT mention the image at all if no image was provided AND the 
   user didn't ask about it.

Microphone rules: same principle — only discuss audio if the user asks.

Memory rules: 
- Use memory silently to inform your answers. Do NOT announce that you 
  remember something.
- Do NOT repeatedly address the user by name. Use their name only when 
  it's natural — typically once at the start of a new conversation, or 
  when it adds warmth. In follow-up messages, do NOT repeat the name 
  in every reply.
- Never open every message with the user's name."""

# ============================================================
# MEMORY HELPER FUNCTIONS
# ============================================================


# ============================================================
# MEMORY HELPER FUNCTIONS
# ============================================================
def generate_embedding(text):
    """Generate 768-dim embedding for given text using Gemini."""
    try:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        embedding = result.embeddings[0].values
        print(f"✅ Embedding generated: {len(embedding)} dimensions", file=sys.stderr)
        sys.stderr.flush()
        return embedding
    except Exception as e:
        print(f"⚠️ Embedding error: {e}", file=sys.stderr)
        sys.stderr.flush()
        return None


def extract_memories(user_message, ai_response):
    """Call Gemini to extract memorable facts using native JSON mode."""
    try:
        prompt = MEMORY_EXTRACTION_PROMPT.replace(
            "{user_message}", user_message[:2000]
        ).replace(
            "{ai_response}", ai_response[:1000]
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Concise fact about user in 3rd person"
                        },
                        "category": {
                            "type": "string",
                            "enum": ["identity", "preference", "project", "goal", "context"]
                        },
                        "importance": {
                            "type": "integer",
                            "description": "1-10 scale, 10 is core identity"
                        }
                    },
                    "required": ["content", "category", "importance"]
                }
            }
        )

        response = client.models.generate_content(
            model=MEMORY_EXTRACTION_MODEL,
            contents=prompt,
            config=config
        )

        text = response.text.strip()
        print(f"🔍 Raw extraction: {text[:300]}", file=sys.stderr)
        sys.stderr.flush()

        if not text or text == "[]":
            return []

        memories = json.loads(text)

        if not isinstance(memories, list):
            return []

        valid_memories = []
        for m in memories:
            if isinstance(m, dict) and 'content' in m and m['content']:
                valid_memories.append({
                    'content': str(m.get('content', '')).strip(),
                    'category': str(m.get('category', 'context')).strip(),
                    'importance': int(m.get('importance', 5))
                })

        print(f"✅ Extracted {len(valid_memories)} valid memories", file=sys.stderr)
        sys.stderr.flush()
        return valid_memories

    except json.JSONDecodeError as e:
        print(f"⚠️ Memory JSON parse error: {e}", file=sys.stderr)
        sys.stderr.flush()
        return []
    except Exception as e:
        print(f"⚠️ Memory extraction error: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        return []


def store_memory(user_id, session_id, memory_item):
    """Store memory with SMART deduplication (category-aware + similarity)."""
    try:
        content = memory_item.get('content', '').strip()
        if not content or len(content) < 5:
            return
        category = memory_item.get('category', 'context')
        importance = max(1, min(10, int(memory_item.get('importance', 5))))

        # Generate embedding for the new memory
        embedding = generate_embedding(content)
        if not embedding:
            return

        # STEP 1: Get MULTIPLE candidates for dedup
        existing = supabase_admin.rpc('match_memories', {
            'query_embedding': embedding,
            'match_user_id': user_id,
            'match_threshold': DEDUP_THRESHOLD,
            'match_count': 5
        }).execute()

        # STEP 2: Find best match (same category OR high similarity)
        best_match = None
        best_score = 0

        if existing.data:
            for candidate in existing.data:
                sim = candidate.get('similarity', 0)
                cand_cat = candidate.get('category', '')

                # Same category = instant priority
                if cand_cat == category:
                    if sim > best_score:
                        best_match = candidate
                        best_score = sim
                # Very high similarity = force update
                elif sim > DEDUP_HIGH_SIMILARITY:
                    if sim > best_score:
                        best_match = candidate
                        best_score = sim

        if best_match:
            # UPDATE existing memory
            supabase_admin.table('user_memories').update({
                'content': content,
                'category': category,
                'importance': importance,
                'embedding': embedding,
                'updated_at': datetime.now().isoformat()
            }).eq('id', best_match['id']).execute()
            print(f"🔄 Memory UPDATED (sim={best_score:.3f}, cat={category}): {content[:50]}...", file=sys.stderr)
        else:
            # INSERT new memory
            supabase_admin.table('user_memories').insert({
                'user_id': user_id,
                'content': content,
                'category': category,
                'importance': importance,
                'embedding': embedding,
                'source_session_id': session_id
            }).execute()
            print(f"✅ Memory STORED: {content[:50]}...", file=sys.stderr)

        sys.stderr.flush()
    except Exception as e:
        print(f"⚠️ Store memory error: {e}", file=sys.stderr)
        sys.stderr.flush()


def process_memories_async(user_id, session_id, user_message, ai_response):
    """Background thread: extract and store memories."""
    try:
        memories = extract_memories(user_message, ai_response)
        if not memories:
            return
        print(f"🧠 Extracted {len(memories)} memories", file=sys.stderr)
        sys.stderr.flush()
        for memory in memories:
            store_memory(user_id, session_id, memory)
    except Exception as e:
        print(f"⚠️ Async memory processing error: {e}", file=sys.stderr)
        sys.stderr.flush()


def retrieve_memories(user_id, query_text, limit=MAX_MEMORIES_PER_QUERY):
    """Retrieve memories: semantic search + high-importance always included."""
    try:
        embedding = generate_embedding(query_text)

        semantic_memories = []

        # 1️⃣ SEMANTIC SEARCH (if embedding available)
        if embedding:
            result = supabase_admin.rpc('match_memories', {
                'query_embedding': embedding,
                'match_user_id': user_id,
                'match_threshold': MEMORY_THRESHOLD,
                'match_count': limit
            }).execute()
            semantic_memories = result.data if result.data else []

        # 2️⃣ HIGH-IMPORTANCE MEMORIES (always fetched)
        important_res = supabase_admin.table('user_memories') \
            .select('id, content, category, importance') \
            .eq('user_id', user_id) \
            .gte('importance', IMPORTANCE_THRESHOLD) \
            .order('importance', desc=True) \
            .limit(3) \
            .execute()
        important_memories = important_res.data if important_res.data else []

        # 3️⃣ MERGE (semantic + important, dedupe by id)
        seen_ids = set()
        merged = []

        for m in semantic_memories:
            if m['id'] not in seen_ids:
                merged.append(m)
                seen_ids.add(m['id'])

        for m in important_memories:
            if m['id'] not in seen_ids:
                m['similarity'] = 0.5  # synthetic score for consistency
                merged.append(m)
                seen_ids.add(m['id'])

        # 4️⃣ UPDATE ACCESS STATS
        if merged:
            memory_ids = [m['id'] for m in merged]
            try:
                supabase_admin.rpc('update_memory_access', {'memory_ids': memory_ids}).execute()
            except Exception:
                pass

        return merged

    except Exception as e:
        print(f"⚠️ Retrieve memories error: {e}", file=sys.stderr)
        sys.stderr.flush()
        return []


def format_memories_for_prompt(memories):
    """Format retrieved memories as context string."""
    if not memories:
        return ""

    lines = ["[USER MEMORY - Things you remember about this user]"]
    for m in memories:
        category = m.get('category', 'context').upper()
        content = m.get('content', '')
        lines.append(f"• [{category}] {content}")
    lines.append("[END MEMORY]")
    return "\n".join(lines)


# ============================================================
# RESPONSE STYLE SYSTEM
# ============================================================
STYLE_INSTRUCTIONS = {
    'short': (
        "RESPONSE STYLE: SHORT\n"
        "- Give the direct answer in 1-3 sentences maximum.\n"
        "- No preamble, no filler phrases.\n"
        "- Skip examples unless the user explicitly asks.\n"
        "- Prefer bullet points over paragraphs when listing."
    ),
    'balanced': (
        "RESPONSE STYLE: BALANCED\n"
        "- Give a clear, complete answer in 4-8 sentences.\n"
        "- Include brief context or reasoning where helpful.\n"
        "- One example or analogy is acceptable if it clarifies.\n"
        "- Use light structure (short paragraph + optional bullets)."
    ),
    'detailed': (
        "RESPONSE STYLE: DETAILED\n"
        "- Give a thorough, well-structured response — but SCALE THE DEPTH "
        "TO THE QUESTION.\n"
        "- A simple factual question (e.g., '2+2', 'what is the capital of "
        "India') gets a SHORT answer even in detailed mode. Never bloat "
        "trivial questions.\n"
        "- A complex question (e.g., 'explain how decorators work', 'write "
        "a function to do X') gets depth: code examples, edge cases, "
        "complexity analysis, real-world context.\n"
        "- Use headings, bullet points, or numbered steps when the answer "
        "is genuinely long (400+ words). Do NOT force structure on short "
        "answers.\n"
        "- Do NOT add 'next step' suggestions or follow-up advice unless "
        "the user explicitly asks.\n"
        "- Do NOT add summary sections at the end unless the answer is "
        "very long (600+ words).\n"
        "- Aim for depth when it helps. Aim for brevity when the question "
        "doesn't need depth."
    )
}


# ============================================================
# PERSONALITY MODES
# ============================================================
PERSONALITY_INSTRUCTIONS = {
    'professional': (
        "PERSONALITY MODE: PROFESSIONAL\n"
        "- Communicate with a formal, business-like tone.\n"
        "- Be direct, efficient, and solution-oriented.\n"
        "- Avoid slang, emojis, and casual filler phrases.\n"
        "- Use clear, structured language suitable for a workplace.\n"
        "- Treat the user as a capable professional peer."
    ),
    'friendly': (
        "PERSONALITY MODE: FRIENDLY\n"
        "- Communicate in a warm, conversational, and approachable tone.\n"
        "- Speak like a helpful, knowledgeable friend — not a corporate bot.\n"
        "- Light use of emojis is fine (1-2 per reply, only when natural).\n"
        "- Match the user's language (Hinglish if they use it).\n"
        "- Show genuine interest and empathy; celebrate their wins.\n"
        "- Match the user's energy. If they send a short casual greeting, "
        "reply warmly but briefly. Do not lecture or add life advice.\n"
        "- Never be preachy, judgmental, or push unsolicited motivation.\n"
        "- Do not turn a casual greeting into a deep conversation."
    ),
    'motivational': (
        "PERSONALITY MODE: MOTIVATIONAL\n"
        "- Communicate with high energy and an inspiring tone.\n"
        "- Use action-oriented, empowering language (power verbs, forward momentum).\n"
        "- Frame answers as opportunities, not obstacles.\n"
        "- Push the user gently toward their goals and next steps.\n"
        "- End replies with a short action nudge or encouraging line.\n"
        "- Avoid empty hype — back motivation with real, concrete guidance."
    )
}


def get_user_personality(user_id):
    """Fetch user's preferred personality mode. Default: friendly."""
    try:
        result = supabase_admin.table('user_preferences') \
            .select('personality') \
            .eq('user_id', user_id) \
            .limit(1) \
            .execute()

        if result.data and len(result.data) > 0:
            p = result.data[0].get('personality', 'friendly')
            if p in PERSONALITY_INSTRUCTIONS:
                return p
        return 'friendly'
    except Exception as e:
        print(f"⚠️ Get personality error: {e}", file=sys.stderr)
        sys.stderr.flush()
        return 'friendly'


def build_personality_instruction(personality):
    """Build the personality directive block for the AI prompt."""
    instruction = PERSONALITY_INSTRUCTIONS.get(personality, PERSONALITY_INSTRUCTIONS['friendly'])
    return f"[PERSONALITY DIRECTIVE]\n{instruction}\n[END DIRECTIVE]"


def get_user_response_style(user_id):
    """Fetch user's preferred response style from Supabase. Default: balanced."""
    try:
        result = supabase_admin.table('user_preferences') \
            .select('response_style') \
            .eq('user_id', user_id) \
            .limit(1) \
            .execute()

        if result.data and len(result.data) > 0:
            style = result.data[0].get('response_style', 'balanced')
            if style in STYLE_INSTRUCTIONS:
                return style
        return 'balanced'
    except Exception as e:
        print(f"⚠️ Get response style error: {e}", file=sys.stderr)
        sys.stderr.flush()
        return 'balanced'


def build_style_instruction(style):
    """Build the response style instruction block for the AI prompt."""
    instruction = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS['balanced'])
    return f"[SYSTEM DIRECTIVE]\n{instruction}\n[END DIRECTIVE]"


def cleanup_old_memories(user_id, days_old=30, importance_cutoff=5):
    """Auto-cleanup: delete old low-importance memories."""
    try:
        result = supabase_admin.rpc('cleanup_old_memories', {
            'target_user_id': user_id,
            'days_old': days_old,
            'importance_cutoff': importance_cutoff
        }).execute()
        deleted = result.data if result.data else 0
        if deleted > 0:
            print(f"🧹 Auto-cleanup: deleted {deleted} old memories", file=sys.stderr)
            sys.stderr.flush()
        return deleted
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}", file=sys.stderr)
        sys.stderr.flush()
        return 0


# ============================================================
# SUPABASE CLIENTS
# ============================================================
if not SUPABASE_SERVICE_ROLE_KEY:
    SUPABASE_SERVICE_ROLE_KEY = SUPABASE_KEY

supabase_admin = supabase.create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

# ============= FLASK APP =============
app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY or "spinal-ai-fallback-secret-key-do-not-use-in-prod"

# Render sets RENDER env var in production
IS_PRODUCTION = os.getenv("RENDER") is not None

app.config.update(
    SECRET_KEY=app.secret_key,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=IS_PRODUCTION,   # True only on HTTPS (production)
    SESSION_COOKIE_PATH='/',
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
)

socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*", manage_session=False)

# ============================================================
# STOP SIGNAL REGISTRY (per-user)
# ============================================================
STOP_REQUESTS = set()


# ============================================================
# GEMINI HELPERS — Multi-Model Fallback + Smart Retry
# ============================================================
def call_gemini_with_retry(contents, max_retries=2):
    """Gemini API call with fallback chain + retry. Accepts contents list."""
    config = types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,
            include_thoughts=False
        )
    )

    models_to_try = [
        'gemini-3.5-flash-lite',
        'gemini-3.5-flash',
        'gemini-3.1-flash-lite',
    ]

    last_error = None

    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                print(f"🔄 Trying {model_name} (attempt {attempt + 1}/{max_retries})...", file=sys.stderr)
                sys.stderr.flush()

                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config
                    )
                except Exception as think_err:
                    if 'thinking' in str(think_err).lower() or 'unsupported' in str(think_err).lower():
                        fallback_config = types.GenerateContentConfig(
                            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                        )
                        response = client.models.generate_content(
                            model=model_name,
                            contents=contents,
                            config=fallback_config
                        )
                    else:
                        raise

                if response and response.text:
                    print(f"✅ Success with {model_name}", file=sys.stderr)
                    sys.stderr.flush()
                    return response.text

            except Exception as e:
                error_msg = str(e)
                last_error = error_msg

                if '503' in error_msg or 'UNAVAILABLE' in error_msg or 'high demand' in error_msg:
                    wait_time = (1.5 ** attempt) + 1
                    print(f"⏳ {model_name} busy. Retry in {wait_time:.1f}s...", file=sys.stderr)
                    sys.stderr.flush()
                    time.sleep(wait_time)
                    continue

                elif '429' in error_msg or 'RESOURCE_EXHAUSTED' in error_msg:
                    print(f"⚠️ {model_name} quota exhausted. Trying next model...", file=sys.stderr)
                    sys.stderr.flush()
                    break

                elif '404' in error_msg or 'NOT_FOUND' in error_msg:
                    print(f"❌ {model_name} not found. Trying next model...", file=sys.stderr)
                    sys.stderr.flush()
                    break

                else:
                    print(f"❌ Error with {model_name}: {error_msg}", file=sys.stderr)
                    sys.stderr.flush()
                    break

    print(f"❌ All models failed. Last error: {last_error}", file=sys.stderr)
    sys.stderr.flush()
    return "Sorry, the AI service is experiencing high demand. Please try again after some time."


def call_gemini_streaming(contents):
    """
    Generator — yields text chunks from Gemini streaming API.
    Accepts a pre-built `contents` list (prompt + optional image/file parts).
    Fallback chain if model fails BEFORE first chunk.
    """
    config = types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,   # -1 = dynamic thinking (AI decides how much)
            include_thoughts=False
        )
    )

    models_to_try = [
        'gemini-3.5-flash-lite',
        'gemini-3.5-flash',
        'gemini-3.1-flash-lite',
    ]

    for model_name in models_to_try:
        first_chunk_received = False
        try:
            print(f"🌊 Streaming with {model_name}...", file=sys.stderr)
            sys.stderr.flush()

            # Try with thinking; if model doesn't support it, retry without
            try:
                stream = client.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=config
                )
            except Exception as think_err:
                if 'thinking' in str(think_err).lower() or 'unsupported' in str(think_err).lower():
                    print(f"⚠️ {model_name} doesn't support thinking — retrying without", file=sys.stderr)
                    sys.stderr.flush()
                    fallback_config = types.GenerateContentConfig(
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                    )
                    stream = client.models.generate_content_stream(
                        model=model_name,
                        contents=contents,
                        config=fallback_config
                    )
                else:
                    raise

            for chunk in stream:
                # Gemini chunks mein text hamesha nahi hota — safe access
                chunk_text = getattr(chunk, 'text', None)
                if chunk_text:
                    first_chunk_received = True
                    yield chunk_text

            if first_chunk_received:
                print(f"✅ Stream complete with {model_name}", file=sys.stderr)
                sys.stderr.flush()
                return

        except Exception as e:
            error_msg = str(e)

            # If we already sent chunks, we can't fallback — just stop
            if first_chunk_received:
                print(f"⚠️ Stream interrupted after chunks: {error_msg}", file=sys.stderr)
                sys.stderr.flush()
                return

            if '404' in error_msg or 'NOT_FOUND' in error_msg:
                print(f"❌ {model_name} not found. Next...", file=sys.stderr)
                sys.stderr.flush()
                continue
            elif '503' in error_msg or 'UNAVAILABLE' in error_msg:
                print(f"⏳ {model_name} busy. Next...", file=sys.stderr)
                sys.stderr.flush()
                time.sleep(1)
                continue
            elif '429' in error_msg or 'RESOURCE_EXHAUSTED' in error_msg:
                print(f"⚠️ {model_name} quota. Next...", file=sys.stderr)
                sys.stderr.flush()
                continue
            else:
                print(f"❌ Error {model_name}: {error_msg}", file=sys.stderr)
                sys.stderr.flush()
                continue

    # All models failed
    yield "Sorry, the AI service is experiencing high demand. Please try again after some time."


# ========================================
# HEALTH CHECK (for Render)
# ========================================
@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'service': 'spinal-ai'}), 200


# ========================================
# DEBUG ROUTE — only in local development
# ========================================
@app.route('/debug_session')
def debug_session():
    # Only available in local development, never on production
    if os.getenv("RENDER"):
        return jsonify({'error': 'Not available in production'}), 404
    return jsonify({
        'user_id': session.get('user_id'),
        'user_email': session.get('user_email'),
        'session_keys': list(session.keys()),
        'cookie_present': 'session' in request.cookies,
        'all_cookies': list(request.cookies.keys()),
        'secret_key_set': bool(app.secret_key)
    })


# ========================================
# PAGE ROUTES
# ========================================
@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/auth')
def auth_page():
    return render_template('auth.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/privacy')
def privacy():
    return render_template('privacy.html')


# ========================================
# AUTH
# ========================================
@app.route('/verify_token', methods=['POST'])
def verify_token():
    token = request.json.get('token')
    if not token:
        return jsonify({'error': 'No token'}), 401

    try:
        user_response = supabase_admin.auth.get_user(token)
        if user_response.user:
            session.permanent = True
            session['user_id'] = user_response.user.id
            session['user_email'] = user_response.user.email
            session.modified = True

            print(f"✅ Session SET for user: {user_response.user.email}", file=sys.stderr)
            print(f"✅ Session ID: {session.get('user_id')}", file=sys.stderr)
            sys.stderr.flush()

            user_id = user_response.user.id
            check_session = supabase_admin.table('sessions').select('id').eq('user_id', user_id).limit(1).execute()
            if not check_session.data:
                supabase_admin.table('sessions').insert({
                    'user_id': user_id,
                    'title': 'New Chat'
                }).execute()

            return jsonify({'success': True})
        else:
            return jsonify({'error': 'User not found'}), 401
    except Exception as e:
        print(f"❌ Verification error: {e}", file=sys.stderr)
        sys.stderr.flush()
        return jsonify({'error': str(e)}), 500


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('landing'))


# ========================================
# SESSION API ENDPOINTS
# ========================================
@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    user_id = session.get('user_id')
    if not user_id:
        print(f"❌ /api/sessions - No user_id in session. Cookies: {list(request.cookies.keys())}", file=sys.stderr)
        sys.stderr.flush()
        return jsonify({'error': 'Not authenticated'}), 401

    # Auto-cleanup trigger (runs once per day)
    try:
        last_cleanup = session.get('last_cleanup')
        today = datetime.now().date().isoformat()
        if last_cleanup != today:
            cleanup_old_memories(user_id, days_old=30, importance_cutoff=5)
            session['last_cleanup'] = today
            session.modified = True
    except Exception as e:
        print(f"⚠️ Cleanup trigger error: {e}", file=sys.stderr)

    try:
        result = supabase_admin.table('sessions') \
            .select('id, title, created_at, updated_at') \
            .eq('user_id', user_id) \
            .order('updated_at', desc=True) \
            .execute()
        return jsonify({'sessions': result.data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sessions', methods=['POST'])
def create_session():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        data = request.json or {}
        title = data.get('title', 'New Chat')
        result = supabase_admin.table('sessions').insert({
            'user_id': user_id,
            'title': title
        }).execute()

        if result.data:
            return jsonify({'session': result.data[0]})
        return jsonify({'error': 'Failed to create session'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        check = supabase_admin.table('sessions').select('id').eq('id', session_id).eq('user_id', user_id).execute()
        if not check.data:
            return jsonify({'error': 'Session not found'}), 404

        supabase_admin.table('chat_history').delete().eq('session_id', session_id).execute()
        supabase_admin.table('sessions').delete().eq('id', session_id).execute()

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sessions/<session_id>/messages', methods=['GET'])
def get_session_messages(session_id):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        check = supabase_admin.table('sessions').select('id').eq('id', session_id).eq('user_id', user_id).execute()
        if not check.data:
            return jsonify({'error': 'Session not found'}), 404

        result = supabase_admin.table('chat_history') \
            .select('prompt, response, created_at') \
            .eq('session_id', session_id) \
            .order('created_at', desc=False) \
            .execute()

        return jsonify({'messages': result.data})
    except Exception as e:
        print(f"❌ Error in get_session_messages: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/sessions/<session_id>/title', methods=['PATCH'])
def update_session_title(session_id):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        data = request.json
        title = data.get('title')
        if not title:
            return jsonify({'error': 'Title required'}), 400

        supabase_admin.table('sessions') \
            .update({'title': title}) \
            .eq('id', session_id) \
            .eq('user_id', user_id) \
            .execute()

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========================================
# CHAT SEARCH
# ========================================
@app.route('/api/sessions/search', methods=['GET'])
def search_sessions():
    """Search user's chat sessions by title + message prompts."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    query = (request.args.get('q') or '').strip()
    if len(query) < 2:
        return jsonify({'sessions': [], 'query': query})

    try:
        pattern = f"%{_escape_ilike(query)}%"

        # 1️⃣ Search sessions by title
        title_res = supabase_admin.table('sessions') \
            .select('id, title, created_at, updated_at') \
            .eq('user_id', user_id) \
            .ilike('title', pattern) \
            .order('updated_at', desc=True) \
            .limit(20) \
            .execute()
        title_matches = title_res.data or []
        title_match_ids = {s['id'] for s in title_matches}

        # 2️⃣ Search chat_history prompts
        history_res = supabase_admin.table('chat_history') \
            .select('session_id, prompt, created_at') \
            .eq('user_id', user_id) \
            .ilike('prompt', pattern) \
            .order('created_at', desc=True) \
            .limit(50) \
            .execute()
        history_matches = history_res.data or []

        # 3️⃣ First prompt that matched per session → snippet source
        snippet_by_session = {}
        for h in history_matches:
            sid = h.get('session_id')
            if sid and sid not in snippet_by_session:
                snippet_by_session[sid] = h.get('prompt', '')

        # 4️⃣ Fetch sessions that matched only via prompt (not title)
        extra_ids = [sid for sid in snippet_by_session if sid not in title_match_ids]
        extra_sessions = []
        if extra_ids:
            extra_res = supabase_admin.table('sessions') \
                .select('id, title, created_at, updated_at') \
                .eq('user_id', user_id) \
                .in_('id', extra_ids[:20]) \
                .execute()
            extra_sessions = extra_res.data or []

        # 5️⃣ Merge + attach snippet + sort by updated_at desc
        all_sessions = title_matches + extra_sessions
        for s in all_sessions:
            snip = snippet_by_session.get(s['id'])
            if snip:
                s['snippet'] = snip[:140]
        all_sessions.sort(key=lambda x: x.get('updated_at', ''), reverse=True)

        print(f"🔍 Search '{query}': {len(all_sessions)} matches", file=sys.stderr)
        sys.stderr.flush()

        return jsonify({
            'sessions': all_sessions[:30],
            'query': query,
            'total': len(all_sessions)
        })
    except Exception as e:
        print(f"❌ Search error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
# ========================================
# CHAT EXPORT
# ========================================
def _build_markdown_export(sess_data, messages):
    """Build a clean Markdown representation of a chat session."""
    title = sess_data.get('title') or 'New Chat'
    created = (sess_data.get('created_at') or '')[:19].replace('T', ' ')
    total = len(messages)

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"> **Exported from Spinal AI** · {datetime.now().strftime('%B %d, %Y')}")
    lines.append(f">")
    lines.append(f"> **Messages:** {total}  ·  **Started:** {created}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, m in enumerate(messages, 1):
        prompt = (m.get('prompt') or '').strip()
        response = (m.get('response') or '').strip()
        ts = (m.get('created_at') or '')[:19].replace('T', ' ')

        lines.append(f"### 💬 Exchange {i}")
        if ts:
            lines.append(f"*{ts}*")
        lines.append("")
        lines.append("**🧑 You**")
        lines.append("")
        lines.append(prompt)
        lines.append("")
        lines.append("**⚡ Spinal**")
        lines.append("")
        lines.append(response)
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append(f"*End of chat · {total} exchange{'s' if total != 1 else ''} · Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    return "\n".join(lines)

def _escape_ilike(text):
    """Escape special chars for Postgres ILIKE pattern."""
    return text.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')



def _sanitize_pdf_text(text):
    """Make text safe for fpdf2 built-in fonts (Latin-1). Replaces common emojis/symbols."""
    if not text:
        return ""
    replacements = {
        '✅': '[OK]', '❌': '[X]', '⚠️': '[!]', '🔥': '!!',
        '💡': '*', '🚀': '>>', '📷': '[cam]', '🎤': '[mic]',
        '🧠': '[brain]', '⚡': '!', '💬': '', '📄': '',
        '😊': ':)', '😄': ':D', '🙂': ':)', '😀': ':D',
        '❤️': '<3', '👍': '+1', '👎': '-1', '🎯': '[*]',
        '🌊': '~', '🛑': '[STOP]', '⏹️': '[stop]', '📸': '[cam]',
        '🖥️': '[screen]', '📤': '[>>]', '📥': '[<<]', '📦': '[pkg]',
        '—': '-', '–': '-', '“': '"', '”': '"', '‘': "'", '’': "'",
        '…': '...', '·': '.', '•': '*', '→': '->', '←': '<-',
        '✕': 'x', '✓': 'v', '✔': 'v', '★': '*', '☆': '*',
    }
    for emoji, repl in replacements.items():
        text = text.replace(emoji, repl)
    # Strip anything outside Latin-1
    text = text.encode('latin-1', 'replace').decode('latin-1')
    return text


def _build_pdf_export(sess_data, messages):
    """Build a professional, text-based PDF using fpdf2."""
    from fpdf import FPDF

    title = _sanitize_pdf_text(sess_data.get('title') or 'New Chat')
    created = (sess_data.get('created_at') or '')[:19].replace('T', ' ')

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    # ---- Header: brand ----
    pdf.set_text_color(255, 122, 26)
    pdf.set_font('Helvetica', 'B', 22)
    pdf.cell(0, 10, 'SPINAL', ln=True)

    pdf.set_text_color(108, 117, 125)
    pdf.set_font('Helvetica', '', 8)
    pdf.cell(0, 4, 'VISION-NATIVE AI  |  CHAT EXPORT', ln=True)

    pdf.ln(2)
    pdf.set_draw_color(255, 122, 26)
    pdf.set_line_width(0.8)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)

    # ---- Title ----
    pdf.set_text_color(11, 12, 14)
    pdf.set_font('Helvetica', 'B', 18)
    pdf.multi_cell(0, 8, title)
    pdf.ln(2)

    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(108, 117, 125)
    meta = f"{len(messages)} message{'s' if len(messages) != 1 else ''}  |  Started: {created}"
    pdf.cell(0, 5, _sanitize_pdf_text(meta), ln=True)
    pdf.ln(8)

    # ---- Messages ----
    for i, m in enumerate(messages, 1):
        prompt = (m.get('prompt') or '').strip()
        response = (m.get('response') or '').strip()
        ts = (m.get('created_at') or '')[:19].replace('T', ' ')

        # Exchange number badge
        pdf.set_fill_color(11, 12, 14)
        pdf.set_text_color(255, 122, 26)
        pdf.set_font('Helvetica', 'B', 8)
        pdf.cell(14, 6, f'#{i:02d}', fill=True, align='C')

        if ts:
            pdf.set_text_color(154, 160, 166)
            pdf.set_font('Helvetica', '', 8)
            pdf.cell(0, 6, f'  {_sanitize_pdf_text(ts)}', ln=True)
        else:
            pdf.ln(6)
        pdf.ln(1)

        # User message
        pdf.set_text_color(108, 117, 125)
        pdf.set_font('Helvetica', 'B', 8)
        pdf.cell(0, 5, 'YOU', ln=True)

        pdf.set_fill_color(255, 244, 234)
        pdf.set_text_color(26, 29, 32)
        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(0, 5, _sanitize_pdf_text(prompt), fill=True)
        pdf.ln(2)

        # AI message
        pdf.set_text_color(108, 117, 125)
        pdf.set_font('Helvetica', 'B', 8)
        pdf.cell(0, 5, 'SPINAL', ln=True)

        pdf.set_fill_color(248, 249, 250)
        pdf.set_text_color(26, 29, 32)
        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(0, 5, _sanitize_pdf_text(response), fill=True)
        pdf.ln(6)

        # Divider between exchanges
        if i < len(messages):
            pdf.set_draw_color(229, 231, 235)
            pdf.set_line_width(0.2)
            pdf.line(15, pdf.get_y(), 195, pdf.get_y())
            pdf.ln(5)

    # ---- Footer ----
    pdf.ln(4)
    pdf.set_draw_color(229, 231, 235)
    pdf.set_line_width(0.3)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)
    pdf.set_font('Helvetica', 'I', 7)
    pdf.set_text_color(154, 160, 166)
    footer = f'End of chat  |  {len(messages)} exchange{"s" if len(messages) != 1 else ""}  |  spinal.site'
    pdf.cell(0, 5, _sanitize_pdf_text(footer), align='C', ln=True)

    # ✅ FIX: fpdf2 returns bytearray; force to immutable bytes
    output = pdf.output()
    if isinstance(output, bytearray):
        output = bytes(output)
    return output


@app.route('/api/sessions/<session_id>/export/pdf', methods=['GET'])
def export_session_pdf(session_id):
    """Export a chat session as a text-based PDF."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        sess_res = supabase_admin.table('sessions') \
            .select('id, title, created_at, updated_at') \
            .eq('id', session_id).eq('user_id', user_id).execute()
        if not sess_res.data:
            return jsonify({'error': 'Session not found'}), 404
        sess_data = sess_res.data[0]

        msg_res = supabase_admin.table('chat_history') \
            .select('prompt, response, created_at') \
            .eq('session_id', session_id).eq('user_id', user_id) \
            .order('created_at', desc=False).execute()
        messages = msg_res.data or []

        # Build PDF — guaranteed to be bytes now
        pdf_data = _build_pdf_export(sess_data, messages)
        print(f"📕 PDF built: {len(pdf_data)} bytes, {len(messages)} messages", file=sys.stderr)
        sys.stderr.flush()

        raw_title = (sess_data.get('title') or 'chat').strip()
        safe_title = re.sub(r'[^\w\s\-]', '', raw_title)[:40].strip().replace(' ', '_') or 'chat'
        base_name = f"Spinal_{safe_title}_{datetime.now().strftime('%Y-%m-%d')}.pdf"

        # ✅ FIX: BytesIO + send_file — correct Content-Length, correct bytes
        return send_file(
            io.BytesIO(pdf_data),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=base_name,
            max_age=0
        )
    except Exception as e:
        print(f"❌ PDF export error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/sessions/<session_id>/export', methods=['GET'])
def export_session(session_id):
    """Export a chat session as Markdown or JSON."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    fmt = (request.args.get('format') or 'md').lower()
    if fmt not in ('md', 'json'):
        return jsonify({'error': 'Invalid format. Use: md or json'}), 400

    try:
        # Fetch session (ownership-checked)
        sess_res = supabase_admin.table('sessions') \
            .select('id, title, created_at, updated_at') \
            .eq('id', session_id) \
            .eq('user_id', user_id) \
            .execute()

        if not sess_res.data:
            return jsonify({'error': 'Session not found'}), 404
        sess_data = sess_res.data[0]

        # Fetch messages
        msg_res = supabase_admin.table('chat_history') \
            .select('prompt, response, created_at') \
            .eq('session_id', session_id) \
            .eq('user_id', user_id) \
            .order('created_at', desc=False) \
            .execute()
        messages = msg_res.data or []

        # Safe filename: Spinal_<title>_<YYYY-MM-DD>
        raw_title = (sess_data.get('title') or 'chat').strip()
        safe_title = re.sub(r'[^\w\s\-]', '', raw_title)[:40].strip().replace(' ', '_') or 'chat'
        date_str = datetime.now().strftime('%Y-%m-%d')
        base_name = f"Spinal_{safe_title}_{date_str}"

        if fmt == 'md':
            content = _build_markdown_export(sess_data, messages)
            print(f"📄 Markdown export: {base_name}.md ({len(messages)} messages)", file=sys.stderr)
            sys.stderr.flush()
            return Response(
                content,
                mimetype='text/markdown; charset=utf-8',
                headers={'Content-Disposition': f'attachment; filename="{base_name}.md"'}
            )

        # fmt == 'json'
        payload = {
            'app': 'Spinal AI',
            'exported_at': datetime.now().isoformat(),
            'session': {
                'id': sess_data['id'],
                'title': sess_data.get('title') or 'New Chat',
                'created_at': sess_data.get('created_at'),
                'updated_at': sess_data.get('updated_at'),
                'message_count': len(messages)
            },
            'messages': [
                {
                    'user': m.get('prompt', ''),
                    'assistant': m.get('response', ''),
                    'timestamp': m.get('created_at')
                }
                for m in messages
            ]
        }
        json_str = json.dumps(payload, indent=2, ensure_ascii=False)
        print(f"📦 JSON export: {base_name}.json ({len(messages)} messages)", file=sys.stderr)
        sys.stderr.flush()
        return Response(
            json_str,
            mimetype='application/json; charset=utf-8',
            headers={'Content-Disposition': f'attachment; filename="{base_name}.json"'}
        )

    except Exception as e:
        print(f"❌ Export error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ========================================
# DELETE USER DATA
# ========================================
@app.route('/delete_user_data', methods=['POST'])
def delete_user_data():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    try:
        supabase_admin.table('user_memories').delete().eq('user_id', user_id).execute()
        supabase_admin.table('chat_history').delete().eq('user_id', user_id).execute()
        supabase_admin.table('sessions').delete().eq('user_id', user_id).execute()
        supabase_admin.table('user_profiles').delete().eq('user_id', user_id).execute()
        supabase_admin.table('user_preferences').delete().eq('user_id', user_id).execute()
        return jsonify({'success': True})
    except Exception as e:
        print(f"❌ Delete error: {e}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500


# ========================================
# MEMORY API ENDPOINTS
# ========================================
@app.route('/api/memories', methods=['GET'])
def get_memories():
    """Get all memories for current user."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        result = supabase_admin.table('user_memories') \
            .select('id, content, category, importance, created_at, updated_at, access_count, is_pinned') \
            .eq('user_id', user_id) \
            .order('importance', desc=True) \
            .order('updated_at', desc=True) \
            .execute()
        return jsonify({'memories': result.data})
    except Exception as e:
        print(f"❌ Get memories error: {e}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500


@app.route('/api/memories/<memory_id>', methods=['DELETE'])
def delete_memory(memory_id):
    """Delete a specific memory."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        check = supabase_admin.table('user_memories') \
            .select('id') \
            .eq('id', memory_id) \
            .eq('user_id', user_id) \
            .execute()

        if not check.data:
            return jsonify({'error': 'Memory not found'}), 404

        supabase_admin.table('user_memories') \
            .delete() \
            .eq('id', memory_id) \
            .eq('user_id', user_id) \
            .execute()

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/memories/<memory_id>', methods=['PATCH'])
def update_memory(memory_id):
    """Edit memory content, category, importance. Re-embeds if content changes."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        # Verify ownership
        check = supabase_admin.table('user_memories') \
            .select('content, category, importance') \
            .eq('id', memory_id) \
            .eq('user_id', user_id) \
            .execute()

        if not check.data:
            return jsonify({'error': 'Memory not found'}), 404

        old = check.data[0]
        data = request.json or {}

        new_content = data.get('content', old['content']).strip()
        new_category = data.get('category', old['category'])
        new_importance = max(1, min(10, int(data.get('importance', old['importance']))))

        update_payload = {
            'content': new_content,
            'category': new_category,
            'importance': new_importance,
            'updated_at': datetime.now().isoformat()
        }

        # Re-embed only if content changed
        if new_content != old['content']:
            embedding = generate_embedding(new_content)
            if embedding:
                update_payload['embedding'] = embedding
                print(f"✅ Re-embedded edited memory", file=sys.stderr)
                sys.stderr.flush()

        supabase_admin.table('user_memories') \
            .update(update_payload) \
            .eq('id', memory_id) \
            .eq('user_id', user_id) \
            .execute()

        return jsonify({'success': True})
    except Exception as e:
        print(f"❌ Update memory error: {e}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500


@app.route('/api/memories/<memory_id>/pin', methods=['POST'])
def toggle_memory_pin(memory_id):
    """Pin/unpin memory to prevent auto-cleanup."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        check = supabase_admin.table('user_memories') \
            .select('is_pinned') \
            .eq('id', memory_id) \
            .eq('user_id', user_id) \
            .execute()

        if not check.data:
            return jsonify({'error': 'Memory not found'}), 404

        new_state = not check.data[0].get('is_pinned', False)

        supabase_admin.table('user_memories') \
            .update({'is_pinned': new_state}) \
            .eq('id', memory_id) \
            .eq('user_id', user_id) \
            .execute()

        return jsonify({'success': True, 'is_pinned': new_state})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/memories/stats', methods=['GET'])
def get_memory_stats_endpoint():
    """Get memory counts by category for badge and filters."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        result = supabase_admin.rpc('get_memory_stats', {
            'target_user_id': user_id
        }).execute()

        return jsonify(result.data or {'total': 0, 'by_category': {}})
    except Exception as e:
        return jsonify({'total': 0, 'by_category': {}}), 500


@app.route('/api/memories/cleanup', methods=['POST'])
def trigger_cleanup():
    """Manual cleanup trigger."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    deleted = cleanup_old_memories(user_id)
    return jsonify({'success': True, 'deleted': deleted})


# ========================================
# USER PREFERENCES ENDPOINTS
# ========================================
@app.route('/api/preferences', methods=['GET'])
def get_preferences():
    """Get user preferences (response_style + personality)."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        result = supabase_admin.table('user_preferences') \
            .select('response_style, personality') \
            .eq('user_id', user_id) \
            .limit(1) \
            .execute()

        if result.data and len(result.data) > 0:
            return jsonify({
                'response_style': result.data[0].get('response_style', 'balanced'),
                'personality': result.data[0].get('personality', 'friendly')
            })

        # No preferences yet — return defaults
        return jsonify({'response_style': 'balanced', 'personality': 'friendly'})
    except Exception as e:
        print(f"❌ Get preferences error: {e}", file=sys.stderr)
        return jsonify({'response_style': 'balanced', 'personality': 'friendly'}), 500


@app.route('/api/preferences', methods=['POST'])
def update_preferences():
    """Update user preferences (response_style + personality)."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        data = request.json or {}

        # ✅ Partial updates allowed — sirf bheja hua field change karo
        update_payload = {'updated_at': datetime.now().isoformat()}

        if 'response_style' in data:
            response_style = data['response_style']
            if response_style not in STYLE_INSTRUCTIONS:
                return jsonify({
                    'error': 'Invalid response_style. Must be: short, balanced, detailed'
                }), 400
            update_payload['response_style'] = response_style

        if 'personality' in data:
            personality = data['personality']
            if personality not in PERSONALITY_INSTRUCTIONS:
                return jsonify({
                    'error': 'Invalid personality. Must be: professional, friendly, motivational'
                }), 400
            update_payload['personality'] = personality

        if len(update_payload) == 1:  # sirf updated_at hai, kuch bhi change nahi
            return jsonify({'error': 'No preference field provided'}), 400

        # Upsert (insert or update)
        supabase_admin.table('user_preferences').upsert({
            'user_id': user_id,
            **update_payload
        }, on_conflict='user_id').execute()

        print(f"✅ Preferences updated: {list(update_payload.keys())}", file=sys.stderr)
        sys.stderr.flush()

        return jsonify({'success': True, **update_payload})
    except Exception as e:
        print(f"❌ Update preferences error: {e}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500


# ========================================
# SOCKET.IO
# ========================================
@socketio.on('connect')
def handle_connect():
    print('✅ Client connected', file=sys.stderr)
    sys.stderr.flush()
    emit('response', {'data': 'Spinal AI is ready...'})


@socketio.on('frame')
def handle_frame(data):
    try:
        # SECURITY: Server-side identity check — client-supplied userId pe trust NAHI karo
        session_user_id = session.get('user_id')
        client_user_id = data.get('userId')

        if not session_user_id:
            print("❌ Frame rejected: No session user_id", file=sys.stderr)
            sys.stderr.flush()
            emit('response', {'data': 'Error: Not authenticated. Please log in again.'})
            return

        # Agar client ne userId bheja hai, toh session se match hona chahiye
        if client_user_id and client_user_id != session_user_id:
            print(f"⚠️ Frame rejected: userId mismatch. session={session_user_id}, client={client_user_id}", file=sys.stderr)
            sys.stderr.flush()
            emit('response', {'data': 'Error: Authentication mismatch. Please log in again.'})
            return

        # Always trust the SERVER session, not the client
        user_id = session_user_id

        session_id = data.get('sessionId')
        if not session_id:
            session_res = supabase_admin.table('sessions').insert({
                'user_id': user_id,
                'title': 'New Chat'
            }).execute()
            if session_res.data:
                session_id = session_res.data[0]['id']
                emit('session_created', {'sessionId': session_id})
            else:
                emit('response', {'data': 'Error: Could not create session'})
                return

        image_data = data.get('image')  # may be None now
        user_prompt = data.get('prompt', 'Hello')
        # ---- Fetch last 5 chat history for context ----
        # ---- Fetch last 20 chat history for deeper context ----
        history_res = supabase_admin.table('chat_history') \
            .select('prompt, response') \
            .eq('user_id', user_id) \
            .eq('session_id', session_id) \
            .order('created_at', desc=True) \
            .limit(20) \
            .execute()
        history = history_res.data if history_res.data else []
        history.reverse()

        # ---- RETRIEVE RELEVANT MEMORIES ----
        memories = retrieve_memories(user_id, user_prompt)
        memory_context = format_memories_for_prompt(memories)
        if memories:
            print(f"🧠 Retrieved {len(memories)} memories for context", file=sys.stderr)
            sys.stderr.flush()

        # ---- Fetch user's response style preference ----
        # ---- Fetch user's response style + personality ----
        # ---- Fetch user's response style + personality ----
        response_style = get_user_response_style(user_id)
        style_instruction = build_style_instruction(response_style)
        personality = get_user_personality(user_id)
        personality_instruction = build_personality_instruction(personality)
        print(f"🎨 Response style: {response_style} | 🎭 Personality: {personality}", file=sys.stderr)
        sys.stderr.flush()

        # ---- Build full prompt ----
        # Order (ChatGPT-style): 
        # Layer 1 (System) → Layer 2 (Personality) → Layer 3 (Style) 
        # → Memory → History → Current question
        context_parts = [
            SYSTEM_PROMPT,
            personality_instruction,
            style_instruction,
        ]

        if memory_context:
            context_parts.append(memory_context)

        if history:
            session_context = "Recent conversation in this session:\n"
            for entry in history:
                session_context += f"User: {entry['prompt']}\nSpinal: {entry['response']}\n"
            context_parts.append(session_context)

        full_prompt = "\n\n".join(context_parts) + f"\n\nCurrent question: {user_prompt}"

        # ---- FILE ATTACHMENT (optional) ----
        file_path = data.get('filePath')
        file_mime = (data.get('fileMime') or '').lower()
        file_name = data.get('fileName') or 'file'
        attached_file_part = None
        text_file_content = None

        if file_path:
            if not file_path.startswith(f"{user_id}/"):
                print(f"⚠️ File path mismatch: {file_path} vs {user_id}", file=sys.stderr)
                sys.stderr.flush()
                file_path = None
            else:
                try:
                    file_bytes = supabase_admin.storage.from_(UPLOAD_BUCKET).download(file_path)
                    print(f"📎 Attached file loaded: {file_name} ({len(file_bytes)} bytes, {file_mime})", file=sys.stderr)
                    sys.stderr.flush()

                    if file_mime in GEMINI_FILE_MIMES:
                        attached_file_part = types.Part.from_bytes(
                            data=file_bytes,
                            mime_type='application/pdf' if file_mime == 'application/pdf' else file_mime
                        )
                    elif file_mime in TEXT_FILE_MIMES:
                        try:
                            text_file_content = file_bytes.decode('utf-8', errors='replace')
                        except Exception:
                            text_file_content = file_bytes.decode('latin-1', errors='replace')
                        if len(text_file_content) > MAX_TEXT_FILE_CHARS:
                            text_file_content = text_file_content[:MAX_TEXT_FILE_CHARS] + "\n\n[...truncated...]"
                    else:
                        print(f"⚠️ Unsupported mime for attachment: {file_mime}", file=sys.stderr)
                except Exception as e:
                    print(f"⚠️ Failed to load attached file: {e}", file=sys.stderr)
                    sys.stderr.flush()

        # ---- Inject text file content into prompt (if CSV/TXT/MD) ----
        if text_file_content:
            full_prompt = (
                f"[Attached file: {file_name}]\n"
                f"```\n{text_file_content}\n```\n\n"
                f"{full_prompt}"
            )

        # ---- Assemble contents (image optional) ----
        contents = [full_prompt]

        if image_data:
            try:
                header, encoded = image_data.split(',', 1)
                image_bytes = base64.b64decode(encoded)
                image_part = types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')
                contents.append(image_part)
                print("📸 Image attached to request", file=sys.stderr)
                sys.stderr.flush()
            except Exception as e:
                print(f"⚠️ Image decode failed, sending text-only: {e}", file=sys.stderr)
                sys.stderr.flush()

        if attached_file_part is not None:
            contents.append(attached_file_part)

        # ---- STREAMING: Signal start ----
        emit('response_start', {'sessionId': session_id})

        # ---- Stream chunks (with stop support) ----
        ai_text_parts = []
        stopped_by_user = False
        try:
            for chunk in call_gemini_streaming(contents):
                # Check if user requested stop
                if user_id in STOP_REQUESTS:
                    STOP_REQUESTS.discard(user_id)
                    stopped_by_user = True
                    print(f"🛑 Streaming stopped by user", file=sys.stderr)
                    sys.stderr.flush()
                    break

                ai_text_parts.append(chunk)
                emit('response_chunk', {'chunk': chunk})
        except Exception as e:
            print(f"❌ Stream error: {e}", file=sys.stderr)
            sys.stderr.flush()
        finally:
            # Cleanup just in case
            STOP_REQUESTS.discard(user_id)

        # Combine parts into final text
        ai_text = "".join(ai_text_parts)

        # Fallback: agar bilkul khaali hai toh placeholder
        if not ai_text.strip():
            ai_text = "Response was stopped." if stopped_by_user else "Sorry, I couldn't generate a response."

        print(f"🤖 AI Says (streamed): {ai_text[:100]}...", file=sys.stderr)
        sys.stderr.flush()

        # ---- Save chat history ----
        try:
            supabase_admin.table('chat_history').insert({
                'user_id': user_id,
                'session_id': session_id,
                'prompt': user_prompt,
                'response': ai_text
            }).execute()

            session_check = supabase_admin.table('sessions').select('title').eq('id', session_id).eq('user_id', user_id).execute()
            if session_check.data and session_check.data[0]['title'] == 'New Chat':
                new_title = user_prompt[:50] + ('...' if len(user_prompt) > 50 else '')
                supabase_admin.table('sessions').update({
                    'title': new_title,
                    'updated_at': datetime.now().isoformat()
                }).eq('id', session_id).eq('user_id', user_id).execute()
        except Exception as e:
            print(f"⚠️ Save history error: {e}", file=sys.stderr)

        # ---- Signal end with full text ----
        emit('response_end', {
            'data': ai_text,
            'sessionId': session_id
        })

        # ---- ASYNC MEMORY EXTRACTION (only if not stopped) ----
        if not stopped_by_user and ai_text.strip():
            memory_thread = threading.Thread(
                target=process_memories_async,
                args=(user_id, session_id, user_prompt, ai_text),
                daemon=True
            )
            memory_thread.start()

    except Exception as e:
        print(f"❌ Error in handle_frame: {e}", file=sys.stderr)
        emit('response', {'data': f'Error: {str(e)}'})


@socketio.on('stop_response')
def handle_stop_response(data=None):
    """User requested to stop the current AI response."""
    # SECURITY: Always use server session, never trust client
    user_id = session.get('user_id')

    if user_id:
        STOP_REQUESTS.add(user_id)
        print(f"🛑 Stop requested by user: {user_id}", file=sys.stderr)
        sys.stderr.flush()
    else:
        print(f"⚠️ Stop requested but no user_id available", file=sys.stderr)
        sys.stderr.flush()


if __name__ == '__main__':
    print("🚀 Starting Flask server...", file=sys.stderr)
    sys.stderr.flush()
    # Debug only in local dev; production mein gunicorn use hota hai
    socketio.run(app, debug=not IS_PRODUCTION, port=5000)