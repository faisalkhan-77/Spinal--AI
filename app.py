import sys
print("🚀 Script started", file=sys.stderr)
sys.stderr.flush()

from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from flask_socketio import SocketIO, emit
import base64
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
import supabase
from datetime import datetime, timedelta
import uuid
import threading
import json
import re

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
IMPORTANCE_THRESHOLD = 8      # ✅ Memories with importance >= 8 always injected
DEDUP_THRESHOLD = 0.78          # ✅ Lowered — catches name corrections
DEDUP_HIGH_SIMILARITY = 0.90    # Force update threshold
MAX_MEMORIES_PER_QUERY = 5

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

        # 🔥 STEP 1: Get MULTIPLE candidates for dedup
        existing = supabase_admin.rpc('match_memories', {
            'query_embedding': embedding,
            'match_user_id': user_id,
            'match_threshold': DEDUP_THRESHOLD,
            'match_count': 5
        }).execute()

        # 🔥 STEP 2: Find best match (same category OR high similarity)
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
            # 🔄 UPDATE existing memory
            supabase_admin.table('user_memories').update({
                'content': content,
                'category': category,
                'importance': importance,
                'embedding': embedding,
                'updated_at': datetime.now().isoformat()
            }).eq('id', best_match['id']).execute()
            print(f"🔄 Memory UPDATED (sim={best_score:.3f}, cat={category}): {content[:50]}...", file=sys.stderr)
        else:
            # ✅ INSERT new memory
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


# Supabase Clients
if not SUPABASE_SERVICE_ROLE_KEY:
    SUPABASE_SERVICE_ROLE_KEY = SUPABASE_KEY

supabase_admin = supabase.create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

# ============= FLASK APP =============
app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY or "spinal-ai-fallback-secret-key-do-not-use-in-prod"

app.config.update(
    SECRET_KEY=app.secret_key,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False,
    SESSION_COOKIE_PATH='/',
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
)

socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*", manage_session=False)


# ============================================================
# GEMINI HELPER — Multi-Model Fallback + Smart Retry
# ============================================================
def call_gemini_with_retry(prompt, image_part, max_retries=2):
    """Gemini API call with fallback chain + retry."""
    config = types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
    )

    models_to_try = [
        'gemini-2.5-flash',
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

                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, image_part],
                    config=config
                )

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

def call_gemini_streaming(prompt, image_part):
    """
    Generator — yields text chunks from Gemini streaming API.
    Fallback chain if model fails BEFORE first chunk.
    """
    config = types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
    )

    models_to_try = [
        'gemini-2.5-flash',
        'gemini-3.5-flash-lite',
        'gemini-3.5-flash',
        'gemini-3.1-flash-lite',
    ]

    for model_name in models_to_try:
        first_chunk_received = False
        try:
            print(f"🌊 Streaming with {model_name}...", file=sys.stderr)
            sys.stderr.flush()

            stream = client.models.generate_content_stream(
                model=model_name,
                contents=[prompt, image_part],
                config=config
            )

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
# DEBUG ROUTE
# ========================================
@app.route('/debug_session')
def debug_session():
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
        # ✅ Auto-cleanup trigger (runs once per day)
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
            .select('id, content, category, importance, created_at, updated_at, access_count') \
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
        user_id = data.get('userId')
        if not user_id:
            emit('response', {'data': 'Error: Not authenticated'})
            return

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

        image_data = data['image']
        user_prompt = data.get('prompt', 'Describe exactly what you see in this image in ONE short, friendly sentence.')

        # ---- Fetch last 5 chat history for context ----
        history_res = supabase_admin.table('chat_history') \
            .select('prompt, response') \
            .eq('user_id', user_id) \
            .eq('session_id', session_id) \
            .order('created_at', desc=True) \
            .limit(5) \
            .execute()
        history = history_res.data if history_res.data else []
        history.reverse()

        # ---- RETRIEVE RELEVANT MEMORIES ----
        memories = retrieve_memories(user_id, user_prompt)
        memory_context = format_memories_for_prompt(memories)
        if memories:
            print(f"🧠 Retrieved {len(memories)} memories for context", file=sys.stderr)
            sys.stderr.flush()

        # ---- Build full prompt ----
        context_parts = []

        if memory_context:
            context_parts.append(memory_context)

        if history:
            session_context = "Recent conversation in this session:\n"
            for entry in history:
                session_context += f"User: {entry['prompt']}\nSpinal: {entry['response']}\n"
            context_parts.append(session_context)

        if context_parts:
            full_prompt = "\n\n".join(context_parts) + f"\n\nCurrent question: {user_prompt}"
        else:
            full_prompt = user_prompt

        # ---- Decode image ----
        header, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        image_part = types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')

        # ---- Call Gemini with retry ----
        # ---- 🌊 STREAMING: Signal start ----
        emit('response_start', {'sessionId': session_id})

        # ---- Stream chunks ----
        ai_text_parts = []
        try:
            for chunk in call_gemini_streaming(full_prompt, image_part):
                ai_text_parts.append(chunk)
                emit('response_chunk', {'chunk': chunk})
        except Exception as e:
            print(f"❌ Stream error: {e}", file=sys.stderr)
            sys.stderr.flush()

        ai_text = "".join(ai_text_parts)
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

        # ---- 🌊 Signal end with full text ----
        emit('response_end', {
            'data': ai_text,
            'sessionId': session_id
        })
        
        # ---- ASYNC MEMORY EXTRACTION ----
        memory_thread = threading.Thread(
            target=process_memories_async,
            args=(user_id, session_id, user_prompt, ai_text),
            daemon=True
        )
        memory_thread.start()

    except Exception as e:
        print(f"❌ Error in handle_frame: {e}", file=sys.stderr)
        emit('response', {'data': f'Error: {str(e)}'})


if __name__ == '__main__':
    print("🚀 Starting Flask server...", file=sys.stderr)
    sys.stderr.flush()
    socketio.run(app, debug=True, port=5000)