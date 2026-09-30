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
# GEMINI HELPER — Retry + Model Fallback + AFC Disabled
# ============================================================
# ============================================================
# GEMINI HELPER — Multi-Model Fallback + Smart Retry
# ============================================================
def call_gemini_with_retry(prompt, image_part, max_retries=2):
    """
    Gemini API call with:
    - AFC (Automatic Function Calling) disabled
    - Exponential backoff retry for 503 errors
    - Stable model fallback chain (overloaded models avoided)
    """
    config = types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
    )

    # ✅ Stable models — high-demand wale avoid kiye
    models_to_try = [
        'gemini-2.5-flash',          # ✅ Most stable, reliable, free tier available
        'gemini-3.5-flash-lite',     # ✅ Lightweight, less demand
        'gemini-3.5-flash',          # ✅ Fallback
        'gemini-3.1-flash-lite',     # ✅ Last resort
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

                # 503 UNAVAILABLE — transient, retry with short backoff
                if '503' in error_msg or 'UNAVAILABLE' in error_msg or 'high demand' in error_msg:
                    wait_time = (1.5 ** attempt) + 1  # 2s, 2.5s
                    print(f"⏳ {model_name} busy. Retry in {wait_time:.1f}s...", file=sys.stderr)
                    sys.stderr.flush()
                    time.sleep(wait_time)
                    continue

                # 429 RESOURCE_EXHAUSTED — quota exceeded, try next model
                elif '429' in error_msg or 'RESOURCE_EXHAUSTED' in error_msg:
                    print(f"⚠️ {model_name} quota exhausted. Trying next model...", file=sys.stderr)
                    sys.stderr.flush()
                    break

                # 404 NOT_FOUND — model doesn't exist, skip
                elif '404' in error_msg or 'NOT_FOUND' in error_msg:
                    print(f"❌ {model_name} not found. Trying next model...", file=sys.stderr)
                    sys.stderr.flush()
                    break

                # Other errors
                else:
                    print(f"❌ Error with {model_name}: {error_msg}", file=sys.stderr)
                    sys.stderr.flush()
                    break

    # All models failed
    print(f"❌ All models failed. Last error: {last_error}", file=sys.stderr)
    sys.stderr.flush()
    return "Sorry, the AI service is experiencing high demand. Please try again after some time."


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
        supabase_admin.table('chat_history').delete().eq('user_id', user_id).execute()
        supabase_admin.table('sessions').delete().eq('user_id', user_id).execute()
        supabase_admin.table('user_profiles').delete().eq('user_id', user_id).execute()
        return jsonify({'success': True})
    except Exception as e:
        print(f"❌ Delete error: {e}", file=sys.stderr)
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

        profile_res = supabase_admin.table('user_profiles').select('name').eq('user_id', user_id).execute()
        user_name = profile_res.data[0]['name'] if profile_res.data else None

        history_res = supabase_admin.table('chat_history') \
            .select('prompt, response') \
            .eq('user_id', user_id) \
            .eq('session_id', session_id) \
            .order('created_at', desc=True) \
            .limit(5) \
            .execute()
        history = history_res.data if history_res.data else []
        history.reverse()

        context = ""
        if user_name:
            context += f"The user's name is {user_name}.\n"
        if history:
            context += "Previous conversation from this session:\n"
            for entry in history:
                context += f"User: {entry['prompt']}\nSpinal: {entry['response']}\n"

        full_prompt = f"{context}\nCurrent question: {user_prompt}" if context else user_prompt
        # Prompt ko improve karo
        system_instruction = (
            "You are Spinal AI, a helpful and friendly AI companion. "
            "When user asks for code, ALWAYS respond with properly formatted markdown code blocks "
            "using ```language ... ``` syntax. Use headings, lists, and bold formatting for clarity. "
            "Keep responses concise but complete.\n\n"
        )

        full_prompt = f"{system_instruction}{context}\nCurrent question: {user_prompt}" if context else f"{system_instruction}{user_prompt}"
        

        # Decode image
        header, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        image_part = types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')

        # ✅ AI call with retry + fallback + AFC disabled
        ai_text = call_gemini_with_retry(full_prompt, image_part)
        print(f"🤖 AI Says: {ai_text}", file=sys.stderr)
        sys.stderr.flush()

        # Save name if user introduced themselves
        if user_prompt and ("my name is" in user_prompt.lower() or "call me" in user_prompt.lower() or "i am" in user_prompt.lower()):
            words = user_prompt.split()
            name_candidate = None
            for i, w in enumerate(words):
                if w.lower() == "name" and i + 1 < len(words):
                    name_candidate = words[i + 1].strip('.,!?')
                    break
                elif w.lower() in ["call", "am"] and i + 1 < len(words):
                    name_candidate = words[i + 1].strip('.,!?')
                    break
            if name_candidate:
                try:
                    supabase_admin.table('user_profiles').upsert({
                        'user_id': user_id,
                        'name': name_candidate,
                        'updated_at': datetime.now().isoformat()
                    }, on_conflict='user_id').execute()
                except Exception as e:
                    print(f"⚠️ Profile update error: {e}", file=sys.stderr)

        # Save chat history
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

        emit('response', {
            'data': ai_text,
            'sessionId': session_id
        })

    except Exception as e:
        print(f"❌ Error in handle_frame: {e}", file=sys.stderr)
        emit('response', {'data': f'Error: {str(e)}'})


if __name__ == '__main__':
    print("🚀 Starting Flask server...", file=sys.stderr)
    sys.stderr.flush()
    socketio.run(app, debug=True, port=5000)