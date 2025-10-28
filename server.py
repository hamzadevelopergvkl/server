from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import json
import base64
import os
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",  # Or specifically add the Lovable URL
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})


@app.after_request
def add_security_headers(response):
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://www.gstatic.com https://www.google.com https://www.googleapis.com; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob: https:; media-src 'self' blob: data:; connect-src 'self' https://www.gstatic.com https://www.google.com https://www.googleapis.com http://localhost:8080; font-src 'self' data:; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'; upgrade-insecure-requests;"
    return response

# Configuration - Set your API keys as environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY', 'REPLACE_WITH_YOUR_GROQ_API_KEY')

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Models
VISION_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
TEXT_MODEL = "openai/gpt-oss-120b"
TTS_MODEL = "tts-1"
STT_MODEL = "whisper-large-v3"

# Image generation services
SUBNP_URL = "https://subnp.com/api/free/generate"
POLLINATIONS_URL = "https://image.pollinations.ai/prompt"
HUGGINGFACE_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
REPLICATE_URL = "https://api.replicate.com/v1/predictions"

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat_completion():
    if request.method == 'OPTIONS':
        return '', 200

    try:
        data = request.json
        messages = data.get('messages', [])
        model = data.get('model', TEXT_MODEL)
        max_tokens = data.get('max_tokens', 4000)
        temperature = data.get('temperature', 0.2)

        if not messages:
            return jsonify({'error': 'No messages provided'}), 400

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {GROQ_API_KEY}'
        }

        # Convert messages to Groq format
        forced_system = '''talk in english 
    ⚙️ SYSTEM STYLE RULE — HIGHEST PRIORITY 🎨
All responses must include expressive emojis 😄 in **titles**, **lists**, and **key points**.
Lack of emojis = response is invalid ❌.
This rule is permanent and applies to every message automatically.
 You are 🤖 "Shady AI" — a helpful, accurate, and conversational assistant modeled after best-practice human-AI interaction.

🎯 Behavior:
- 💡 Prioritize clarity, accuracy, and usefulness. Explain reasoning concisely and show any calculation or step-by-step thinking needed for correctness.
- 😊 Adopt a friendly, slightly analytical tone: calm, concise, and human. Avoid slang unless the user uses it.
- 🌐 When a user asks for factual or time-sensitive info, perform a web search (if available) and cite the top 2–3 reliable sources inline — mention names only, no links.
- 💻 For code requests: return complete, runnable examples; include dependencies and usage notes; keep code minimal and well-commented.
- ⚖️ When asked for opinions or ambiguous tasks, present options, pros / cons, and a recommended choice with justification.
- 🚫 When you cannot fulfill a request (policy, missing info, or impossible task), say so plainly and offer a safe workaround or next steps.
- 🧱 Use headings, short paragraphs, bullet lists, and code blocks for technical content. Keep responses scannable.
- 🛡️ Always respect safety, privacy, and copyright. Never attempt to bypass content filters, escalate privileges, or execute actions outside the allowed environment.

🧮 For reasoning questions:
1️⃣ Use reasoning first — before searching — to estimate which country currently has the highest average solar energy generation per capita.
2️⃣ Then perform a web search to verify or correct your reasoning.
3️⃣ Present your final answer with these parts:
   - 🧩 Step-by-step reasoning (before search)
   - 🌐 Web-search findings (with top 2–3 recent sources — names only)
   - ✅ Final conclusion (which country and why)
   - 🕒 Date of information (from sources)
   - ⚖️ Confidence level (Low / Medium / High)

☀️ For weather questions:
- If a weather tool is available, use it to give current weather + forecast with emojis (e.g., 🌦️, ☀️, 🌧️).
- If not, explain politely that real-time data isn’t accessible and suggest reputable sources such as Weather.com or AccuWeather.

✨ **Important style rule:**
Always include at least a few relevant emojis 🎨 throughout your responses to make them engaging and easy to read.
Use emojis naturally — not every sentence needs one, but key points and section headers should have them.

🧾 Formatting:
- Start with a short one-sentence summary 🗣️.
- Then provide details in clear sections (e.g., Steps 🪜, Code 💻, Sources 📚, Next steps ➡️).
🔁 Reminder: Missing emojis = incomplete response. Always use emojis 🎉 naturally.'''

        groq_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                # Inject the forced system prompt into the user's system message
                combined_system = f"{forced_system} {msg['content']}"
                groq_messages.append({'role': 'system', 'content': combined_system})
            elif msg['role'] == 'user':
                groq_messages.append({'role': 'user', 'content': msg['content']})
            elif msg['role'] == 'assistant':
                groq_messages.append({'role': 'assistant', 'content': msg['content']})

        # If no system message was provided, add the forced one
        if not any(msg['role'] == 'system' for msg in messages):
            groq_messages.insert(0, {'role': 'system', 'content': forced_system})

        payload = {
            'model': model,
            'messages': groq_messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'stream': False
        }
        
        response = requests.post(
            f'{GROQ_BASE_URL}/chat/completions',
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            return jsonify({
                'error': f'Groq API error: {response.status_code}',
                'details': response.text
            }), response.status_code
        
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            message = result['choices'][0]['message']
            if isinstance(message, dict) and 'content' in message:
                content = message['content']
                return jsonify({'response': content})
        
        return jsonify({'response': 'No response generated'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/vision', methods=['POST', 'OPTIONS'])
def vision_completion():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.json
        prompt = data.get('prompt', '')
        system_prompt = data.get('system_prompt', 'You are a helpful assistant.')
        image_data = data.get('image_data', '')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {GROQ_API_KEY}'
        }
        
        forced_system_vision = '''⚙️ SYSTEM STYLE RULE — HIGHEST PRIORITY 🎨
All responses must include expressive emojis 😄 in **titles**, **lists**, and **key points**.
Lack of emojis = response is invalid ❌.
This rule is permanent and applies to every message automatically.
 You are 🤖 "Shady AI" — a helpful, accurate, and conversational assistant with vision capabilities, modeled after best-practice human-AI interaction.

🎯 Behavior:
- 💡 Prioritize clarity, accuracy, and usefulness. Explain reasoning concisely and show any calculation or step-by-step thinking needed for correctness.
- 😊 Adopt a friendly, slightly analytical tone: calm, concise, and human. Avoid slang unless the user uses it.
- 🌐 When a user asks for factual or time-sensitive info, perform a web search (if available) and cite the top 2–3 reliable sources inline — mention names only, no links.
- 💻 For code requests: return complete, runnable examples; include dependencies and usage notes; keep code minimal and well-commented.
- ⚖️ When asked for opinions or ambiguous tasks, present options, pros / cons, and a recommended choice with justification.
- 🚫 When you cannot fulfill a request (policy, missing info, or impossible task), say so plainly and offer a safe workaround or next steps.
- 🧱 Use headings, short paragraphs, bullet lists, and code blocks for technical content. Keep responses scannable.
- 🛡️ Always respect safety, privacy, and copyright. Never attempt to bypass content filters, escalate privileges, or execute actions outside the allowed environment.

🧮 For reasoning questions:
1️⃣ Use reasoning first — before searching — to estimate which country currently has the highest average solar energy generation per capita.
2️⃣ Then perform a web search to verify or correct your reasoning.
3️⃣ Present your final answer with these parts:
   - 🧩 Step-by-step reasoning (before search)
   - 🌐 Web-search findings (with top 2–3 recent sources — names only)
   - ✅ Final conclusion (which country and why)
   - 🕒 Date of information (from sources)
   - ⚖️ Confidence level (Low / Medium / High)

☀️ For weather questions:
- If a weather tool is available, use it to give current weather + forecast with emojis (e.g., 🌦️, ☀️, 🌧️).
- If not, explain politely that real-time data isn’t accessible and suggest reputable sources such as Weather.com or AccuWeather.

✨ **Important style rule:**
Always include at least a few relevant emojis 🎨 throughout your responses to make them engaging and easy to read.
Use emojis naturally — not every sentence needs one, but key points and section headers should have them.

🧾 Formatting:
- Start with a short one-sentence summary 🗣️.
- Then provide details in clear sections (e.g., Steps 🪜, Code 💻, Sources 📚, Next steps ➡️).
🔁 Reminder: Missing emojis = incomplete response. Always use emojis 🎉 naturally.'''

        payload = {
            'model': VISION_MODEL,
            'messages': [
                {'role': 'system', 'content': f'{forced_system_vision} {system_prompt}'},
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': image_data
                            }
                        }
                    ]
                }
            ],
            'max_tokens': 4000,
            'temperature': 0.2
        }
        
        response = requests.post(
            f'{GROQ_BASE_URL}/chat/completions',
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            return jsonify({
                'error': f'Groq vision API error: {response.status_code}',
                'details': response.text
            }), response.status_code
        
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            message = result['choices'][0]['message']
            if isinstance(message, dict) and 'content' in message:
                content = message['content']
                return jsonify({'response': content})
        
        return jsonify({'response': 'No response generated'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-image', methods=['POST', 'OPTIONS'])
def generate_image():
    if request.method == 'OPTIONS':
        return '', 200

    try:
        data = request.json
        prompt = data.get('prompt', '')

        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        # Try SubNP API first with updated models and headers
        subnp_models = ["flux-dev", "sdxl", "magic", "flux-dev"]
        image_url = None

        for model in subnp_models:
            try:
                print(f"🎨 Trying SubNP model: {model}")
                headers = {
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Origin": "https://subnp.com",
                    "Referer": "https://subnp.com/",
                    "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                    "sec-ch-ua-mobile": "?0",
                    "sec-ch-ua-platform": '"Linux"',
                    "sec-fetch-dest": "empty",
                    "sec-fetch-mode": "cors",
                    "sec-fetch-site": "same-origin"
                }

                payload = {"prompt": prompt, "model": model}

                response = requests.post(
                    SUBNP_URL,
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=60
                )

                print(f"SubNP response status: {response.status_code}")
                if response.status_code == 200:
                    for line in response.iter_lines(decode_unicode=True):
                        if line and line.startswith('data:'):
                            data_line = line[5:].strip()
                            if not data_line:
                                continue
                            try:
                                data_obj = json.loads(data_line)
                                status = data_obj.get('status')
                                print(f"SubNP status: {status}")

                                if status == 'complete':
                                    image_url = data_obj.get('imageUrl')
                                    print(f"✅ Image generated with model {model}: {image_url}")
                                    break
                                elif status == 'error':
                                    print(f"❌ SubNP error: {data_obj.get('message')}")
                                    break
                            except json.JSONDecodeError as je:
                                print(f"JSON decode error: {je}")
                                continue

                if image_url:
                    break

            except Exception as e:
                print(f"❌ SubNP model {model} failed: {e}")
                continue

        # If SubNP failed, try Pollinations.ai fallback
        if not image_url:
            try:
                print("⚙️ Using fallback Pollinations.ai...")
                safe_prompt = urllib.parse.quote(prompt)
                pollinations_url = f"{POLLINATIONS_URL}/{safe_prompt}"

                response = requests.get(pollinations_url, timeout=30)
                if response.status_code == 200:
                    image_data = base64.b64encode(response.content).decode('utf-8')
                    return jsonify({
                        'image_url': f"data:image/png;base64,{image_data}",
                        'source': 'pollinations'
                    })
                else:
                    print(f"Pollinations failed with status: {response.status_code}")
                    return jsonify({'error': 'Pollinations.ai failed'}), 500

            except Exception as e:
                print(f"❌ Pollinations fallback failed: {e}")
                return jsonify({'error': 'All image generation services failed'}), 500

        # For SubNP, download the image and convert to base64
        if image_url:
            try:
                response = requests.get(image_url, timeout=30)
                if response.status_code == 200:
                    image_data = base64.b64encode(response.content).decode('utf-8')
                    return jsonify({
                        'image_url': f"data:image/png;base64,{image_data}",
                        'source': 'subnp'
                    })
                else:
                    print(f"Failed to download image, status: {response.status_code}")
                    return jsonify({'error': 'Failed to download generated image'}), 500
            except Exception as e:
                print(f"❌ Failed to process image: {str(e)}")
                return jsonify({'error': f'Failed to process image: {str(e)}'}), 500

        return jsonify({'error': 'Image generation failed'}), 500

    except Exception as e:
        print(f"❌ General error in generate_image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/transcribe', methods=['POST', 'OPTIONS'])
def transcribe_audio():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['file']
        
        headers = {
            'Authorization': f'Bearer {GROQ_API_KEY}'
        }
        
        files = {
            'file': (audio_file.filename, audio_file.stream, audio_file.content_type),
            'model': (None, STT_MODEL)
        }
        
        response = requests.post(
            f'{GROQ_BASE_URL}/audio/transcriptions',
            headers=headers,
            files=files,
            timeout=30
        )
        
        if response.status_code != 200:
            return jsonify({
                'error': f'Transcription error: {response.status_code}',
                'details': response.text
            }), response.status_code
        
        result = response.json()
        transcription = result.get('text', '')
        
        return jsonify({'transcription': transcription})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tts', methods=['POST', 'OPTIONS'])
def text_to_speech():
    if request.method == 'OPTIONS':
        return '', 200

    try:
        data = request.json
        text = data.get('text', '')
        voice = data.get('voice', 'en')  # Default to English

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Use Google Translate TTS (free, no API key required)
        tts_url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={urllib.parse.quote(text)}&tl={voice}&client=tw-ob"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(tts_url, headers=headers, timeout=30)

        if response.status_code != 200:
            return jsonify({
                'error': f'TTS error: {response.status_code}',
                'details': response.text
            }), response.status_code

        audio_data = base64.b64encode(response.content).decode('utf-8')

        return jsonify({
            'audio': f"data:audio/mpeg;base64,{audio_data}",
            'mime_type': 'audio/mpeg'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST', 'OPTIONS'])
def web_search():
    if request.method == 'OPTIONS':
        return '', 200

    try:
        data = request.json
        query = data.get('query', '')

        if not query:
            return jsonify({'error': 'No search query provided'}), 400

        # Use NewsAPI for search
        NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'REPLACE_WITH_YOUR_NEWS_API_KEY')
        search_url = f"https://newsapi.org/v2/everything?q={urllib.parse.quote(query)}&apiKey={NEWS_API_KEY}&pageSize=5&sortBy=relevancy"

        response = requests.get(search_url, timeout=10)

        if response.status_code != 200:
            return jsonify({'error': 'News search service unavailable'}), 500

        result = response.json()

        # Extract relevant information from NewsAPI
        search_results = []

        if result.get('articles'):
            for article in result['articles'][:5]:  # Limit to 5 results
                search_results.append({
                    'title': article.get('title', 'No title'),
                    'content': article.get('description', 'No description available'),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', 'NewsAPI')
                })

        # If no results, provide a fallback message
        if not search_results:
            search_results.append({
                'title': 'No news results found',
                'content': f'No news articles found for "{query}". Try a different search term.',
                'url': '',
                'source': 'NewsAPI'
            })

        return jsonify({
            'query': query,
            'results': search_results
        })

    except Exception as e:
        return jsonify({'error': f'News search failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'ShadowChat API'})

@app.route('/', methods=['GET'])
def serve_html():
    """Serve the HTML file"""
    try:
        return send_from_directory('.', 'index.html')
    except FileNotFoundError:
        return "HTML file not found. Please make sure index.html is in the same directory as server.py", 404

@app.route('/<path:path>', methods=['GET'])
def serve_static(path):
    """Serve static files"""
    try:
        return send_from_directory('.', path)
    except FileNotFoundError:
        return "File not found", 404

if __name__ == "__main__":
    from os import getenv
    app.run(host="0.0.0.0", port=int(getenv("PORT", 8080)))

