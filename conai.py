import os
import sys
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
from elevenlabs.play import play
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv


class VoiceAI:

    def __init__(self, gemini_api_key, elevenlabs_api_key, voice_id):

        try:
            # Configure Gemini
            genai.configure(api_key=gemini_api_key)

            # Configure ElevenLabs
            self.elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)
            self.elevenlabs_voice_id = voice_id

        except Exception as e:
            print(f"Error configuring APIs: {e}")
            sys.exit(1)

        # Speech recognition + fallback TTS
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.fallback_tts_engine = pyttsx3.init()

        # System prompt
        system_prompt = (
            "You are a friendly and helpful voice assistant. "
            "Respond naturally, concisely, and conversationally. "
            "Do not act like a language model, act like a person."
        )

        # Gemini setup
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-pro",
            system_instruction=system_prompt
        )
        self.chat_session = self.model.start_chat(history=[])

        # Microphone calibration
        print("Calibrating microphone, please wait...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
        print("Microphone calibrated. Ready to chat!")

    def listen(self):
        try:
            with self.microphone as source:
                print("\nListening...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=8)

            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            print("Sorry, I could not understand that.")
            return None
        except Exception as e:
            print(f"Listening error: {e}")
            return None

    def get_ai_response(self, user_input):

        try:
            print("Thinking...")
            response = self.chat_session.send_message(user_input)
            ai_text = response.text.strip()
            print(f"AI says: {ai_text}")
            return ai_text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return "trouble responding."

    def speak(self, text):
        try:
            print("Speaking with ElevenLabs...")
            audio = self.elevenlabs_client.text_to_speech.convert(
                voice_id=self.elevenlabs_voice_id,
                optimize_streaming_latency="0",
                output_format="mp3_44100_128",
                text=text,
                model_id="eleven_multilingual_v2"
            )
            play(audio)
        except Exception as e:
            print(f"ElevenLabs error: {e}")
            print("-> Using fallback pyttsx3 TTS.")
            self.fallback_speak(text)

    def fallback_speak(self, text):
        self.fallback_tts_engine.say(text)
        self.fallback_tts_engine.runAndWait()

    def run(self):
        print("\n" + "=" * 50)
        print("Voice AI Agent Activated")
        print("Say something to start chatting.")
        print("Say 'quit', 'exit', or 'goodbye' to stop.")

        while True:
            user_text = self.listen()
            if not user_text:
                continue

            if user_text.lower().strip() in ["quit", "exit", "stop", "goodbye"]:
                farewell = "Goodbye! It was nice talking to you."
                print(f"{farewell}")
                self.speak(farewell)
                print("Session ended.")
                break

            ai_response = self.get_ai_response(user_text)
            self.speak(ai_response)


def main():
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY")
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")

    if not gemini_key or not elevenlabs_key or not voice_id:
        print("Missing API keys or voice ID. Please check your .env file:")
        print("GEMINI_API_KEY=...")
        print("ELEVENLABS_API_KEY=...")
        print("ELEVENLABS_VOICE_ID=...")
        sys.exit(1)

    try:
        ai_agent = VoiceAI(gemini_api_key=gemini_key,
                           elevenlabs_api_key=elevenlabs_key,
                           voice_id=voice_id)
        ai_agent.run()
    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
