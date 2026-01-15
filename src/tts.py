from gtts import gTTS
import os
import time
from playsound import playsound

TEMP_AUDIO = "temp_audio.mp3"

def speak(text, lang="id"):
    """
    lang: 'id' (Indonesia) | 'jw' (Jawa)
    """
    tts = gTTS(text=text, lang=lang)
    tts.save(TEMP_AUDIO)
    playsound(TEMP_AUDIO)
    os.remove(TEMP_AUDIO)
