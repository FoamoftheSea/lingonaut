from concurrent.futures import ThreadPoolExecutor
import os
import pyaudio
import wave
from tempfile import TemporaryDirectory

import ollama
import torch
from TTS.api import TTS


def play_audio(file_path):
    with wave.open(file_path, "rb") as wf:
        p = pyaudio.PyAudio()

        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )

        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)

    stream.stop_stream()
    stream.close()
    p.terminate()


# Get device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Init TTS
voices = [
    'Claribel Dervla',  # Deep-voice woman
    'Daisy Studious',  # Child girl
    'Gracie Wise',  # British woman
    'Tammie Ema',  # Forceful woman
    'Alison Dietlinde',  # Very brit woman
    'Ana Florence',  # Regular woman
    'Annmarie Nele',
    'Asya Anara',
    'Brenda Stern',
    'Gitta Nikolina',
    'Henriette Usha',
    'Sofia Hellen',
    'Tammy Grit',
    'Tanja Adelina',
    'Vjollca Johnnie',
    'Andrew Chipper',
    'Badr Odhiambo',
    'Dionisio Schuyler',
    'Royston Min',
    'Viktor Eka',
    'Abrahan Mack',
    'Adde Michal',
    'Baldur Sanjin',
    'Craig Gutsy',
    'Damien Black',
    'Gilberto Mathias',
    'Ilkin Urbano',
    'Kazuhiko Atallah',
    'Ludvig Milivoj',
    'Suad Qasim',
    'Torcull Diarmuid',
    'Viktor Menelaos',
    'Zacharie Aimilios',
    'Nova Hogarth',
    'Maja Ruoho',
    'Uta Obando',
    'Lidiya Szekeres',
    'Chandra MacFarland',
    'Szofi Granger',
    'Camilla Holmström',
    'Lilya Stainthorpe',
    'Zofija Kendrick',
    'Narelle Moon',
    'Barbora MacLean',
    'Alexandra Hisakawa',
    'Alma María',
    'Rosemary Okafor',
    'Ige Behringer',
    'Filip Traverse',
    'Damjan Chapman',
    'Wulf Carlevaro',
    'Aaron Dreschner',
    'Kumar Dahl',
    'Eugenio Mataracı',
    'Ferran Simen',
    'Xavier Hayasaka',
    'Luis Moray',
    'Marcos Rudaski'
]
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
# tts = TTS(model_name="tts_models/eng/fairseq/vits").to(device)


def process_stream(user_input):
    stream = ollama.chat(
        model='mistral:lingonaut',
        messages=[{'role': 'user', 'content': user_input}],
        stream=True,
    )

    with ThreadPoolExecutor(max_workers=1) as play_pool:
        with TemporaryDirectory() as tmp:
            def dump_to_audio(current_sentence):
                sentence = "".join(current_sentence)
                tts.tts_to_file(
                    text=sentence,
                    speaker=voices[10],
                    language="en",
                    file_path=wav_path,
                    split_sentences=False,
                )
                play_pool.submit(play_audio, wav_path)

            current_sentence = []
            for i, chunk in enumerate(stream):
                wav_path = os.path.join(tmp, f"{i}.wav")
                text_chunk = chunk['message']['content']
                print(text_chunk, end="", flush=True)
                text_chunk = text_chunk.replace('"', "").replace("(", "").replace(")", "")
                if len(text_chunk) == 0:
                    continue
                elif text_chunk in [",", ".", "!", "?"]:
                    current_sentence[-1] += text_chunk
                    if len(current_sentence) > 10:
                        dump_to_audio(current_sentence)
                        current_sentence = []
                else:
                    current_sentence.append(text_chunk)

            if len(current_sentence) > 0:
                dump_to_audio(current_sentence)

            play_pool.shutdown(wait=True)

# List available 🐸TTS models
# print(TTS().list_models())


if __name__ == "__main__":
    process_stream("How do you say that you're going to bed in Russian?")
