import os
import pickle

import numpy as np
from IPython.lib.display import Audio
from fastapi import FastAPI
from fastapi.responses import FileResponse
import keras
from music21 import converter, instrument, note, chord, duration, tempo, stream
from midi2audio import FluidSynth
import time
import collections

app = FastAPI()


def get_music_midi_filename_from_chords(input_chords):
    midi_stream = stream.Stream()

    for note_pattern, duration_pattern in input_chords:
        notes_in_chord = note_pattern.split('.')

        chord_notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(current_note)
            new_note.duration = duration.Duration(duration_pattern)
            new_note.storedInstrument = instrument.Violoncello()
            chord_notes.append(new_note)
        new_chord = chord.Chord(chord_notes)

        midi_stream.append(new_chord)

        new_tempo = tempo.MetronomeMark(number=50)

        midi_stream.append(new_tempo)

    midi_stream = midi_stream.chordify()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    new_file = 'output-' + timestr + '.mid'
    return midi_stream.write('midi', fp=new_file)


@app.get("/generate_audio")
async def generate_audio():
    processed_chords = pickle.load(open("chords.data", "rb"))

    # Define pruned_chord_data based on count condition
    pruned_chord_data = []
    count = collections.Counter(processed_chords)
    for c in processed_chords:
        if count[c] > 200:
            pruned_chord_data.append(c)

    # Define mapping_from_id dictionary
    c = list(set(pruned_chord_data))
    mapping_from_id = dict(zip(range(len(c)), c))
    mapping_to_id = dict(zip(c, range(len(c))))
    chord_id_data = [mapping_to_id[ele] for ele in pruned_chord_data]

    # Load the saved model
    model = keras.models.load_model("music_model.h5")

    generated_music_id = chord_id_data[:32]
    n_notes = len(mapping_from_id)

    for i in range(100):
        model_input = generated_music_id[-32:]

        model_input = np.reshape(model_input, (1, 32, 1)) / n_notes

        model_output = model.predict(model_input)
        model_output = model_output.argmax(axis=-1)[0]

        generated_music_id.append(model_output)

    generated_music_id = generated_music_id[32:]  # remove the random part of the song
    print(generated_music_id)

    generated_music = [mapping_from_id[ele] for ele in generated_music_id]
    generated_music_to_midi = get_music_midi_filename_from_chords(generated_music)

    midi_file_path = "generated_music.mid"

    with open(midi_file_path, "w") as file:
        file.write(generated_music_to_midi)

    audio_file_path = "music.wav"

    fluidsynth = FluidSynth()

    fluidsynth.midi_to_audio(midi_file_path, audio_file_path)

    os.remove(midi_file_path)

    generated_music_to_midi = get_music_midi_filename_from_chords(generated_music)
    FluidSynth().midi_to_audio(generated_music_to_midi, "music.wav")

    return FileResponse(audio_file_path, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
