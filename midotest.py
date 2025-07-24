import mido
import json
import matplotlib.pyplot as plt

def parseMidi(midi_path,output_json='tilemap.json'):
    midi = mido.MidiFile(midi_path)
    ticks_per_beat = midi.ticks_per_beat
    print(ticks_per_beat)

    active_notes  = {}
    tilemap = []

    current_time = 0
    tempo = 652164

    for msg in midi:
        current_time += mido.tick2second(msg.time,ticks_per_beat,tempo)
        if msg.type=='set_tempo':
            tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            active_notes[msg.note] = {
                'start_time': current_time,
                'velocity':msg.velocity }
        elif (msg.type == 'note_off') or (msg.type =='note_on' and msg.velocity == 0):
            if msg.note in active_notes:
                note_data = active_notes.pop(msg.note)
                duration = current_time - note_data['start_time']

                tilemap.append({
                    'time': round(note_data['start_time'],3),
                    "note": msg.note,
                    'velocity': note_data['velocity'],
                    'duration': round(duration,3)
                })
    for i, track in enumerate(midi.tracks):
        print(f'Track {i}: {track.name}')  # Accessing track name (a type of meta message)

        for msg in track:
            if msg.is_meta:  # Check if the message is a meta message
                # Process meta message based on its type
                if msg.type == 'track_name':
                    print(f"  Track Name: {msg.name}")
                elif msg.type == 'time_signature':
                    print(f"  Time Signature: {msg.numerator}/{msg.denominator}")
                elif msg.type == 'set_tempo':
                    print(f"  Tempo: {msg.tempo} microseconds per beat")
                # Add more elif conditions to handle other meta message types (e.g., key_signature, instrument_name, lyrics)
            else:
                # Handle regular MIDI messages (e.g., note_on, note_off)
                print(msg)
    tilemap.sort(key=lambda x: x['time'])
    with open (output_json, 'w') as f:
        json.dump(tilemap,f,indent=2)
        
    print(len(tilemap))
    print(tilemap)
    print(midi.length)

    times = [tile['time'] for tile in tilemap]
    plt.plot(times, range(len(times)))
    plt.xlabel("TIMES")
    plt.ylabel("NOTE INDEX")
    plt.title("Note timing")
    plt.show()
    return tilemap

parseMidi("bells.mid")
