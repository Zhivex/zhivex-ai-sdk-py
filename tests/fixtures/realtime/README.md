# Realtime audio fixture

`orbit-seven.wav` is synthetic English speech: “Say only orbit seven.” It was
created locally with macOS `say`, Samantha voice, as mono PCM signed 16-bit audio
at 24 kHz. It contains no microphone recording or personal data.

The live round-trip test sends the waveform as its only user input and requires
both output audio and an assistant transcript containing the spoken marker.
The marker is never included in text instructions sent to the model. Evidence
records only the input PCM digest/size and boolean assertions, not response audio.

Reproduction on macOS:

```sh
say -v Samantha --file-format=WAVE --data-format=LEI16@24000 \
  -o tests/fixtures/realtime/orbit-seven.wav 'Say only orbit seven.'
```

`orbit-seven-16k.wav` contains the same synthetic phrase at Gemini's native input
rate, generated with the same command using `LEI16@16000`. Both files have one
channel and 16-bit samples. The runner sends 100 ms frames with trailing silence.
