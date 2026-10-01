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

## Qwen LiveTranslate multi-speaker check

The optional Qwen check combines `orbit-seven-16k.wav` (Samantha) with another
synthetic voice, separated by one second of silence. No microphone recording is
used. Reproduce on macOS:

```sh
say -v Daniel --file-format=WAVE --data-format=LEI16@16000 \
  -o /tmp/qwen-second.wav 'The mission is orbit seven. Thank you for your help.'
python3 - <<'PY'
import wave
with wave.open('tests/fixtures/realtime/orbit-seven-16k.wav', 'rb') as wav:
    first = wav.readframes(wav.getnframes())
with wave.open('/tmp/qwen-second.wav', 'rb') as wav:
    second = wav.readframes(wav.getnframes())
    assert second, 'Synthetic voice generation failed'
with wave.open('/tmp/qwen-two-speakers.wav', 'wb') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(16000)
    wav.writeframes(first + bytes(32000) + second)
PY
```

Pass `--multi-speaker-fixture /tmp/qwen-two-speakers.wav` to
`scripts/smoke_qwen_translate.py`. This checks multiple completed source items,
translation and synthesized audio with dynamic cloning enabled. It does not score
speaker-identification accuracy or perceptual voice similarity.

The audiovisual acceptance check uses a synthetic solid-color JPEG. Reproduce:

```sh
python3 - <<'PY'
from pathlib import Path
Path('/tmp/qwen-context.ppm').write_bytes(b'P6\n64 64\n255\n' + bytes([20, 80, 170]) * 64 * 64)
PY
sips -s format jpeg /tmp/qwen-context.ppm --out /tmp/qwen-context.jpg
```

Pass `--image /tmp/qwen-context.jpg`. Acceptance of the image alongside translated
speech is a transport check, not an evaluation of visual grounding quality.
