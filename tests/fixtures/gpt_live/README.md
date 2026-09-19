# Synthetic GPT-Live fixtures

Mono PCM16, 24 kHz, generated with macOS `say -v Samantha` and converted with
`afconvert -f WAVE -d LEI16@24000 -c 1`. No microphone or personal recordings.

- `delegation.wav`: “Ask your backend to look up the secret launch code, then tell me the code.”
- `interrupt.wav`: “Stop counting. Say only lunar nine.”

The launch code is absent from the user input: only the backend lookup supplies it.
The interruption recording is sent only after output audio has started.
