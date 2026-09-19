# Vertex synthetic document fixture

`document-marker.pdf` is an SDK-authored one-page PDF, generated with ReportLab
using invariant metadata. It contains no user data. Its expected contents are:

- Verification code: `ORCHID-7284`
- Preferred notebook color: `turquoise`

The integration prompt asks for the code and color without revealing either.
Text extraction was checked with pypdf and the rendered page inspected with
PyMuPDF. The fixture is used by `scripts/verify_vertex_multimodal_integration.py`.
Generating or rendering it is not a runtime dependency of the integration runner.

## Temporal video fixture

`color-sequence.mp4` is SDK-authored synthetic media with no user data: 128×128,
2 fps, six seconds, no audio, H.264/yuv420p. The first four frames are red, the
next four blue, and the final four green. It was generated from raw RGB bytes
using FFmpeg (provided temporarily by imageio-ffmpeg 0.6.0), then decoded back to
RGB to verify all twelve frames and their dominant channel. SHA256:
`e3d0cb9c90920279e9f1488f17c2bdf0bde9e3ba06179de1b4e31e9225c2d19b`.

The runner `scripts/verify_vertex_video_input_integration.py` requests the color
sequence without revealing its expected values. It sends the whole video inline
and validates the ordered JSON result. FFmpeg is not a runtime dependency of the
runner or SDK.

## Veo first-frame fixture

`red-circle-720p.png` is a deterministic 1280×720 RGB PNG: a centered red circle
of radius 160 pixels on white. It is encoded with standard-library struct/zlib
from synthetic pixels and contains no user data. SHA256:
`4dc1d177ef0e67e6f43f4a96d9bfe92f37d5a4e83be5db03028ad7651ee92781`.
The Veo runner accepts it through `--image` and binds its digest to the private
operation checkpoint. The generated video's binary checks do not establish
visual fidelity to this fixture.
