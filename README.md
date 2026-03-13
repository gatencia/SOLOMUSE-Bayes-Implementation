# Hardcoded Live Improv

Streaming-first Python prototype for live improv:

- Captures mono audio in real time (`sounddevice` / PortAudio)
- Stores rolling audio in a thread-safe ring buffer
- Supports listen/play demo phases
- Estimates tempo + beat times from a rolling window (`librosa`)
- Detects chord timeline + infers key from rolling windows

## Install

```bash
python -m pip install -r requirements.txt
```

## Configuration

Edit [config.yaml](config.yaml) for sample rate, block size, input device, and timing.

## CLI

Run improv demo:

```bash
python -m hardcoded_improv.cli run --config config.yaml
```

List input devices:

```bash
python -m hardcoded_improv.cli --list-devices
```

Estimate tempo from live input (last 12 seconds):

```bash
python -m hardcoded_improv.cli tempo --seconds 12 --config config.yaml
```

The tempo command prints:

- Estimated BPM
- Beat times (seconds)

Estimate chords from live input (last 12 seconds):

```bash
python -m hardcoded_improv.cli chords --seconds 12 --config config.yaml
```

The chords command prints:

- Estimated BPM
- Inferred key
- Chord timeline with confidence

## Tests

```bash
python -m pytest -q
```

Includes:

- Ring buffer behavior tests
- WAV save test
- Tempo estimator tests on synthetic click tracks
- Chord detector tests (template matching and synthetic chord sequence)
