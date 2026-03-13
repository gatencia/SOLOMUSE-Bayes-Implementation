# Hardcoded Live Improv

Streaming-first Python prototype for live improv:

- Captures mono audio in real time (`sounddevice` / PortAudio)
- Stores rolling audio in a thread-safe ring buffer
- Supports listen/play demo phases
- Estimates tempo + beat times from a rolling window (`librosa`)
- Detects chord timeline + infers key from rolling windows
- Generates rule-based pentatonic improv and outputs real-time MIDI
- Trains/uses a lightweight Bayesian model for next-note choice

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

Run baseline pentatonic improv:

```bash
python -m hardcoded_improv.cli improv --listen-bars 2 --play-bars 8 --config config.yaml
```

Dry-run mode (no MIDI output):

```bash
python -m hardcoded_improv.cli improv --listen-bars 2 --play-bars 8 --dry-run --seed 42 --config config.yaml
```

Use a trained Bayesian model during improv:

```bash
python -m hardcoded_improv.cli improv --listen-bars 2 --play-bars 8 --bayes-model model.json --config config.yaml
```

Train Bayesian model from MIDI solos:

```bash
python -m hardcoded_improv.cli train-bayes --midi-dir ./midi_solos --out model.json
```

End-to-end live demo (input audio → BPM/chords → MIDI solo + artifacts):

```bash
python -m hardcoded_improv.cli live --midi-port "<your-port>" --listen-bars 2 --play-bars 16 --output-mid --config config.yaml
```

Simulation mode (WAV input, CI-friendly, always writes artifacts):

```bash
python -m hardcoded_improv.cli live --input-wav ./example.wav --listen-bars 2 --play-bars 8 --artifacts-dir ./artifacts/sim --config config.yaml
```

Select Scarlett + MIDI port:

- List audio inputs and use your Scarlett name/index in YAML `input_device` (or rely on auto-match):

```bash
python -m hardcoded_improv.cli --list-devices
```

- List MIDI outputs and copy exact port name for `--midi-port`:

```bash
python -m hardcoded_improv.cli --list-midi-ports
```

## Tests

```bash
python -m pytest -q
```

Includes:

- Ring buffer behavior tests
- WAV save test
- Tempo estimator tests on synthetic click tracks
- Chord detector tests (template matching and synthetic chord sequence)
- Scale utility tests
- Improv engine/event scheduling order tests
- Bayesian model training/sampling test with synthetic MIDI
- Integration test for simulation live demo artifacts
