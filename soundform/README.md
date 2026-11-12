# Soundform

Soundform is a desktop prototype for architectural design exploration. It reads a first-order Ambisonic recording, estimates how sound energy arrives over time, and grows an abstract 3D form in response. It is not a certified acoustic simulator, and it does not ask you to place sound sources or draw a site plan.

## What you can do

1. Open Soundform.
2. Import a four-channel first-order Ambisonic WAV, or write a synthetic example from the Audio panel.
3. Read the file's duration, sample rate, channel count, bit depth, and channel levels.
4. Confirm the channel convention, then press **Analyze Audio**.
5. Read the directional map, spectrum, timeline, and confidence.
6. Move the occupant zone and edit the design settings.
7. Press **Generate Architecture**.
8. Inspect the mesh in the 3D view and the heuristic before/after comparison.
9. Export OBJ or STL, export an analysis table, or save a project.

There is no map editor and no source-node tool. A four-channel file is not treated as Ambisonics until metadata names a convention or you choose one.

## Run

The environment used here is Python 3.10.10, which is compatible with the stack below. Python 3.11 or 3.12 is also suitable. Dependencies live in a virtual environment so they stay off the system interpreter.

```powershell
cd C:\Users\Owner\Documents\Cursor\gravel-field
soundform\.venv\Scripts\python.exe -m pip install -r soundform\requirements.txt
soundform\.venv\Scripts\python.exe -m soundform
```

Tests:

```powershell
soundform\.venv\Scripts\python.exe -m pytest soundform\tests
```

## A real recording

Soundform does not download or redistribute third-party audio. Use a file you are allowed to use.

The [Urban Soundscapes of the World](https://urban-soundscapes.org/recordings/) collection publishes research and education recordings. Each location can include a 4-channel first-order Ambisonic file and a separate binaural file. The project states that the Ambisonic files are **ACN/SN3D**, 48 kHz, 24-bit PCM. The archive is on Zenodo: [https://zenodo.org/records/10106181](https://zenodo.org/records/10106181).

1. Open the recordings page and follow the link to the database.
2. Download an `ambisonics` zip, for example `ambisonics_1to50.zip`. Do not import the binaural WAV. A binaural file is two channels and is rejected for spatial analysis.
3. Unzip and choose a file such as `R0001_segment_ambisonics.wav`.
4. In Soundform, use **Import WAV**.
5. If the file does not carry convention metadata, select **ACN / SN3D (AmbiX)**. That choice is the dataset's published convention. Soundform does not infer it from the channel count.
6. Azimuth 0° is the direction the recording rig called forward (toward the dominant source or view), not north. Use **Rotation offset** if you want a different design forward.
7. The dataset's published LAeq figures belong to the binaural recordings. They are not applied to the Ambisonic samples. Soundform shows relative level.

The rig used a Core Sound TetraMic, but the published Ambisonic files are already encoded ACN/SN3D. Soundform does not decode raw A-format capsule signals.

A local substitute, clearly marked as synthetic, is **Synthetic example** in the Audio panel. It writes a 3-second AmbiX file: forward for 1.5 s, then left and slightly elevated.

## Channel conventions

Internal analysis always uses SN3D components in this order:

| Component | Axis |
| --- | --- |
| W | omnidirectional pressure |
| X | microphone forward |
| Y | microphone left |
| Z | up |

Azimuth is 0° at forward and increases toward the left. Elevation is 0° on the horizontal plane and positive upward.

| Convention | File order | Conversion before analysis |
| --- | --- | --- |
| ACN / SN3D (AmbiX) | W, Y, Z, X | none |
| FuMa | W, X, Y, Z | multiply W by √2 |
| ACN / N3D | W, Y, Z, X | divide X, Y, Z by √3 |

A unit plane wave then satisfies `W = s` and `(X, Y, Z) = s · direction`. The active-intensity vector is `Re{ W · conj([X, Y, Z]) }`. For that scaling it points toward the source, and its magnitude matches `|W|²` for a single plane wave. Diffuse or out-of-phase bins are spread over the sphere instead of pinned to one cell.

Choosing the wrong convention swaps or mirrors the architectural response. No convention stores a compass bearing.

## Analysis

- Short-time Fourier transform, Hann window. The window is 2048, 4096, or 8192 samples depending on sample rate, with a half-window hop. Files that would exceed about 1500 frames use a coarser hop, and the interface says so.
- Default bands: low 20–250 Hz, mid 250–2000 Hz, high 2000–12000 Hz, clamped to Nyquist. Broadband used for the map starts at 20 Hz. The spin boxes change the next analysis; they do not silently rewrite an existing one.
- Each time-frequency tile adds its direct energy to one spherical cell and its diffuse remainder in proportion to solid angle. Tiles below 20 Hz are left out of the directional estimate because a short window localises them poorly.
- The timeline's dominant angle is the net intensity of that window. It is withheld when frequency bins disagree. The map can still show more than one arrival when they fall in different tiles.
- Colour scales are relative to the peak of the analysed interval. A quiet moment stays quieter than a loud one.
- Silence and near-silence stay undefined. No direction is invented.

A first-order tile cannot separate two sources that occupy it together, and it cannot recover distance or the room.

## Design frame

The site is an abstract circle or rectangle. The microphone is placed at the centre of the occupant zone because the recording only has direction at the capsule. Design +Y is forward when the rotation offset is 0. A positive offset rotates the decoded field counter-clockwise about up.

Moving the zone moves that listening point inside the site. It does not locate a source in a real street.

## Forms

The score at each direction is a weighted mix of relative broadband energy, the three bands, persistence, transient intensity, and directional confidence. Each term is divided by its own peak before weighting, and the weights are divided by their sum. Low frequency is not treated as automatically more important.

- **Terrain.** A grid rises toward the horizontal score. Smoothing is applied, then the occupant disc is returned to the minimum height so it is not buried.
- **Walls.** Segments stand off from the zone by the barrier spacing. Height follows the score, quieter directions can be omitted, and an entry is cut at the quietest direction. This arrangement is a directional occlusion heuristic, not a transmission-loss model.
- **Pavilion.** Walls and a roof enclose the zone. Louder directions get more wall. The quietest direction is an opening under a lintel. Over the occupant the roof stays at the crown. If the maximum height is below 2.2 m, the interface warns that standing clearance is gone.

## Protection readout

Rays are cast toward a listening point 1.5 m above the zone centre. A direction is "covered" when the mesh blocks the ray above the ground and before the listener. Exposure after is incoming relative energy times `(1 − coverage)`.

That number is not a decibel reduction, reverberation time, laboratory transmission loss, or compliance result. Reflection, diffraction, absorption, source distance, and materials are not in the recording and are not inferred.

## Projects and export

A project JSON stores the audio path, convention, analysis settings, and design parameters. Samples are not embedded. A sibling `.analysis.npz` stores the directional field, and a sibling `.mesh.obj` stores the last mesh. If the WAV has moved, Soundform also looks beside the project file. A missing recording is reported and the saved settings remain.

OBJ and STL export checks for finite vertices, triangular faces, and non-zero area, then reads the file back. Analysis CSV/JSON repeats the relative-level warning.

## Layout

```
soundform/
    main.py
    ui/            desktop panels and the 3D view
    audio/         WAV import, convention detection, monitor playback
    acoustics/     decode, STFT, directional energy, response, occlusion heuristic
    geometry/      site frame, terrain, walls, pavilion
    export/        OBJ/STL and project files
    models/        parameters
    tests/         synthetic FOA directions, geometry, and export
```

## Assumptions that affect the result

- Supported spatial input is first-order Ambisonics only: ACN/SN3D, FuMa, or ACN/N3D.
- Stereo, binaural, mono, surround, higher-order Ambisonics, and raw A-format are not converted into a fake first-order field.
- Monitor playback is a stereo fold (`left = W + Y + 0.25 X`, `right = W − Y + 0.25 X`) for headphones or speakers. The analyser does not use it.
- Sample rates outside 8 kHz–384 kHz are rejected.
- Levels are dBFS or normalized energy. They are not dB SPL.
- The three building modes are design responses to that directional score. They are buildable as meshes, not as predictions of insertion loss.
