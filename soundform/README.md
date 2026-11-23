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


## Cloud points and reconstruction

After creating a cloud from the selected pavilion origins and analyzing audio, set **Point energy threshold** (0–1) in **Cloud**, then generate surface points. Only areas meeting that relative-energy cutoff receive points. Zero includes all energy levels. Density controls still determine how many points appear in eligible areas. Change the cutoff and regenerate to apply it.

Expand **Delaunay mesh** or **Poisson reconstruction** to generate a separate mesh from the current points. Delaunay extracts the boundary of a 3D tetrahedralization: alpha 0 gives a convex hull, while positive alpha limits tetrahedron circumradius. Poisson uses oriented sample normals and a uniform-grid Poisson solver, with resolution 32–128; it smooths and may bridge unsampled areas. This implementation is not screened or adaptive-octree Poisson. Both methods run in the background and trim the result to the terrain. Each result appears in Layers. Changing cloud geometry or regenerating points clears previous reconstructions.

Every layer group in **Layers** has a visibility checkbox. Hiding a layer affects display only; it does not remove geometry or exclude it from analysis. Visibility survives layer-list rebuilds during the current session.


The Cloud panel has a collapsible **Voronoi mesh** section. Its button partitions the actual metaball surface into restricted Voronoi cells using existing cloud points as seeds. Each surface triangle is clipped to the nearest-site regions, so the result stays on the surface, follows concavities, and ends at the terrain intersection without creating interior walls or a bottom cap. Distances are Euclidean in 3D, rather than geodesic distances along the surface.

Voronoi controls:
- **Cell count limit:** maximum number of cloud points used as seeds (default 250). Zero uses all points. The actual number cannot exceed the available unique points. Subsampling preserves the sound-weighted distribution statistically.
- **Cell gap:** approximate separation in metres between cells (default 0.08 m). Zero tiles the full surface; larger values leave openings between panels. Large gaps can eliminate small panels.
- **Pattern seed:** repeatable random choice of seed points when subsampling. It has no effect when all points are used.
- **Color cells individually:** distinguish the cells visually, or turn it off to show relative sound energy.

Press Generate after changing settings. Voronoi results appear in their own layer and run in the background like other reconstructions.

Layer groups start collapsed. Both layers and individual objects have visibility checkboxes. Parent-layer visibility and individual-object visibility are independent, so hiding and showing a layer preserves which of its objects were hidden. Pavilion-part visibility affects display only and retains the underlying design for editing and analysis.


## Layer context menu

Right-click an object or layer in **Layers** for **Copy**, **Delete**, and **Paste into this layer**. Object commands act on that row; layer commands act on all objects directly inside the layer. Empty layers remain available as paste targets. Paste uses the clicked object's parent layer or the clicked layer itself, gives copies unique names, and keeps their original coordinates.

Imported site objects and pavilion origins remain functional when pasted into their corresponding layers. Pasted pavilion components in the Pavilion layer remain editable parts. Copied analysis spheres, clouds, surface point sets, and reconstructed meshes are independent geometry snapshots, so changing the live analysis will not regenerate those copies. Deleting live origins or terrain invalidates dependent cloud analysis. These scene changes are in memory; the original Rhino file is not edited.


## Connect cloud points

Expand **Connect points** in Cloud, then press **Generate connection lines**. Settings are **Maximum distance** in metres (default 1 m), **Minimum distance** (default 0 m), **Connections per point** (default 8), and **Total line limit** (default 100,000). Each line joins two existing cloud points within the inclusive distance range. Zero-length and duplicate edges are excluded. Nearby candidates are preferred; the connections-per-point value is an upper bound at both endpoints, not a target degree. Candidate searches are bounded to keep dense point clouds manageable, so this is a sparse network rather than every qualifying pair.

Connection checks are enabled by default:
- **Follow metaball surface** connects neighboring surface regions with mesh-based paths. A multi-source surface graph search establishes neighbors and routes, with direct links within a single triangle. Disconnected surface pieces never connect. Maximum/minimum distance refers to total path length, rather than the chord through the cloud. Routing is a discrete mesh approximation, not an exact continuous geodesic solver.
- **Limit by local point spacing** caps length by **Spacing multiplier** (default 2.5) times median nearby point spacing, using the denser endpoint. The absolute distance limit still applies.
- **Reject crossings and overlaps** rejects paths approaching an accepted path within **Crossing tolerance** (default 0.005 m), while permitting shared cloud endpoints. This tests 3D proximity, including collinear overlap, rather than crossings visible only in camera projection.

Disable checks to compare simpler networks. Mesh routes can have bends, and stricter checks can leave some points disconnected. The status bar reports generated paths and rejections by spacing/crossing checks. Connections retain energy colors and appear in their own **Connection lines** layer, with visibility and Copy/Delete support. Generating new cloud points clears the live connection network, alongside other reconstructions.

### Delaunay network
The Delaunay panel now builds an approximate intrinsic triangulation using the metaball's existing surface topology. Its only adjustable mesh parameter is the maximum surface-edge distance (default 2.5 m; zero means unlimited); the existing point positions supply the samples. Sound intensity controls point generation density only and is neither read by this triangulation nor used to color its network.

A bounded refinement of the reference mesh resolves geodesic Voronoi regions; their junctions define triangles between the existing points. Connections never cross between disconnected reference components. The distance cap removes whole triangles using conservative graph-route surface distance estimates, rather than distance through the cloud. Original point coordinates are retained, and the network is displayed in a single color.

This is an experimental surface approximation, not an exact global unwrap or a guaranteed intersection-free remesher. Straight triangle edges approximate curved surface routes; very sparse samples, tight folds, or limited reference resolution can leave holes or poor triangles. Refinement is bounded at 300,000 reference faces, so very dense samples may not all appear in the output. No automatic hole filling bridges distance-cap gaps. The former planar helper remains available for standalone calls without a reference surface; the app always requires the metaball surface.

### Circulation curves and floating panels
Rhino objects on Circulation (including sublayers) may be meshes or curves. Polylines retain their vertices; other curves are sampled adaptively with a 1 mm target chord tolerance and bounded refinement for viewport display. Curves retain their source names, local metre coordinates, layer visibility, and copy/paste behavior. They are imported as line geometry, not filled surfaces. Floating note panels hide individually when dismissed, and all close when the main window closes successfully.

### Cloud workflow
Cloud contains Metaball, Points, and Circulation sections. The origin table controls inclusion and per-origin radius scale; Generate metaball applies the selected membership and scales. Delaunay, Poisson and Voronoi tools live in Generate. Audio and Analysis share the upper-left sloping edge; Layers and Properties occupy the left vertical edge. Export occupies the lower-right sloping edge. Four direct view buttons run along the lower-left edge.

Points provides density, energy threshold, and merging distance (metres, default 0/off). Merging keeps the first existing surface sample per distance neighbourhood, with its metadata, so points stay on the surface and connected chains do not collapse. Circulation lists imported curve objects with individual pipe radii (default 2 m, 0 skips a curve). Subtract points removes samples within the 3D distance of the curve's sampled segments, including rounded end caps. Regenerate points to undo subtraction. Meshes are invalidated after point changes and must be regenerated.

Selecting an origin exposes editable X/Y/Z in Properties, in local site metres relative to the original imported reference origin. An edit moves that origin's analysis sphere/metaball contribution, updates the viewport, and clears dependent points and meshes. Editing these coordinates changes the running Soundform scene; it does not modify the source Rhino file.

### Voronoi maximum reach and sound colours
Generate > Voronoi provides Maximum cell reach (default 1.25 m; 0 means unlimited). It clips each cell to a straight-line 3D radius about its existing generating sample, leaving uncovered regions where no cell reaches. It is independent of panel gap. Reach crossings on the whole tile boundary are joined with straight cutting edges in a local surface plane; the clipped fragments retain the metaball surface. If the reach sits wholly inside a tile and has no boundary crossings, a square footprint is used. Tight curvature can introduce additional facets to stay within the 3D cap. This is Euclidean reach, not geodesic surface distance.

Color by center sound intensity is on by default. Every triangle belonging to a tile uses that tile's generating sample's relative energy (0-1), including when samples are subsampled by the cell limit. It uses a common magma scale, not random cell colours or interpolated vertex energy. Disabling the checkbox gives a uniform neutral material. Audio intensity does not change reach or cell geometry. Regenerate Voronoi after changing settings.

Preserve circulation openings is enabled by default in Generate > Voronoi. It subtracts the Circulation table pipe radii from the final tiles, independently of point subtraction, so neighbouring cells cannot refill entrances. Zero-radius curves are ignored. Pipes are approximated by conservative polygonal capsules with rounded ends and bends; clearance may be slightly wider than the exact round pipe. Cuts split crossing triangle interiors, preserve center energy and cell labels, and leave open boundary rims without filling circulation holes.

### Selection navigation and export
Middle-click frames the existing selection only; it never picks under the pointer and does nothing with no selection. File > Export selected exports selected scene geometry (including multiple selected objects in Layers). Rhino .3dm is the default and supports meshes, polylines and points in local site metres; tile center-energy colours are retained as mesh vertex colours. OBJ supports geometry and lines/points; STL accepts surfaces only. Analysis spheres now default to 3 m radius independently of pavilion dimensions; explicit per-origin radius overrides and radius scales remain effective.

Circulation cutting is batched against the original Voronoi triangles using a spatial index and vectorized separating-plane checks. Distant triangles are copied in bulk, straight redundant polyline samples are removed without simplifying bends, and surviving convex pieces are triangulated only after their nearby cuts. The final mesh is assembled and cleaned once, preserving tile labels and center-energy values. Representative 24-segment tests reduced the cutting stage from 2.21 s to 0.056 s for a straight path and 6.68 s to 0.269 s for a curved path, with matching remaining surface areas. Actual generation time depends on surface resolution and curve complexity.

### Hexagon frame and Export panel
A thin sage/charcoal hexagonal band surrounds the viewport, with Chrome-style tab shoulders joining its inner edge. Tab anchors remain fractional positions along each edge. Audio/Analysis are on the former Manual edge, Layers/Properties on the left vertical edge, and Export on the former Layers/Properties edge. The Manual UI and its manual-generation action were removed; shared design parameters remain in a nonvisual DesignState for viewport, project and AI use. Connect points and its controls were removed from Generate.

Export lists the scene's layers and objects with checkboxes. Checking a parent includes its objects; individual children can also be checked. All, None and Use selection shortcuts are available. The list refreshes as scene objects change. Choose Rhino .3dm, OBJ, or STL and press Export checked objects. STL requires surfaces; Rhino and OBJ support curves and point sets. File > Export selected remains available.

### Struts, nodes and solid tiles
Generate > Delaunay enables its thickness controls after the base mesh exists. Uniform sizing uses a strut/node radius (default 0.05 m). Sound-weighted sizing interpolates between minimum and maximum radius (defaults 0.02-0.15 m) using each original connected sample's relative energy, clipped to 0-1. Shared triangle edges generate only one strut. Weighted struts are capped tapered cylinders joining endpoint radii; node spheres use the same local radius. Primitives overlap at joints and are not Boolean-unioned. The low-poly primitives use 12-sided struts and 12-by-10 spheres to limit generation cost.

Generate > Voronoi enables uniform total thickness (default 0.10 m) or sound-weighted minimum/maximum thickness (defaults 0.03-0.30 m). Weight is the generating sample's relative energy; all faces of a tile keep its center colour. By default each tile projects onto the tangent plane of the metaball face at its generating point, then extrudes symmetrically by half the total thickness on each side. Preserve surface curvature retains the tile surface and offsets its vertices along averaged tile normals instead. Boundary walls close exterior rims and circulation holes without filling holes. Cut-edge T-junctions are split before wall construction. Large folded cells that cannot project safely request curvature mode; large offsets on tight curves can still self-intersect and require smaller thickness or tiles.

Outputs appear separately as Delaunay struts and spheres and Voronoi solid tiles in Layers and Export. Uniform and weighted buttons replace the same derived output. Rebuilding or deleting the source mesh clears its derived geometry; point changes clear all reconstructions. Generation runs in the background and discards results if their source changed. Each output is limited to four million triangles; dense clouds may need fewer source points.

### Drawing tools and active layer
Generate shares the upper-right edge with AI pavilion and Cloud. The right vertical edge has Point, Line and Spline tools. A horizontal Layer dropdown joins the viewport at that edge and may extend outside the outer band; it chooses the destination for newly drawn geometry. Available scene layers are listed alongside Drawing, and imported sites also expose Circulation, Pavilion Origin and Sound Sources.

Left-click adds points/control points to a draft. Right-click or Enter finishes; Escape cancels the entire uncommitted draft. Point mode may place several points before finishing; Line creates a line/polyline (at least two clicks); Spline creates a clamped control-point B-spline (at least three clicks). Preview curves and control markers are temporary. Switching tools cancels the previous draft. Right-drag orbit is reserved while drawing because right-click finishes.

Clicks snap to visible mesh surfaces when hit. Otherwise Top and Perspective use the XY ground plane, Front uses an XZ plane through the camera focus, and Right uses a YZ plane through that focus. Curves on Circulation join its clearance-radius table immediately; points on Pavilion Origin become metaball origins, and Sound Sources points join the source list. Other geometry becomes a selectable scene object in the chosen layer. Use Export to save newly drawn geometry; Rhino export retains true control-point splines, while OBJ uses their sampled polylines. Source Rhino files are never modified by drawing.


Viewport selection: left-click selects an object; left-drag draws a box that selects
whole objects fully inside it, including geometry behind other objects. Ctrl-drag
adds to the current selection. Selected objects are highlighted in gold and listed
as selected in Layers. Delete or Backspace removes the selection while the viewport
has keyboard focus. Drawing tools keep their click-to-place gestures while active.
The drawing layer selector sits above Point, Line, and Spline on the right edge and
shows only the selected layer name.
