# Environment Objects

This folder stores reusable environment objects for Mode A.

Current objects:
- `clear_sky_earth.json` - daylight Earth sky gradient (background object)
- `deep_space.json` - deep-space starfield (background object)
- `sun_earth_view.json` - sun disk/glow seen from Earth (sun object)
- `baseplate_default.json` - default baseplate scene cube (no texture)
- `cube_default.json` - default cube scene cube (no texture)
- `light_default.json` - default key light object
- `camera_default.json` - default camera object
- `scene_defaults.json` - list of scene object files auto-loaded at app startup

You can add more object files here and wire them into commands/UI as needed.

Mode-A editor hotkeys:
- `G` move, `R` rotate, `S` scale selected object
- `X` / `Y` constrain current transform axis
- `Enter` confirm transform, `Esc` cancel
- `Shift+D` duplicate selected object
- `Ctrl+Z` undo, `Ctrl+Y` redo
- `Shift+Tab` toggle snapping

3D notes:
- Object cache entries are normalized as 3D with `dimensionality: "3d"` and `z` depth.
- Right mouse drag rotates the camera (mouse-look).
