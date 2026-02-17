Python conversion of the C++ simulation sources in ../src.

Converted entry points:
- atom.py                      <- src/atom.cpp
- atom_realtime.py             <- src/atom_realtime.cpp
- atom_raytracer.py            <- src/atom_raytracer.cpp
- wave_atom_2d.py              <- src/wave_atom_2d.cpp
- ogabooga.py                  <- src/ogabooga.cpp
- schrodinger.py               <- src/schrodinger.py
- file_saves/2D_bohr_rutherford.py      <- src/file_saves/2D_bohr_rutherford.cpp
- file_saves/orbital_visualizer_raw.py  <- src/file_saves/orbital_visualizer_raw.cpp
- file_saves/raw_3d_generation.py       <- src/file_saves/raw_3d_generation.cpp

Shared helpers:
- common_quantum.py

Install deps:
- pip install -r requirements.txt

Run examples:
- python atom.py
- python atom_realtime.py
- python atom_raytracer.py
- python file_saves/orbital_visualizer_raw.py
- python file_saves/raw_3d_generation.py
