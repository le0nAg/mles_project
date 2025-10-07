# Project Description

This project aims to develop a machine learning model capable of performing handwritten digit recognition on embedded systems such as the Raspberry Pi Pico.

## Current Status

The following components have been implemented:
- [x] Raw data reading/writing
- [x] Data pre-processing
- [x] Feature extraction

### Debugging Serial Communication
```bash
sudo picocom -b 115200 /dev/ttyACM0
```

## TODO List

- [ ] Topological-preserving downsampling
- [ ] Script.py for fast image translation and feature analysis
- [ ] Performance optimizations (processing and memory)
- [ ] add a .gitignore 
