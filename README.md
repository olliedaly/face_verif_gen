# face_verif_gen
## Face verification trained from scratch, tested for failure modes, and improved with generated data.

## Data loading

Training images per second with no model attached, from `benchmark_loader.py`: 50,000 random
training images, batches of 256, 8 workers. Measured on a Ryzen 5 7600 with 30 GB of RAM and a
SATA SSD (LUKS, btrfs).

| Source | Cold (from disk) | Warm (from RAM) |
|---|---:|---:|
| PNGs, decoded with Pillow | 8,600 | 16,900 |
| uint8 array, one read per image | 10,800 | 118,000 |

Cold runs drop the files from the page cache first. Directory entries stay cached, which flatters
the PNGs slightly. Reading the array through a memmap instead gave 800 cold and 5,900 warm:
page faults on random rows made Linux read about 2 MB of the file per 37 KB image.
