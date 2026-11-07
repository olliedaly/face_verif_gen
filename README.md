# face_verif_gen
## Face verification trained from scratch, tested for failure modes, and improved with generated data.

## Data

CASIA-WebFace: where it comes from, its licence and its known problems are in the
[datasheet](docs/casia_datasheet.md).

## Data loading

Training images per second with no model attached, from `benchmark_loader.py`: two epochs through
the whole training split (422,091 images) in shuffled order, batches of 256, 8 workers. Median of
three runs on a Ryzen 5 7600 with 30 GB of RAM, 16 to 17 GB of it free, and a SATA SSD (LUKS,
btrfs).

| Source | Training split on disk | Epoch 1 (from disk) | Epoch 2 |
|---|---:|---:|---:|
| PNGs, decoded with Pillow | 8.2 GB | 8,900 | 18,600 |
| uint8 array, one read per image | 15.9 GB | 8,300 | 12,500 |

Epoch 1 starts with the files dropped from the page cache. By epoch 2 the PNGs are in RAM, but the
array is too big to stay there, so much of it is read from disk again. So from the second epoch on,
the PNGs are faster here; the array only wins on a machine with RAM to cache 16 GB. One of the
three PNG first epochs ran at 2,860, for a reason not yet found; the other two ran at about 9,000.

In an earlier test on 50,000 images, reading the array through a memmap gave only 800 images/s
from disk: page faults on random rows made Linux read about 2 MB of the file per 37 KB image.
