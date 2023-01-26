# pento-diaref

# Reproduction

## Preparation

Checkout the repository

Install the requirements:

```
pip install -r requirements.txt
```

For all commands we assume that you are in the top level project directory and executed in before:

```
source prepare_path.sh
```

## Data Generation

### Create the training and testing annotations for the `DIDACT` dataset

Create the data directory at a path of your choice.

```
mkdir -p /data/pento_diaref/didact
```

And copy the required files into the directory

```
cp resources/* /data/pento_diaref/didact
```

Then execute the script

```
python3 scripts/generate_annos_didactic.py \
    --data_dir /data/pento_diaref/didact \
    --train_num_sets_per_utterance_type 10 \
    --test_num_sets_per_utterance_type 1 \
    --gid_start 0
```

This will create `148,400/10,000/10,000` in-distribution samples for training/validation/testing
and the `756/840/840` out-of-distribution (holdout target piece symbols) samples
for the color, position and utterance type generalization tests.

The script additionally filters out training samples where the extra target selection accidentally
produced a sample that has an utterance type reserved for the uts-holdout. So the remaining number
of training samples is probably between 120k-130k.

Note: During training, we only use the in-distribution validation samples for model selection.

### Create the training and testing annotations for the `NAIVE` dataset

Create the data directory at a path of your choice.

```
mkdir -p /data/pento_diaref/naive
```

And copy the required files into the directory

```
cp resources/* /data/pento_diaref/naive
```

Then execute the script

```
python3 scripts/generate_annos_naive.py -ho \
    --data_dir /data/pento_diaref/naive \
    --with_ho \
    --gid_start 1_000_000
```

This will create `148,400/10,000/10,000` in-distribution samples for training/validation/testing
using the same target piece symbols as above. For generalization testing we use the holdouts splits generated above.

Note: The holdouts computation is deterministic and only depends on the order
in the color, shape and position listings because we use `itertools.product(colors, shapes, positions)`.
Thus, the target piece symbols seen during training are the same as above.

### Check targets symbols for training

We briefly check the number of target piece symbols contained in the in-distribution samples.
These are a bit lower for the ``DIDACT`` training, because we removed the unintended samples for the uts-holdut.
Overall the numbers should not vary too much between ``DIDACT`` and ``NAIVE``.

```
python3 scripts/generate_annos_check.py \
    --didact_dir /data/pento_diaref/didact \
    --naive_dir /data/pento_diaref/naive
```

### Data format: Annotation

```
{'id': 1,
 'size': 7,
 'pieces': [['blue', 'F', 'top left', 90],
  ['olive green', 'F', 'top left', 180],
  ['blue', 'F', 'top right', 0],
  ['blue', 'F', 'top center', 0],
  ['blue', 'F', 'left center', 90],
  ['blue', 'F', 'right center', 270],
  ['blue', 'F', 'right center', 270]],
 'target': 1,
 'refs': [{'user': 'ia',
   'instr': 'Take the olive green piece',
   'type': 0,
   'sent_type': 1441,
   'props': {'color': 'olive green'}}],
 'global_id': 1,
 'split_name': 'data_train'
}
```

### Create the images for the `DIDACT` dataset

```
python3 scripts/generate_images_didactic.py \
    --data_dir /data/pento_diaref/didact \
    --image_size 224 224 \
    --split_name all \
    --force_reindex
```

Note: The ``-force_reindex`` option is necessary to align annotation ids with the positions of the images in the h5py
files. For example, the image for a validation annotation with id 120_000 might be located at the 1_000 position
in the h5py. So the annotation id becomes the position in the h5yp file and the global_id is the general identifier.

### Create the images for the `NAIVE` dataset

```
python3 scripts/generate_images_naive.py \
    --data_dir /data/pento_diaref/naive \
    --image_size 224 224 \
    --split_name all \
    --force_reindex
```

### Data format: Annotation with Bounding Boxes

...