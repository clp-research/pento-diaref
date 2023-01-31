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

### Create the images for the `DIDACT` dataset

The generation process takes about 3 hours (more or less depending on the machine).

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

The generation process takes about 3 hours (more or less depending on the machine).

```
python3 scripts/generate_images_naive.py \
    --data_dir /data/pento_diaref/naive \
    --image_size 224 224 \
    --split_name all \
    --force_reindex
```

### Data format: Annotation with Bounding Boxes

```
{
  "id": 1,
  "size": 6,
  "pieces": [
    ["orange", "Y", "top right", 0], ["grey", "Y", "top right", 90], ["orange", "Y", "top left", 270],
    ["orange", "Y", "top left", 270], ["orange", "Y", "bottom right", 270], ["orange", "Y", "bottom right", 270]
  ],
  "target": 5,
  "refs": [
    {
      "user": "ia",
      "instr": "Take the orange piece in the bottom right",
      "type": 2,
      "sent_type": 907,
      "props": {
        "color": "orange",
        "rel_position": "bottom right"
      }
    }
  ],
  "bboxes": [
    [149, 179, 37, 52], [171, 186, 14, 44], [29, 44, 22, 52], [59, 74, 7, 37], [164, 179, 171, 201],
    [186, 201, 194, 224]
  ],
  "global_id": 123147,
  "split_name": "data_val"
}
```

## Training

The models will be saved to `saved_models` in the project folder.

### Train the Classifier+VSE model on `DIDACT`

The data mode `sequential_generation` is assumed (should not be changed)

```
python3 scripts/train_classifier_vse.py \
    --data_dir /data/pento_diaref/didact \
    --logdir /cache/tensorboard-logdir \
    --gpu 0 \
    --model_name classifier-vse-didact \
    --batch_size 24 \
    --d_model 512
```

### Train the Classifier+VSE model on `NAIVE`

The data mode `sequential_generation` is assumed (should not be changed)

```
python3 scripts/train_classifier_vse.py \
    --data_dir /data/pento_diaref/naive \
    --logdir /cache/tensorboard-logdir \
    --gpu 0 \
    --model_name classifier-vse-naive \
    --batch_size 24 \
    --d_model 512
```

### Train the Transformer+VSE model on `DIDACT`

We use the data mode `sequential_generation`

```
python3 scripts/train_transformer.py \
    --data_dir /data/pento_diaref/didact \
    --logdir /cache/tensorboard-logdir \
    --gpu 0 \
    --model_name transformer-vse-didact \
    --data_mode sequential_generation \
    --batch_size 24 \
    --d_model 512 \
    --dim_feedforward 1024 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --n_head 4 \
    --dropout 0.2
```

### Train the Transformer+VSE model on `NAIVE`

We use the data mode `sequential_generation`

```
python3 scripts/train_transformer.py \
    --data_dir /data/pento_diaref/naive \
    --logdir /cache/tensorboard-logdir \
    --gpu 0 \
    --model_name transformer-vse-naive \
    --data_mode sequential_generation \
    --batch_size 24 \
    --d_model 512 \
    --dim_feedforward 1024 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --n_head 4 \
    --dropout 0.2
```

### Train the Transformer model on `DIDACT`

We use the data mode `default_generation`

```
python3 scripts/train_transformer.py \
    --data_dir /data/pento_diaref/didact \
    --logdir /cache/tensorboard-logdir \
    --gpu 0 \
    --model_name transformer-didact \
    --data_mode default_generation \
    --batch_size 24 \
    --d_model 512 \
    --dim_feedforward 1024 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --n_head 4 \
    --dropout 0.2
```

### Train the Transformer model on `NAIVE`

We use the data mode `default_generation`

```
python3 scripts/train_transformer.py \
    --data_dir /data/pento_diaref/naive \
    --logdir /cache/tensorboard-logdir \
    --gpu 0 \
    --model_name transformer-naive \
    --data_mode default_generation \
    --batch_size 24 \
    --d_model 512 \
    --dim_feedforward 1024 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --n_head 4 \
    --dropout 0.2
```

### Train the LSTM model on `DIDACT`

The data mode `default_generation` is assumed (should not be changed)

```
python3 scripts/train_lstm.py \
    --data_dir /data/pento_diaref/didact \
    --logdir /cache/tensorboard-logdir \
    --gpu 0 \
    --model_name lstm-didact \
    --batch_size 24 \
    --lstm_hidden_size 1024 \
    --word_embedding_dim 512 \
    --dropout 0.5
```

### Train the LSTM model on `NAIVE`

The data mode `default_generation` is assumed (should not be changed)

```
python3 scripts/train_lstm.py \
    --data_dir /data/pento_diaref/naive \
    --logdir /cache/tensorboard-logdir \
    --gpu 0 \
    --model_name lstm-naive \
    --batch_size 24 \
    --lstm_hidden_size 1024 \
    --word_embedding_dim 512 \
    --dropout 0.5
```

## Evaluation

Choose the best model for each case and move them to the `saved_models` folder.
We used the model with the highest BLEU score and if equal, the most epochs trained on.

```
python3 scripts/evaluate_model.py \
    --data_dir /data/pento_diaref/didact \
    --results_dir results \
    --model_dir saved_models \
    --model_name <model_name> \
    --stage_name <stage_name> \
    --gpu 0 \
```

### Evaluate `NAIVE`-ly trained models

```
python3 scripts/evaluate_model.py \
    --data_dir /data/pento_diaref/didact \
    --model_name lstm-naive \
    --stage_name test \
    --gpu 0

python3 scripts/evaluate_model.py \
    --data_dir /data/pento_diaref/didact \
    --model_name transformer-naive \
    --stage_name test \
    --gpu 0

python3 scripts/evaluate_model.py \
    --data_dir /data/pento_diaref/didact \
    --model_name transformer-vse-naive \
    --stage_name test \
    --gpu 0

python3 scripts/evaluate_model.py \
    --data_dir /data/pento_diaref/didact \
    --model_name classifier-vse-naive \
    --stage_name test \
    --gpu 0
```

### Evaluate `DIDACT`-icly trained models

```
python3 scripts/evaluate_model.py \
    --data_dir /data/pento_diaref/didact \
    --model_name lstm-didact \
    --stage_name test \
    --gpu 0

python3 scripts/evaluate_model.py \
    --data_dir /data/pento_diaref/didact \
    --model_name transformer-didact \
    --stage_name test \
    --gpu 0

python3 scripts/evaluate_model.py \
    --data_dir /data/pento_diaref/didact \
    --model_name transformer-vse-didact \
    --stage_name test \
    --gpu 0

python3 scripts/evaluate_model.py \
    --data_dir /data/pento_diaref/didact \
    --model_name classifier-vse-didact \
    --stage_name test \
    --gpu 0
```

## Compute the results

```
python3 scripts/evaluate_results.py \
    --data_dir /data/pento_diaref/didact \
    --results_dir results \
    --stage_name test
```