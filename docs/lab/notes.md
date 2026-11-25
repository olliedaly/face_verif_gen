# M3 lab notes

The experiments in `notebooks/02-lab.ipynb`, on the first 1,000 training people, scored on val.
Each has the expectation, written before the run; its plot; and three sentences on what happened
and why.

## L1. Learning rate

**Range test** (`L1-range.png`): one epoch from a fresh model, the rate rising geometrically from
1e-4 to 10. The lowest loss, 3.92, was at 0.28, so the usual rule of a tenth of that gives 0.028.
The loss falls fastest between about 0.01 and 0.1, by about 1.5 per tenfold rise, and steepest
around 0.06. Above 0.28 it is unstable, spiking at 0.4 and 0.6, and it blows up past 3. Picked
0.05, the middle of the steepest stretch.

**Runs:** 0.005, 0.05 and 0.5, a tenth of the pick, the pick and 10x it, at constant rates for 5
epochs each, as in the baseline. Checking the scheduler's rate moves to L2, which needs a scheduler
for its warmup.

**Expected:** 0.05 to achieve best loss. Potentially needs a slow warm up so 0.005 might start better. 0.5 looks unstable at 1 epoch, but may settle. The greater noise and SGD weight decay being relative to LR could mean 0.5 gets a better TAR.

**Plot:** `L1-runs.png`

**What happened and why:**
0.005: Worst loss, TAR appears to settle and is lowest, but TAR differences are small.
0.05: Best loss, best TAR, but setteles on TAR.
0.5: Close to 0.05 loss, slightly worse TAR, but trajectory is positive and may surpass 0.05 with more epochs.

## L2. Margin and warmup

**Runs:** ArcFace, s = 64 and m = 0.5, at L1's rate of 0.05 for 5 epochs. 4 runs, every combination of:

- Warmup: none, or the rate rising linearly from 1% of it over the first epoch (159 steps).
- Shift: AdaFace's final BatchNorm has no learnable shift, so in training each batch's embeddings
  average exactly zero. The other model gives that BatchNorm a learnable scale and shift, as
  i_use_arc_btw's neck had: in training, each batch's embeddings then average exactly the shift,
  an easy way for every face to move one way, as the collapse there did.

Each step logs the collapse's two signs:

- theta_y, the average angle between a face and its own identity's row. Above 75.7 degrees,
  pushing every face away from every identity lowers the loss.
- The angle between faces of different people, from their average cosine: 90 degrees when spread
  out, 0 when collapsed.

**Why 75.7 degrees:**
Own identity scored at s cos(theta + m), others at s cos(theta). m = 28.6 degrees.
Loss only depends on the gap: raise own score or lower the others.
Move a face away from every identity: all scores drop. Others drop at rate sin(theta), own at sin(theta + m).
Others drop faster when theta > (180 - 28.6) / 2 = 75.7. Gap grows, loss falls, nothing learnt about faces.
Random directions start ~90 degrees apart, so every run starts on the wrong side of 75.7.

**Expected:**

**Plot:** `L2.png`

**What happened and why:**

**Why a big first step falls in (a guess, untested):**
Step 0 is the same gradient in both runs. Small step: theta_y falls, so the gradient points to pull.
Big step: overshoots, and a big weight change moves all faces the same way.
Rows then move away from that shared direction together, every angle grows, and above 75.7 that lowers the loss, so it keeps going.
Momentum builds up the push, since it points the same way every batch. Pulls differ per identity, so they don't build up.
Warmup keeps early steps small until theta_y is below 75.7, where pushing no longer lowers the loss.

## L3. Optimisers

**Runs:** softmax, 5 epochs. 4 runs, every combination of:

- Optimiser: SGD with momentum 0.9 at L1's 0.05, or AdamW at the rate from its own range test
  (`L3-range.png`).
- Weight decay: on, at 1e-4 for SGD (AdaFace's) and 0.05 for AdamW (a common value), or off.

**Expected:** AdamW is not the best for CNNs, so SGD probably wins. Weight decay is cancelled on the Conv because of the batchnorm at the end. Weight decay shrinks weights by X, but batchnorm then normalises the output by the size of the weights.

**Plot:** `L3.png`

**What happened and why:**

## L4. Gradient checks

**Runs:** L2's ArcFace without the shift, at 0.05 for 5 epochs. 3 runs: warmup; no warmup; no
warmup with clipping. Clipping scales a step's gradient down to a cap whenever it is bigger. The cap
is the warmup run's median gradient size after warmup.

Each step logs, for each conv and linear layer, how far it moved the weights: the change's size
over the weights' size. Rule of thumb: about 1e-3 a step. Much more, the layer changes too fast to
settle; much less, it has stopped learning.

**Expected:**

**Plot:** `L4.png`

**What happened and why:**
Warmup and clipping: about the same. theta_y below 75.7 by ~step 100, TAR ~0.34 at epoch 5.
No warmup: collapses as in L2. Conv layers' steps fall to 1e-3 to 1e-4 by ~step 400, they stop learning. Only the head keeps moving.
Clipped run is clipped almost every step (cap 6.12), so it takes fixed-size steps the whole way.
Working runs: steady steps, 5e-3 to 1e-2. Step / weights = rate x gradient / weight size, all three steady.
Rule of thumb 1e-3 didn't hold: the failed run is the one near it.
Gradient size: 40 at step 0, flat at ~6 from ~step 300. Guess, untested: it's the average of 512 photo gradients. They agree at the start and add up, later they disagree and cancel.
Flat gradient is not flat loss: loss still falls, 28 to 13 (same run in L2), TAR rising.
Steady steps = stable training, not proof of learning. A good model takes smaller steps later: photos fitted, or rate cut.

## L5. Batch norm

**Train mode:** the baseline scored on val four ways: eval mode; train mode, val in its usual order
(each person's photos together); train mode, shuffled; eval mode again after one pass in train
mode.

**Batch size:** softmax, 2 epochs, 3 runs: 256 photos a step; 8 a step; 256 a step in 32 pieces of
8, their gradients added up before one step. Rate 0.05 x photos a step / 512: 0.025 at 256,
0.00078 at 8.

**Expected:** Dropout still on, so each photo only uses a random 60% of the features:
outputs are random and results worse. BatchNorm uses
the batch's mean and spread, not the running averages, so a photo's embedding depends on its batch.

**Plot:** `L5.png`

**What happened and why:**
Train mode, val TAR: eval mode: 0.204; train mode in order: 0.193; train mode shuffled: 0.218; eval mode after a pass in train mode: 0.230.
Dropout hurt less than expected: shuffled train mode, dropout on, still beat eval mode.
In order vs shuffled: same model, same mode, same photos. Only which photos share a batch changed, and TAR moved 0.025. A photo's embedding depends on its batch.
In order, a batch is ~14 people, so each batch is normalised by those few people's mean and spread, different for every batch. Shuffled, every batch is a mix close to all of val, so every batch is normalised about the same.
Eval after a pass in train mode is best, 0.230. Not a fair score: it uses val's photos to score val.
The pass ran val in order, 256 at a time: 38 batches, the last only 20 photos of 1 person. Each running average is updated as 0.9 x old + 0.1 x this batch's, so the last 10 batches (125 people) make up 65% of it, the first half of val 12%, and that 1 person alone 10%.
Its spread is the spread inside batches of ~14 people, which leaves out how much batches differ from each other.
So the running averages are not all of val's mean and spread: they lean on the last people in val's order.
Untested why it beats the training averages: val's people differ from the training people; the data order of the pass; or the training averages were out of date, collected while the weights still moved at rate 0.1.
Batch size, val TAR after 2 epochs: 256 a step: 0.190; 256 in pieces of 8: 0.192; 8 a step: 0.214.
256 vs pieces: same gradient, rate and number of steps. Only BatchNorm differs, 256 photos at once vs 8. Val the same, but pieces' training loss is higher throughout.
Training loss is measured in train mode. The mean and spread of 8 photos vary a lot between groups, so each photo gets a random shift from the 7 it's with. Noisier embeddings, head less sure, higher loss.
Val is in eval mode, which uses running averages over many photos, so that noise is gone.
So accumulation recovers the 256 result. 8 photos per BatchNorm was enough. Guess, untested: much smaller pieces, such as 2, would hurt.
Pieces vs 8 a step: same BatchNorm over 8, same rate per photo. Only timing differs. Pieces takes 32 gradients at the old weights, then one big update. 8 a step updates after every 8 photos, so each gradient uses the latest weights.
8 a step wins on val, though its loss sits above 256's for the same BatchNorm reason. Training loss is a poor guide when BatchNorm noise is in it.
Scaling the rate by photos a step assumes the gradient barely changes over 32 small steps. Guess, untested: it changes enough that following it step by step helps.
One seed each: rerun 8 and 256 a step with a second seed before trusting the 0.024 lead.
Time per epoch: 256 a step 114 s, pieces 140 s, 8 a step 168 s. 8 photos at a time doesn't fill the GPU, and 8 a step also does 32x more updates.

## L6. Precision

**NaN:** ArcFace's head with and without the acos clamp, on 512 faces placed exactly on their
identity's rows, so fp32 rounding puts some cosines just above 1. One pass each, plainly and under
anomaly detection.

**Precision:** softmax at 0.05 for 1 epoch, 3 runs: fp16; fp16 with loss scaling; bf16. Each step
logs the share of the model's gradient values that are exactly 0.

**Speed and memory:** 20 steps of 128 photos in fp32 and in bf16, one batch reused.

**Expected:**

**Plot:** `L6.png`

**What happened and why:**

## L7. Overfitting

**Runs:** softmax at 0.05 for 8 epochs on the first 200 training people, with 10% of each
person's photos held out. 5 runs: no regularisation; weight decay 1e-4 only; dropout 0.4 only;
flips only; all three. After each epoch: the eval-mode loss on the held out photos and on as many
photos it trains on, and val TAR.

**Expected:**

**Plot:** `L7.png`

**What happened and why:**

## L8. Leakage

**Exact copies:** a hash of every val and train photo's pixels. Counts of val photos with a copy
under the same val person, under another val person, and in train.

**Near copies:** every val and train photo embedded by the baseline. Each val photo's highest
cosine to any train photo, plotted next to val's genuine and impostor pairs.

**Expected:**

**Plot:** `L8.png`

**What happened and why:**

## L9. Data loading

**Runs:** softmax fed by a `DataLoader` over the 1,000 people, batches of 512: 0, 4 and 8
workers, each with pinned memory off and on. For each: photos a second from the loader alone and
when training (25 steps), the GPU's utilisation from `nvidia-smi`, and a profiler trace of 3 steps
in `outputs/lab/`.

**Expected:**

**Plot:** `L9.png`

**What happened and why:**

## L10. Memory

**Runs:** one step's peak memory at 64, 128, 256 and 512 photos, split into parameters,
gradients, optimiser state and activations, with and without checkpointing `model.body` in 4
segments. Then the largest batch that fits, each way.

**Expected:**

**Plot:** `L10.png`

**What happened and why:**

## L11. Alignment

**Runs:** the baseline on val, with every photo shifted 2, 4 or 8 pixels (up, down, left or right,
at random) or rotated 5 or 10 degrees (either way, at random). TAR at FAR 1e-4 and 1e-3.

**Expected:**

**Plot:** `L11.png`

**What happened and why:**
