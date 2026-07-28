# Running experiments on a Google Cloud VM

Runbook for executing the BO experiments on a Compute Engine VM and getting the
results back. Workload is **CPU-bound** (many small GP fits in `float64`); a GPU
gives little-to-no benefit here, so we use a compute-optimized CPU VM.

## Config

The VM `savelev-c2` (`c2-standard-16` — 16 vCPU / 64 GB) already exists. Set
your project and zone once per shell:

```bash
export PROJECT=jetbrains-grazie
export ZONE=europe-west4-c
export VM=savelev-c2
```

Confirm it's reachable:

```bash
gcloud compute instances describe $VM --project=$PROJECT --zone=$ZONE \
  --format="value(name,status,machineType.basename())"
```

## 1. (Reference) How the VM was created

Already done — kept here for reproducibility:

```bash
gcloud compute instances create savelev-c2 \
  --project=$PROJECT --zone=$ZONE \
  --machine-type=c2-standard-16 \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=30GB
```

## 2. Get the code onto the VM (git)

The repo is on GitHub: `git@github.com:artemsvl/ParallelBO.git`.

> **Push first.** Local *working-tree* changes are not on the remote until
> committed and pushed — a fresh clone pulls stale code otherwise.

Locally:

```bash
git add -A && git commit -m "..." && git push
```

On the VM — the Debian 12 image is minimal, so install `git`/`tmux`/`curl`
first. The repo is private, so forward your local SSH key with `-A` (no
credentials are copied onto the VM):

```bash
ssh-add ~/.ssh/id_ed25519           # ensure your GitHub key is in the agent (once)

gcloud compute ssh $VM --project=$PROJECT --zone=$ZONE -- -A -t '
  sudo apt-get update -qq && sudo apt-get install -y -qq git tmux curl
  mkdir -p ~/.ssh && ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null
  git clone git@github.com:artemsvl/ParallelBO.git ~/ParallelBO
'
```

## 3. Set up the environment (uv)

```bash
gcloud compute ssh $VM --project=$PROJECT --zone=$ZONE --command='
  curl -LsSf https://astral.sh/uv/install.sh | sh
  cd ~/ParallelBO
  ~/.local/bin/uv sync
'
```

`uv` fetches Python 3.12 and all wheels from `uv.lock` (torch download is a few
hundred MB, one-time).

## 4. Launch the experiment

`run_experiments.py` runs all `(strategy x seed)` jobs in **parallel** across a
process pool (`-w/--workers`, default = all logical CPUs), one BLAS thread per
worker. On `c2-standard-16` use `-w 16` to saturate the box.

Run inside `tmux` so it survives disconnects:

```bash
gcloud compute ssh $VM --project=$PROJECT --zone=$ZONE
# --- now on the VM ---
tmux new -s exp
cd ~/ParallelBO
.venv/bin/python run_experiments.py --dim 8 --batch_size 50 --n_runs 15 -w 16 2>&1 | tee run.log
# detach (leaves it running):  Ctrl-b  then  d
```

> Use `.venv/bin/python` explicitly — robust against whatever venv/conda is
> active on the shell.

What you'll see:
- **`run.log` / tmux** = the parent's clean progress feed, one line per finished
  job: `[12/45] AsyncSimulation run=3 seed=45 -> best=-1.7421`.
- **Per-run verbose output** (per-iteration prints) goes to
  `data/<Strategy>/dim=8/q=50/<timestamp>/logs/run_NNN.log` — `tail -f` one to
  drill into a single run.

## 5. Monitor

```bash
# reattach the live tmux session
gcloud compute ssh $VM --project=$PROJECT --zone=$ZONE -- -t 'tmux attach -t exp'

# peek at aggregated progress without attaching
gcloud compute ssh $VM --project=$PROJECT --zone=$ZONE --command='tail -n 30 ~/ParallelBO/run.log'

# drill into one run's detailed output
gcloud compute ssh $VM --project=$PROJECT --zone=$ZONE --command='tail -n 20 ~/ParallelBO/data/*/dim=8/q=50/*/logs/run_003.log'

# still running? (shows the parent + all worker processes)
gcloud compute ssh $VM --project=$PROJECT --zone=$ZONE --command='pgrep -fal run_experiments | head'
```

## Stop & clean a run

Stop the running process (kills the parent and all pool workers — they share
the foreground process group):

```bash
# from your laptop
gcloud compute ssh $VM --project=$PROJECT --zone=$ZONE --command='pkill -f run_experiments.py'
gcloud compute ssh $VM --project=$PROJECT --zone=$ZONE --command='pgrep -fal run_experiments || echo "stopped"'
# OR: attach the tmux session and press Ctrl-C
gcloud compute ssh $VM --project=$PROJECT --zone=$ZONE -- -t 'tmux attach -t exp'
```

Clean partial results + log (untracked, safe to delete), then optionally drop
the tmux session:

```bash
gcloud compute ssh $VM --project=$PROJECT --zone=$ZONE --command='cd ~/ParallelBO && rm -rf data run.log'
gcloud compute ssh $VM --project=$PROJECT --zone=$ZONE --command='tmux kill-session -t exp 2>/dev/null; true'
```

> Switching git branches does **not** touch `data/` (it's untracked), so always
> clean it explicitly when you want a fresh run.

## 6. Download results

The script writes everything under `data/`:

```bash
cd /Users/Artem.Savelev/PycharmProjects/ParallelBO
gcloud compute scp --recurse --project=$PROJECT --zone=$ZONE \
  $VM:~/ParallelBO/data ./data_from_vm
gcloud compute scp --project=$PROJECT --zone=$ZONE $VM:~/ParallelBO/run.log ./run_vm.log
```

## 7. Tear down (avoid idle billing)

```bash
# delete entirely (recommended for a one-off)
gcloud compute instances delete $VM --project=$PROJECT --zone=$ZONE --quiet

# OR stop to resume later (disk still bills, compute does not)
gcloud compute instances stop $VM --project=$PROJECT --zone=$ZONE
```

## Updating the VM later / switching branches

No re-cloning — push locally, fetch/checkout on the VM. The parallel runner
currently lives on `feature/parallel-runs`:

```bash
# local
git push -u origin feature/parallel-runs
# VM: switch branch + re-sync deps. NOTE the `-- -A`: the repo is private, so
# the SSH agent MUST be forwarded for git fetch/pull to authenticate.
gcloud compute ssh $VM --project=$PROJECT --zone=$ZONE -- -A -t \
  'cd ~/ParallelBO && git fetch && git checkout feature/parallel-runs && git pull && ~/.local/bin/uv sync'
```

> Any VM command that hits GitHub (`git fetch`/`pull`/`clone`) needs `-- -A` to
> forward your local key. `--command=` without `-A` fails with
> `Permission denied (publickey)`. Local-only commands (tail, pgrep) don't.

## Notes

- **Use `-w` to actually spend the 16 vCPUs.** The sweep's `(strategy x seed)`
  jobs are independent, so `run_experiments.py` distributes them over a process
  pool and pins one BLAS thread per worker. Without `-w`, extra cores would only
  speed up each GP fit's BLAS — which is near-useless at these matrix sizes —
  rather than the sweep as a whole.
- **GPU is not worth it here:** `dtype = torch.double` is crippled on most GPUs,
  the matrices (N ≤ ~1000, d=2) are too small to saturate one, and the async /
  single-point strategies are latency-bound sequential GP fits. Reconsider only
  for high dimensions, much larger N, or large-q MC acquisition — and use an
  A100-class card if so.
- For results that survive VM deletion, push `data/` to a GCS bucket from the VM
  (`gsutil cp -r data gs://<bucket>/`) instead of relying on `scp`.
