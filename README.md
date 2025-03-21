# `run_vast`

A command-line tool. Lets you put commands in markdown files, and runs them in parallel on many vast.ai instances.

Uses a `waiting`/`running`/`fail`/`succeed` state machine to represent every command. All state is contained in the markdown file, in human-readable and human-editable form.

I like to provision 10-20 Vast instances, usually with 4x4090s each, at the beginning of the day, with the same custom Dockerfile.

Vast lets me keep these nodes idle for very cheap. So by default, all instances are idle.

Then later, I will add a new ML experiment to my `journal.md` file. Every training run in the experiment is a bash command in a triple-backtick ```` ```vast```` code block.

Then I run `run_vast journal.md` to run them all in parallel. Each Vast instance will go idle when its command succeeds.

If a run fails, the code block will be marked as ```` ```vast:fail/012345````, where `012345` is the instance ID of the machine it ran on. I can then ssh into the instance and debug my training run.

If a run starts up successfully, the code block will be marked as ```` ```vast:running/012345````.

## Installation

```bash
pip install run-vast
```

## Usage

#### Make a list of commands you want to run

You should put these in a markdown file. Each command gets its own triple-backtick code block, annotated with `vast`.

For example, to train nanogpt with two different lrs:

````markdown
# Train nanogpt with different lrs

lr=0.5 and lr=1.5:

```vast
git clone https://github.com/karpathy/nanogpt && \
cd nanogpt && \
pip install torch numpy transformers datasets tiktoken wandb tqdm && \
python data/shakespeare_char/prepare.py &&
python train.py config/train_shakespeare_char.py --min_lr=0.5e-4
```

```vast
git clone https://github.com/karpathy/nanogpt && \
cd nanogpt && \
pip install torch numpy transformers datasets tiktoken wandb tqdm && \
python data/shakespeare_char/prepare.py &&
python train.py config/train_shakespeare_char.py --min_lr=1.5e-4
```
````

#### Set up your Vast account.

You need to make an SSH key to connect to Vast instances.

Register your SSH key on the vast website, then put the private key in `~/.ssh/id_vast`.

#### Run `run_vast my_training_runs.md`.

`run_vast` will prompt you to provision two Vast instances, so it can run both commands in parallel.

Important: in the vast.ai web UI, before provisioning Vast instances, you must edit the instance template to set the environment variable `IS_FOR_AUTORUNNING=1`.

Remember to press the "+" button to save the environment variable.

#### Go to the Vast dashboard and wait for your instances to be "Connected".

This should take a minute or so.

Then, return to the `run_vast` prompt and press Enter to continue.

#### Wait for your commands to finish

You should track your runs via i.e. wandb. `run_vast` doesn't handle any logging for you.

Once your commands have finished, run `run_vast journal.md`.

It will move them from the `vast:running/0123456` state to the `vast:finished` state.

## Requirements

- Python 3.6 or higher
- vast-ai-api
- coloredlogs
- pandas
- VAST_AI_API_KEY environment variable set

## License

MIT License 
