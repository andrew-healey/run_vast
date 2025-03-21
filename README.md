# run-vast

A command-line tool for running commands on vast.ai instances through markdown files.

## Installation

```bash
pip install run-vast
```

## Usage

1. Create a markdown file with vast command blocks:

```markdown
# My Vast Commands

```vast
echo "Hello from vast.ai!"
```

```vast:verified
python train.py
```
```

2. Run the commands:

```bash
run-vast your_commands.md
```

## Requirements

- Python 3.6 or higher
- vast-ai-api
- coloredlogs
- pandas
- VAST_AI_API_KEY environment variable set

## License

MIT License 