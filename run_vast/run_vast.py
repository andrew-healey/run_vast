#!/usr/bin/env python3
"""
This script parses a markdown file containing vast command blocks (triple-backtick codeblocks starting with ```vast or ```vast:state),
progresses them through a series of phases (verify, provision/run, check, finish),
and writes back updated state information to the file.
"""

import re
import argparse
import subprocess
import time
import sys
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import random

import logging
import coloredlogs
import webbrowser
from vast_api import VastAPIHelper
import pandas as pd

logger = logging.getLogger(__name__)
# Remove any existing handlers to avoid duplicate logging
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
logger.propagate = False  # Disable propagation to the root logger
coloredlogs.install(level="INFO", logger=logger)
logger.setLevel(logging.INFO)

# print(len(logger.handlers),"handlers in the logger")
# raise Exception("Stop here")

# Define an enum for the command block states.
class CommandState(Enum):
    EMPTY = "EMPTY"         # User-entered command block, not yet verified.
    VERIFIED = "verified"   # Command block has been verified.
    RUNNING = "running"     # Command block is running on an instance.
    FAIL = "fail"           # Command block failed.
    SUCCESS = "succeed"     # Command block succeeded.
    FINISHED = "finished"   # Command has finished and the instance has been deleted.

# Dataclass representing a command block.
@dataclass
class CommandBlock:
    index: int
    start: int                   # Start position in the file text.
    end: int                     # End position in the file text.
    content: str                 # Command content (inside the triple backticks)
    state: CommandState          # The parsed state.
    instance_id: Optional[str] = None  # Associated instance ID (if any)

# Parse the file to obtain all vast command blocks.
def parse_command_blocks(file_text: str) -> List[CommandBlock]:
    pattern = re.compile(r"```vast(?::(?P<tag>[\w/]+))?\n(?P<content>.*?)\n```", re.DOTALL)
    blocks = []
    for i, match in enumerate(pattern.finditer(file_text)):
        tag = match.group("tag")
        instance_id = None
        if tag is None:
            state = CommandState.EMPTY
        else:
            if "/" in tag:
                tag_parts = tag.split("/")
                if tag_parts[0] == "running":
                    state = CommandState.RUNNING
                    if len(tag_parts) > 1:
                        instance_id = tag_parts[1]
                elif tag_parts[0] == "fail":
                    state = CommandState.FAIL
                    if len(tag_parts) > 1:
                        instance_id = tag_parts[1]
                elif tag_parts[0] == "succeed":
                    state = CommandState.SUCCESS
                    if len(tag_parts) > 1:
                        instance_id = tag_parts[1]
                else:
                    # Unknown prefix; default to EMPTY.
                    state = CommandState.EMPTY
            else:
                if tag == "verified":
                    state = CommandState.VERIFIED
                elif tag == "finished":
                    state = CommandState.FINISHED
                else:
                    state = CommandState.EMPTY
        block = CommandBlock(
            index=i,
            start=match.start(),
            end=match.end(),
            content=match.group("content"),
            state=state,
            instance_id=instance_id,
        )
        blocks.append(block)
    return blocks

# Map a CommandBlock to its header tag string.
def get_tag_string(block: CommandBlock) -> str:
    if block.state == CommandState.EMPTY:
        return ""  # Becomes "```vast"
    elif block.state == CommandState.VERIFIED:
        return "verified"
    elif block.state == CommandState.RUNNING:
        return f"running/{block.instance_id}" if block.instance_id else "running"
    elif block.state == CommandState.FAIL:
        return f"fail/{block.instance_id}" if block.instance_id else "fail"
    elif block.state == CommandState.SUCCESS:
        return f"succeed/{block.instance_id}" if block.instance_id else "succeed"
    elif block.state == CommandState.FINISHED:
        return "finished"
    return ""

# A simple verification function. Besides "verifying" the command it asserts that
# the command (after stripping whitespace) does not contain a double-quote.
def verify_command_block(block: CommandBlock) -> bool:
    cmd = block.content.strip()
    # if '"' in cmd:
    #     print(f"Command {cmd} contains a double-quote. That's not allowed.")
    #     return False
    return True

def verify_phase(blocks: List[CommandBlock]) -> None:
    logger.info("=== Verification Phase ===")
    for block in blocks:
        if block.state == CommandState.EMPTY:
            if verify_command_block(block):
                logger.debug(f"Block {block.index} verified.")
                block.state = CommandState.VERIFIED
            else:
                logger.debug(f"Block {block.index} failed verification; leaving state as EMPTY.")
        else:
            logger.debug(f"Block {block.index} ({block.state}) is past the verification phase; skipping.")
    return

# Simulated instance class.
@dataclass
class Instance:
    instance_id: str
    state: str   # Could be "active", "loading", etc.
    actual_status: str = ""
    ssh_host: str = ""   # For SSH connection
    ssh_port: int = 22   # Default SSH port
    label: Optional[str] = None

def exponential_backoff_retry(func, initial_delay=5, max_delay=60, max_attempts=5):
    """
    Execute a function with exponential backoff retry logic.
    
    Args:
        func: The function to execute
        initial_delay: Initial delay between retries in seconds (default: 10)
        max_delay: Maximum delay between retries in seconds (default: 60)
        max_attempts: Maximum number of attempts before giving up (default: 5)
    
    Returns:
        The result of the function if successful
        
    Raises:
        Exception: If all retry attempts fail
    """
    delay = initial_delay
    attempt = 0
    
    while True:
        try:
            return func()
        except Exception as e:
            attempt += 1
            logger.error(f"Error (attempt {attempt}): {e}")
            if attempt >= max_attempts:
                logger.error("Max attempts reached. Giving up.")
                raise
            logger.info(f"Retrying in {delay} seconds...")
            time.sleep(delay)
            delay = min(delay * 2, max_delay)

# A function to get instances that are "for autorunning" using the vast-ai-api.
def get_autorunning_instances(gpus: Optional[int] = None) -> List[Instance]:
    def _fetch_instances():
        api = VastAPIHelper()
        return api.list_current_instances()
    
    launched_df = exponential_backoff_retry(_fetch_instances)

    instances = []
    # Filter for those instances which have an extra_env entry for IS_FOR_AUTORUNNING=true.
    for index, row in launched_df.iterrows():
        extra_env = row.get("extra_env", [])
        is_autorunning = any(
            (len(pair) == 2 and pair[0] == "IS_FOR_AUTORUNNING" and pair[1].lower() in ("true","1"))
            for pair in extra_env
        )
        if not is_autorunning:
            continue
        if gpus is not None and row['num_gpus'] != gpus:
            continue
        inst = Instance(
            instance_id=str(row["id"]),
            state=str(row.get("cur_state", "unknown")),
            actual_status=str(row.get("actual_status", "unknown")),
            ssh_host=row.get("public_ipaddr", ""),
            ssh_port=int(row['ports']['22/tcp'][0]['HostPort'] if type(row['ports']) == dict else 0),
            label=str(row.get("label", None)),
        )
        instances.append(inst)
    return instances

import random

# Launch a verified command block on a given instance.
def run_command_on_instance(block: CommandBlock, instance: Instance) -> None:
    def _set_label():
        api = VastAPIHelper()
        api.label_instance(instance.instance_id, None)
    
    exponential_backoff_retry(_set_label)

    # Extract the inner command (strip starting and ending quotes).
    command = block.content
    ssh_host = instance.ssh_host
    ssh_port = instance.ssh_port
    ssh_target = f"root@{ssh_host}"

    # Construct the full remote command you want to run.
    # Note: We're keeping the dollar-sign escapes (\\$CONTAINER_ID) as is.
    rand_id = random.randint(0, 1000000)
    remote_command = (
        f"true && echo rand_id={rand_id} && vastai label instance $CONTAINER_ID running && "
        f"{command} && "
        "vastai label instance $CONTAINER_ID succeed && "
        "vastai stop instance $CONTAINER_ID || "
        "vastai label instance $CONTAINER_ID fail"
    )
    
    # Use shlex.quote() to safely wrap the entire command
    import shlex
    quoted_remote_command = shlex.quote(remote_command)

    ssh_command = [
        "ssh",
        "-i", os.path.expanduser("~/.ssh/id_vast"),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-p", str(ssh_port),
        ssh_target,
        "tmux", "new-session", "-d", "-s", f"session_{random.randint(0, 1000000)}",
        "nohup","bash","-c",
        quoted_remote_command
    ]
    logger.info(f"Starting command on instance {instance.instance_id}: {' '.join(ssh_command)}")
    try:
        proc = subprocess.Popen(ssh_command)
        time.sleep(5)  # Let the tmux command start.
        proc.terminate()  # Allow remote command to continue.
        block.state = CommandState.RUNNING
        block.instance_id = instance.instance_id
    except Exception as e:
        logger.error(f"Error running block {block.index} on instance {instance.instance_id}: {e}")
        block.state = CommandState.FAIL
        block.instance_id = instance.instance_id

# For all verified commands with no instance yet, assign free instances from the provided pool.
def run_phase(blocks: List[CommandBlock], instances: List[Instance], gpus: Optional[int] = None) -> None:
    logger.info("=== Run Phase ===")
    # Identify instance IDs already claimed by blocks.
    claimed_ids = {block.instance_id for block in blocks
                   if block.instance_id and block.state in {CommandState.RUNNING, CommandState.FAIL, CommandState.SUCCESS}}
    free_started_instances = [inst for inst in instances if inst.instance_id not in claimed_ids and inst.actual_status == "running"]
    logger.info(f"{len(free_started_instances)} instances of {len(instances)} were free and started.")
    num_blocks_run = 0
    for block in blocks:
        if block.state == CommandState.VERIFIED and block.instance_id is None:
            if free_started_instances:
                instance = free_started_instances.pop(0)
                run_command_on_instance(block, instance)
                logger.debug(f"Assigned block {block.index} to instance {instance.instance_id}. Running command \"{block.content}\".")
                num_blocks_run += 1
            else:
                logger.info(f"No free instances available for block {block.index}. Skipping.")
    
    running_blocks = [block for block in blocks if block.state == CommandState.RUNNING]

    if len(running_blocks) == 0 or num_blocks_run == 0:
        return
    
    logger.info("Sleeping for 20 seconds to allow instances to label themselves as running.")
    time.sleep(20)

    updated_instances = get_autorunning_instances(gpus=gpus)
    logger.info(f"Found {len(updated_instances)} updated instances.")
    for block in running_blocks:
        matching_instances = [inst for inst in updated_instances if inst.instance_id == block.instance_id]
        if matching_instances:
            matching_instance = matching_instances.pop()
            logger.debug(f"Instance {block.instance_id} has label {matching_instance.label}.")
            if matching_instance.label in ("running","succeed","fail"):
                logger.debug(f"Verified that block {block.index} has at least started running on instance {block.instance_id}.")
            elif not matching_instance.label or str(matching_instance.label) == "None":
                logger.warning(f"Bad news - instance {block.instance_id} is not marked as running on vast - so our command block has probably not run at all.")
                block.state = CommandState.VERIFIED
                block.instance_id = None
        else:
            logger.warning(f"Bad news - instance {block.instance_id} not found in updated instances. Marking block {block.index} not running.")
            block.state = CommandState.VERIFIED
            block.instance_id = None
    logger.debug("Finished cross-checking 'running' blocks with vast instances.")

    return

# Check phase: In a real implementation, this would query instance labels (or logs) via vast-ai-api.
# Here we simply print a message.
def check_phase(blocks: List[CommandBlock], gpus: Optional[int] = None) -> None:
    logger.info("=== Check Phase ===")
    instances = get_autorunning_instances(gpus=gpus)
    logger.info(f"Found {len(instances)} instances in check phase.")
    
    for block in blocks:
        if block.state == CommandState.RUNNING and block.instance_id:
            logger.debug(f"Checking block {block.index} on instance {block.instance_id}.")
            matching_instances = [inst for inst in instances if inst.instance_id == block.instance_id]
            if matching_instances:
                details = matching_instances.pop()
                logger.debug(f"Instance {block.instance_id} found in instance details. details: {details}")
                label = details.label
                cur_state = details.state
                if label == "fail":
                    logger.debug(f"Block {block.index} on instance {block.instance_id} has failed.")
                    block.state = CommandState.FAIL
                elif label == "succeed":
                    logger.debug(f"Block {block.index} on instance {block.instance_id} has succeeded.")
                    block.state = CommandState.SUCCESS
                if label in ("fail", "succeed") and cur_state != "stopped":
                    logger.warning(f"Warning: Instance {block.instance_id} in block {block.index} with label {label} is not stopped (current state: {cur_state})")
            else:
                logger.warning(f"Instance details for {block.instance_id} not found.")
        else:
            logger.debug(f"Block {block.index} is not running, so I'm not checking it. (state: {block.state}, instance_id: {block.instance_id})")
    return

# Finish phase: convert SUCCESS or FAIL blocks to FINISHED and simulate deleting their instance.
def finish_phase(blocks: List[CommandBlock], delete_finished_instances: bool = True) -> None:
    logger.info("=== Finish Phase ===")
    api = VastAPIHelper()
    for block in blocks:
         if block.state == CommandState.SUCCESS and block.instance_id:
              logger.info(f"Finishing block {block.index} on instance {block.instance_id}.")
              if delete_finished_instances:
                  logger.debug(f"Deleting instance {block.instance_id} for block {block.index}.")
                  try:
                      exponential_backoff_retry(lambda: api.delete_instance(block.instance_id))
                      logger.debug(f"Deleted instance {block.instance_id} for block {block.index}.")
                  except Exception as e:
                      logger.error(f"Failed to delete instance {block.instance_id} after all retries: {e}")
                      continue
              else:
                  logger.debug(f"Not deleting instance {block.instance_id} as delete_finished_instances is False")
                  exponential_backoff_retry(lambda: api.label_instance(block.instance_id, "")) # wipe the label

              block.state = CommandState.FINISHED
              block.instance_id = None
         # Optionally, you might add similar deletion handling for FAIL blocks if desired.
    return

# Write the updated command blocks back to the file.
def writeback_file(fw, file_text: str, blocks: List[CommandBlock]) -> None:
    pattern = re.compile(r"```vast(?::(?P<tag>[\w/]+))?\n(?P<content>.*?)\n```", re.DOTALL)
    # Create an iterator over the blocks (they occur in order).
    block_iter = iter(blocks)
    def replacer(match):
        try:
            block = next(block_iter)
        except StopIteration:
            return match.group(0)
        new_tag = get_tag_string(block)
        if new_tag:
            header_line = f"```vast:{new_tag}"
        else:
            header_line = "```vast"
        return f"{header_line}\n{match.group('content')}\n```"
    
    new_text = pattern.sub(replacer, file_text)
    fw.seek(0)
    fw.truncate()
    fw.write(new_text)
    logger.debug(f"Wrote updated file back to {fw.name}.")
    fw.flush()
    os.fsync(fw.fileno())
    return

def provision_phase(blocks: List[CommandBlock], should_deprovision: bool = True, should_provision: bool = True, gpus: Optional[int] = None) -> List[Instance]:
    pending_cmds = [block for block in blocks if block.state in {CommandState.EMPTY, CommandState.VERIFIED, CommandState.RUNNING, CommandState.FAIL}]
    pending_cmd_count = len(pending_cmds)
    all_pending_cmds_are_running = all(block.state == CommandState.RUNNING for block in pending_cmds)
    slack = 2 if pending_cmd_count > 3 else 1
    while True:
        current_instances = get_autorunning_instances(gpus=gpus)
        instance_count = len(current_instances)
        if instance_count < pending_cmd_count and should_provision:
            needed = (pending_cmd_count) - instance_count
            logger.info(f"Provisioning Notice: Need {needed} (plus {slack} slack instances) more instance(s).")
            logger.info("Please provision the required instances manually Now. Attempting to open vast.ai in browser...")
            webbrowser.open("https://vast.ai/create/")
            input("Press Enter to continue...")
        else:
            break

    # if all pending commands are running, then we don't need any slack anymore!
    # so we can deprovision any extra instances.
    if all_pending_cmds_are_running:
        assigned_ids = {block.instance_id for block in blocks if block.instance_id}
        free_instances = [inst for inst in current_instances if inst.instance_id not in assigned_ids]
        logger.info(f"All pending commands are running, so we can deprovision {len(free_instances)}/ {len(current_instances)} extra/total instances.")
        if free_instances:
            api = VastAPIHelper()
            for inst in free_instances:
                if should_deprovision:
                    logger.debug(f"Deprovisioning extra instance {inst.instance_id}.")
                    try:
                        exponential_backoff_retry(lambda: api.delete_instance(inst.instance_id))
                        logger.debug(f"Successfully deprovisioned instance {inst.instance_id}.")
                    except Exception as e:
                        logger.error(f"Error deprovisioning instance {inst.instance_id} after all retries: {e}")
                else:
                    logger.debug(f"Setting label to empty for extra instance {inst.instance_id} (deprovision disabled).")
                    try:
                        exponential_backoff_retry(lambda: api.label_instance(inst.instance_id, ""))
                        exponential_backoff_retry(lambda: api.stop_instance(inst.instance_id))
                        logger.debug(f"Successfully set label and stopped instance {inst.instance_id}.")
                    except Exception as e:
                        logger.error(f"Error setting label or stopping instance {inst.instance_id} after all retries: {e}")
        else:
            logger.debug("No extra instances to deprovision.")
    return get_autorunning_instances(gpus=gpus)

def main():

    if "VAST_AI_API_KEY" not in os.environ:
        raise EnvironmentError("VAST_AI_API_KEY environment variable is not set. Please set it before running this script.")

    parser = argparse.ArgumentParser(description="Process vast command blocks from a markdown file.")
    parser.add_argument("filename", help="Markdown file containing vast command blocks")
    parser.add_argument("--deprovision", action="store_true", help="Deprovision extra instances", dest="deprovision")
    parser.set_defaults(deprovision=False)
    parser.add_argument("--provision", action="store_true", help="Do not provision extra instances, just roll with what you have", dest="provision")
    parser.set_defaults(provision=False)
    parser.add_argument("--gpus", type=int, help="Only operate on instances with this # of GPUs", dest="gpus")
    parser.set_defaults(gpus=None)
    parser.add_argument("--repeat", type=int, help="Repeat the script this many times", dest="repeat")
    parser.set_defaults(repeat=1)
    args = parser.parse_args()

    for i in range(args.repeat):
        try:
            with open(args.filename, "r") as f:
                file_text = f.read()
        except FileNotFoundError:
            logger.error(f"File {args.filename} not found.")
            sys.exit(1)
        
        with open(args.filename, "w") as fw:
            try:
                blocks = parse_command_blocks(file_text)
                logger.info(f"Found {len(blocks)} vast command block(s).")
                writeback_file(fw, file_text, blocks)

                # Phase 1: Verification
                verify_phase(blocks)
                writeback_file(fw, file_text, blocks)


                # Phase 2: Provision/Delete Phase
                instances = provision_phase(blocks, should_deprovision=args.deprovision, should_provision=args.provision, gpus=args.gpus)
                writeback_file(fw, file_text, blocks)


                # Phase 3: Run Phase
                run_phase(blocks, instances, gpus=args.gpus)
                writeback_file(fw, file_text, blocks)


                # Phase 4: Check Phase
                check_phase(blocks, gpus=args.gpus)
                writeback_file(fw, file_text, blocks)


                # Phase 5: Finish Phase (if needed)
                finish_phase(blocks, delete_finished_instances=args.deprovision)
                writeback_file(fw, file_text, blocks)

            finally:
                # Final write back.
                writeback_file(fw, file_text, blocks)

if __name__ == "__main__":
    main()