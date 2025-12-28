# Sandbox

Filesystem isolation for agent execution using Linux overlay filesystems and user namespaces.

## What It Does

- Agent code runs in an isolated mount namespace
- Filesystem writes go to a temporary overlay layer, not the real filesystem
- When the session ends, all changes are captured as a tarball
- Changes can be reviewed, applied, or discarded

## Requirements

**Linux only.** The sandbox uses kernel features not available on macOS or Windows.

### User Namespaces

Unprivileged user namespaces must be enabled:

```bash
# Check current setting
cat /proc/sys/kernel/unprivileged_userns_clone

# Enable if needed (requires root)
sudo sysctl kernel.unprivileged_userns_clone=1

# Make persistent
echo 'kernel.unprivileged_userns_clone=1' | sudo tee /etc/sysctl.d/99-userns.conf
```

### GCC

The sandbox helper is compiled from C on first use:

```bash
# Debian/Ubuntu
sudo apt install build-essential

# Fedora/RHEL
sudo dnf install gcc
```

### Overlay Filesystem

Standard on modern Linux kernels (3.18+). No action needed.

## Usage

```python
from agentlib import SandboxMixin, CodeAgent

class SandboxedCodeAgent(SandboxMixin, CodeAgent):
    pass

with SandboxedCodeAgent() as agent:
    result = agent.run("Create a file called ~/test.txt")

# Real filesystem is unchanged
# Review what was modified:
for path, content in agent.get_changed_files().items():
    print(f"{path}: {content[:50]!r}")

# See deleted files (overlay whiteouts):
print(agent.get_deleted_files())

# Apply changes to real filesystem if approved:
agent.apply_changes()

# Or discard (no-op, changes were never applied):
agent.discard_changes()
```

### CLI Integration

When used with `CLIMixin`, the sandbox automatically prompts on exit:

```python
from agentlib import SandboxMixin, CodeAgent
from agentlib.cli import CLIMixin

class SandboxedCLIAgent(SandboxMixin, CLIMixin, CodeAgent):
    pass

# On Ctrl+D to quit, user sees:
# ============================================================
# SANDBOX CHANGES
# ============================================================
#
# Modified/Created: 2 file(s)
#   test.txt (13 bytes)
#   config.json (256 bytes)
#
# What would you like to do?
#   [r] Review diff
#   [a] Apply changes to filesystem
#   [s] Save as patch file
#   [d] Discard (default)
```

The exit prompt:
- Shows summary of all filesystem changes
- `[r]` displays unified diff of all changes
- `[a]` applies changes to real filesystem (with confirmation)
- `[s]` saves tarball to a file for later review
- `[d]` discards all changes (default if Enter pressed)

### Configuration

```python
class SandboxedAgent(SandboxMixin, CodeAgent):
    sandbox_target = "/home/user"  # Directory to overlay (default: $HOME)
```

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│ Main Process                                                │
│                                                             │
│  1. Create temp dirs: /tmp/sandbox.XXX/{upper,work}         │
│  2. Launch sandbox_helper with Python subprocess            │
│  3. Communicate via sockets                                 │
│  4. On close: read tarball of changes, cleanup              │
│                                                             │
│         ┌─────────────────────────────────────────────┐     │
│         │ Sandboxed Worker (child process)            │     │
│         │                                             │     │
│         │  1. unshare(CLONE_NEWUSER | CLONE_NEWNS)    │     │
│         │  2. Write uid_map/gid_map (become "root")   │     │
│         │  3. Mount overlay on target directory       │     │
│         │  4. Execute Python REPL                     │     │
│         │                                             │     │
│         │  Overlay:                                   │     │
│         │    lower = /home/user (read-only)           │     │
│         │    upper = /tmp/sandbox.XXX/upper (writes)  │     │
│         │    merged = /home/user (what process sees)  │     │
│         └─────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Tool Injection

Tools defined on the agent (`read`, `edit`, `bash`, `glob`, `grep`, `web_fetch`, etc.) are extracted via `inspect.getsource()` and injected into the sandboxed REPL as standalone functions. This means:

- Tool code runs inside the sandbox
- File operations are isolated
- Subprocess calls (e.g., `bash()`) inherit the mount namespace

## Tarball Contents

The tarball captures overlay changes:

- **New files**: Files created by the agent
- **Modified files**: Copy-on-write copies with changes
- **Deleted files**: Overlay whiteouts (character devices 0,0)

```python
import tarfile, io

tarball = agent.get_tarball()
with tarfile.open(fileobj=io.BytesIO(tarball)) as tf:
    for member in tf.getmembers():
        if member.isfile():
            print(f"Changed: {member.name}")
        elif member.ischr() and member.devmajor == 0:
            print(f"Deleted: {member.name}")
```

## Limitations

- **Linux only** - Uses user namespaces and overlayfs
- **Single directory** - Only one directory tree is overlaid (default: home)
- **Network not isolated** - Only filesystem is sandboxed
- **Requires gcc** - Binary compiled on first use
- **Allowed paths** - Target must be under `/home/` or `/tmp/`

## Troubleshooting

### "unshare failed: Operation not permitted"

User namespaces are disabled. Enable with:
```bash
sudo sysctl kernel.unprivileged_userns_clone=1
```

### "gcc not found"

Install build tools:
```bash
sudo apt install build-essential  # Debian/Ubuntu
sudo dnf install gcc              # Fedora/RHEL
```

### "overlay mount failed"

- Check kernel supports overlayfs: `cat /proc/filesystems | grep overlay`
- Ensure target directory exists and is readable
- Target must be under `/home/` or `/tmp/`
