# Host-wide provider admission

## Purpose

Agentlib must bound aggregate provider concurrency across unrelated local
processes. Local admission completes before the provider HTTP call begins, so
the provider request timeout does not include time in the local queue.

## Control-plane files

The implementation uses exactly two hidden files by default:

- `~/.agentlib/.provider_admission.sqlite3`
- `~/.agentlib/.provider_admission.notify`

Both are mode 0600. `AGENTLIB_ADMISSION_DB` and
`AGENTLIB_ADMISSION_NOTIFY` can override their paths for tests.

SQLite is authoritative. The notification file is only a wake-all hint.

## Pool identity and configuration

Normalized tables are keyed by a quota-pool key rather than creating dynamic
tables. By default the key is the provider name plus a SHA-256 fingerprint of
the provider host and credential. Secrets are never stored. `admission_pool`
may explicitly identify a shared quota.

Provider admission is enabled if and only if `concurrency` is configured for
the provider/model. Its value is the strict host-wide active-request capacity;
without it, Agentlib does not create or use an admission pool.

When `rpm` is configured, the same admission transaction also enforces a
host-wide request-rate allowance. The allowance replenishes at `rpm / 60`
requests per second. `admission_burst` controls how many requests may accumulate
for a burst and defaults to `concurrency`. Completing a request does not add
capacity back to the rate allowance.

Optional settings:

- `admission_pool`
- `admission_burst` (default `concurrency`)
- `admission_claim_window` (default 0.1 seconds)
- `admission_lease_grace` (default 30 seconds)
- `admission_queue_timeout` (default unlimited)

Callers that configure different concurrency, RPM, or burst values for the
same pool fail clearly.
Whenever a `ProviderAdmission` instance initializes, it deletes all terminal
waiter rows (`finished`, `expired`, and `cancelled`) across every pool. Waiting
and active rows are never pruned.

## Schema

```sql
CREATE TABLE admission_pools (
    pool_key TEXT PRIMARY KEY,
    capacity INTEGER NOT NULL,
    next_ticket INTEGER NOT NULL,
    rate_per_minute REAL,
    burst_capacity REAL,
    available_tokens REAL,
    tokens_updated_at REAL
);

CREATE TABLE admission_slots (
    pool_key TEXT NOT NULL,
    slot_no INTEGER NOT NULL,
    lease_id TEXT,
    waiter_id TEXT,
    ticket INTEGER,
    owner_pid INTEGER,
    acquired_at REAL,
    expires_at REAL,
    PRIMARY KEY (pool_key, slot_no)
);

CREATE TABLE admission_waiters (
    pool_key TEXT NOT NULL,
    ticket INTEGER NOT NULL,
    waiter_id TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL,
    queued_at REAL NOT NULL,
    acquired_at REAL,
    head_claim_expires_at REAL,
    slot_no INTEGER,
    lease_id TEXT,
    finished_at REAL,
    PRIMARY KEY (pool_key, ticket)
);
```

## FIFO protocol

1. A caller installs an inotify watch before checking admission state.
2. A short `BEGIN IMMEDIATE` transaction allocates a monotonic ticket.
3. SQLite replenishes the request-rate allowance from elapsed wall time.
4. If a slot and request allowance are both available, the oldest waiting
   ticket receives a short claim opportunity.
5. Only that ticket may atomically consume one request allowance unit, become
   active, and occupy a free slot in the same `BEGIN IMMEDIATE` transaction.
6. If it does not claim before the opportunity expires, it is marked expired.
   If still alive, it obtains a new ticket at the back.
7. The provider request begins only after the combined claim commits.
8. Release conditionally clears the matching active slot in `finally`; it does
   not replenish the consumed request-rate allowance.

No SQLite transaction remains open during provider I/O. Kernel scheduling
cannot reorder calls because eligibility and claim are checked atomically.

## Wake-up protocol

Every waiter owns an inotify instance watching the one global notification
file. After a committed release, claim, cancellation, expiration, or capacity
change, the process overwrites one byte in that file. All current listeners
wake and recheck SQLite.

The watch is installed before checking SQLite, preventing the check/sleep
lost-wakeup race. Duplicate, coalesced, and spurious events are harmless.

A waiter blocks until the earliest of:

- an inotify event;
- the current head-claim deadline;
- the earliest active lease expiration;
- the exact time at which the request-rate allowance next permits a call;
- its absolute queue deadline.

These are scheduled failure-recovery wakes, not periodic polling.

## Failure recovery

A dead waiting head loses its opportunity at the claim deadline. An active
lease expires after the request timeout plus grace; another caller can reclaim
it. Conditional lease identity prevents stale release.

SQLite commit and notification cannot be atomic. If a process dies after
commit but before notification, waiters recover at the previously observed
head or active-lease deadline.

## LLMClient integration

Every direct provider `_call()` is wrapped as:

```text
enqueue -> acquire lease -> start HTTP call/timeout -> release in finally
```

Retry backoff happens after release, and each retry receives a new FIFO ticket.
Controller and BrowserAgent clients share capacity when their provider and
credential identify the same quota pool.

## Verification requirements

Tests cover cross-process capacity and RPM, atomic rate/slot claims, FIFO
ordering, exact rate-deadline wake-up, wake-all behavior, queue timeout,
dead-head advancement, dead-active lease reclamation, conditional release,
pool isolation, configuration mismatch, exception cleanup, retries, and proof
that provider work starts only after admission.
