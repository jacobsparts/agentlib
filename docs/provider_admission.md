# Host-wide provider admission

Provider admission limits aggregate local requests across processes that use
the same provider host and credential.

`concurrency` and `rpm` are read directly from the current model
configuration. They are never stored in SQLite.

SQLite contains runtime coordination facts only:

- a per-pool ticket counter;
- FIFO waiter and active-lease state;
- recent admission timestamps used for RPM enforcement.

Each caller obtains a monotonically increasing ticket. The lowest waiting
ticket may claim capacity when the active count is below the current
`concurrency` value and the rolling one-minute admission count is below the
current `rpm` value.

Blocked callers wait on a shared inotify notification file rather than polling
SQLite. Admission, release, cancellation, and stale-state cleanup broadcast a
notification so waiting callers immediately recheck their state. Wall-clock
lease and RPM deadlines are used as bounded wake-up times when no notification
is expected.

The provider HTTP request runs outside SQLite. Releasing a request marks its
active waiter finished and broadcasts availability.

No provider configuration, preallocated slots, claim window, pool record, or
configuration reconciliation is persisted.
