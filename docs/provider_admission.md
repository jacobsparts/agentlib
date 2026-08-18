# Host-wide provider admission

Provider admission limits aggregate local requests across processes that use
the same provider host and credential.

`concurrency` and `rpm` are read directly from the current model
configuration. They are never stored in SQLite.

SQLite contains runtime coordination facts only:

- active leases, used to compare the current active count with `concurrency`;
- recent admission timestamps, used to compare the rolling one-minute count
  with `rpm`.

Each admission attempt performs one short transaction:

1. Remove expired or dead-process leases and timestamps older than one minute.
2. Count active leases for the derived provider pool.
3. Count recent admissions when `rpm` is configured.
4. If the current configuration permits the request, insert a lease and
   admission timestamp atomically.
5. Otherwise, close the transaction, wait briefly, and retry.

The provider HTTP request runs outside SQLite. Releasing the request deletes
its lease. No provider configuration, preallocated slots, ticket queue, pool
record, configuration reconciliation, or notification file is used.
