# Frozen corrective protocols

The JSON files in this directory are machine-enforced protocol contracts for
the v2 corrective experiments. Each file owns a complete canonical
`executable_config` plus an integer `protocol_version`; the Python module's
`DEFAULTS` is an implementation copy, not the source of truth. Production
launch loads the protocol artifact and rejects any extra key, missing key, type
change, list change, or value change. Changing a frozen value requires a new
protocol version and experiment name. CLI overrides are allowed only for quick
smoke runs, whose manifests are `smoke_only`.

Every manifest records the protocol path, the SHA-256 digest of the exact JSON
bytes, the protocol name/version, and the canonical executable-config digest.
Production provenance also compares every numerical source file's bytes,
regular-file/symlink type, and executable mode directly against the `HEAD`
tree. This check intentionally does not trust porcelain status: index flags
such as `assume-unchanged` and `skip-worktree` cannot conceal a modified source
file from the evidence gate.
