# Security policy

## Reporting

This is a small audit toolkit; we don't expect security-sensitive surface
area, but we take any report seriously.

If you find a vulnerability — for example, a way to make `convert()` or
`run_audit.py` execute arbitrary code from a malformed input file — please
**open an issue** on GitHub. Choose the "API behaviour issue" template
and label it `security`.

If you'd prefer not to disclose publicly first, please reach out via
the contact in `CITATION.cff` and we'll coordinate disclosure.

## Threat model

This toolkit:

- Reads JSON / JSONL files supplied by the user.
- Computes statistics on those files.
- Optionally writes results to disk.

It does **not**:

- Execute generated code from untrusted sources.
- Make network requests at import time.
- Load model weights or run inference.
- Require root / elevated permissions.

The most realistic concern is malformed input causing an unbounded
parse / memory blowup. We treat such bugs as ordinary correctness
issues, fixed under the regular PR cycle.

## Versions

The toolkit is pre-1.0; we do not currently issue security updates for
older tags. The recommended fix for any reported issue is to update to
`main`.
