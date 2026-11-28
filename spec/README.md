# Guarantee records

What SRD and the libfabric efa provider guarantee, and what NCCL's RMA plugin interface guarantees, written as one record per guarantee so that tools can check them and other documents can refer to them by identifier. [`docs/specification.md`](../docs/specification.md) is generated from these files.

| File | Contents |
|---|---|
| `sources.toml` | The documents and source files the records quote, each pinned to one commit |
| `efa.toml` | The efa provider and the device under it |
| `nccl.toml` | NCCL's RMA plugin interface |

## Fields

| Field | Meaning |
|---|---|
| `id` | `AREA-TOPIC-NNN`. Stable: a record that stops being true is changed, not renumbered |
| `title`, `statement` | The guarantee in one line, then in one sentence precise enough to check |
| `kind` | `forbids`: the fabric never does this, so this project must not either. A capability the fabric always offers is written as a forbidding ("does not refuse ..."). `permits`: the fabric may do this, so this project should be able to |
| `scope` | `fabric`, `operations`, and `when` for the options or conditions the guarantee depends on |
| `layer` | Which part of the real stack keeps the guarantee: `device` (the EFA device and SRD, reached through verbs and efadv), `efa-provider` (libfabric's efa provider, in software), `libfabric-core` (libfabric's common layer) or `nccl`. A fake needs to reproduce only what the layers below it keep; anything above that it reproduces is a place it can drift from the real stack |
| `basis` | `documented`, `read-from-source`, `observed-on-hardware` or `assumed`. An assumed guarantee says in `basis_note` what it rests on |
| `sources` | Each `{ source, quote }`: the quote appears in that source at its pinned commit |
| `checked_by` | `test:<ctest name>`, `model:<property>` or `fact:<hardware fact>`. Empty only with `unchecked` saying why |
| `srdsim` | `status` and `note`: how this project compares. `conforms`; `hides-bugs` when code that is wrong on the hardware can pass here; `invents-bugs` when code that is right on the hardware can fail here; `not-implemented`; `unverified` |

## Checking

```bash
python3 tools/spec.py check
python3 tools/spec.py render --check
```

`check` validates every record, resolves every `test:` reference against the tests CMake registers, and fetches each source to confirm each quote is still in it. Moving a pin in `sources.toml` therefore reports every guarantee whose text changed at the new commit. `render --check` fails when `docs/specification.md` is not what the records produce.
