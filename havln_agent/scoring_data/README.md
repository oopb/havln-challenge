# Released HA-VLN validation collision references

The two `collision_num_val_*.json` files are copies of the **published**
validation annotations from
[UWMILab/HA-VLN](https://github.com/UWMILab/HA-VLN/tree/main/Data/HA-R2R-tools).

- `collision_num_val_seen.json`: 778 episode IDs
- `collision_num_val_unseen.json`: 1,839 episode IDs
- No held-out test annotations are included or reconstructed.

They contain the reference collision allowances used by the published HA-VLN
evaluation. Runtime annotations mounted under `<data-root>/HA-R2R-tools/`
take precedence; these bundled files are only a fallback for validation
scoring. They are never passed into the Codex navigation prompt, MCP tool
responses or policy observations.

Relay may use these annotations to reconstruct a **nonofficial, local,
closed-loop diagnostic Score** from real captured Habitat distance, success,
and collision measurements. Dynamic humans and server-side playback may differ;
only the official leaderboard has ranking authority.
