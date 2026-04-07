# Known Riboswitches — Validation Data

Ground-truth data for biological validation of the riboswitch inverse design algorithm.

## Sources

### Adenine Riboswitch (add A-riboswitch)
- **Rfam:** RF00167
- **PDB:** 1Y26 (ligand-bound aptamer domain)
- **Paper:** Mandal & Breaker (2004). Adenine riboswitches and gene activation by
  disruption of a transcription terminator. *Nature Structural & Molecular Biology*.

### TPP Riboswitch (Thiamine Pyrophosphate)
- **Rfam:** RF00059
- **PDB:** 2GDI
- **Paper:** Serganov et al. (2006). Structural basis for gene regulation by a
  thiamine pyrophosphate-sensing riboswitch. *Nature*.

## Data Format

Each `.json` file contains:
```json
{
  "name": "riboswitch name",
  "rfam_id": "RFxxxxx",
  "organism": "species name",
  "ligand": "ligand name",
  "sequence": "native RNA sequence",
  "on_structure": "dot-bracket for ligand-bound state",
  "off_structure": "dot-bracket for ligand-free state",
  "notes": "any caveats about the structures"
}
```

> **Important:** The structures listed here are approximate. Before using them for
> validation, verify against the actual Rfam entry and published crystal structure data.
