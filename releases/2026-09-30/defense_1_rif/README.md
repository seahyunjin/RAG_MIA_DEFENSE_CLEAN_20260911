# Defense 1: RIF

Frozen main method:

`Retriever -> CLD/G4 -> old CF-Clip -> frozen generator`

Constants:

- `G4 = s1 - mean(s1,s2,s3,s4)`
- alarm iff `G4 > 0.0901547372341156`
- `L_i(v) = [log p_F(v) - log p_-i(v)]_+`
- `z'(v) = z_F(v) - [max_i L_i(v) - 9.565910959243753]_+`
- gold decoding engine: `SERIAL_KVCACHE`

The package does not contain an attack classifier, membership classifier,
training step, external judge, or attack-specific route. A CLD alarm removes
rank 1 from the supplied generator context without refill, matching the exact
frozen lineage. Retrieval scores used by G4 must contain at least four values.

## Integration

The application retrieves ranked documents and scores, then calls
`RIFDefense.generate`. `prompt_builder(documents)` must return token IDs for a
prompt rendered with exactly the supplied document list. RIF builds the FULL
and every leave-one-document-out branch and sends them to the frozen
SERIAL_KVCACHE decoder.

```python
from rif import RIFDefense

def prompt_builder(documents):
    ...  # Render the frozen system/query/context template and return token IDs.

defense = RIFDefense(model, tokenizer)
result = defense.generate(scores, documents, prompt_builder, max_new_tokens=64)
```

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```

