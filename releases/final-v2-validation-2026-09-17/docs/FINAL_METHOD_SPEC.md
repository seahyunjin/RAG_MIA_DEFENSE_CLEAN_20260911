# Final V2 / G4 / LRC-4 method

For Top-4 normalized retrieval similarities \(s_1\ge s_2\ge s_3\ge s_4\),

\[G_4(q)=s_1-rac{s_1+s_2+s_3+s_4}{4}.\]

The alarm rule is strict `G4 > tau`, where `tau` is estimated only from benign calibration queries at the precommitted 2.5% operating budget. On alarm, the current-turn rank-1 source is removed, remaining context is deterministically backfilled, and the frozen generator runs once. No attack label, classifier, family-specific threshold, session state, or attack-specific route is used.
