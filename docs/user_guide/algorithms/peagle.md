















- Depth 0 retains all n positions
- Depth d retains approximately n × r^d positions, where r is the `down-sample-ratio`
- A minimum retention floor (`down-sample-ratio-min`) prevents under-sampling at deep levels

This geometric decay means deeper predictions train on fewer positions per batch, keeping memory usage manageable while still learning to predict multiple tokens ahead.
