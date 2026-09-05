# Extending the starter

This template leaves models, traffic, runtime choices, and hardware allocation to
the consuming project. There are no built-in short-chat, RAG, reasoning, or mixed
workload presets.

## Data boundaries

- **Topology:** pass a GPU matrix to `moeforge.api.parse_topology`. Supply explicit
  relative cost overrides for link labels absent from the default table.
- **Expert telemetry:** convert your observations into shared `ExpertStats` records.
  Analyze one layer at a time; include idle experts and avoid overlapping windows.
- **Placement constraints:** provide GPU IDs and any expert count limits, fixed
  assignments, expert sizes, and GPU memory budgets. Runtime memory overhead is your
  adapter's responsibility.
- **Routing origins:** optionally supply source-to-expert token counts for the same
  observation window. Expert residence is not a substitute for routing origin.
- **Benchmark evidence:** produce shared `BenchmarkResult` objects from your own
  request measurements. Use `None` when a supported measurement is unavailable.

The analysis package imports shared contracts rather than a runtime's internal types.
Keep that boundary when connecting a serving engine or an external telemetry source.

## Optional runtime adapter

`InferenceBackend`, `GenerationRequest`, and `GenerationResponse` define a starting
contract for a future adapter. Implement health checking and generation in your own
integration. Decide how your runtime exposes token counts and timestamps, and test
their semantics before using them in a benchmark. No backend is supplied here.

## Workload ownership

Your application should supply prompts, generation parameters, arrival scheduling,
warmup, and measurement windows. Keep these separate from topology and optimization.
The template deliberately does not prescribe distributions, rates, duration, or a
model. Never use the optimizer's synthetic unit-test fixture as benchmark evidence.

## Results and configuration

Record enough metadata to reproduce your own experiments: hardware, model revision,
runtime version, placement, traffic source, measurement settings, and seed where
applicable. Store real artifacts under a local `results/` directory or in an external
artifact store. Generated results and local credentials are ignored by Git.

## Scope

Greedy placement is deterministic but is not an exhaustive feasibility solver.
Link weights are relative costs, not measured bandwidth or latency. A supplied
benchmark's SLO result does not establish that a newly predicted placement meets SLO.
Read the optimizer module's README for these assumptions before integrating it.
