# Governance

Speculators is part of the [vLLM project](https://github.com/vllm-project). Speculators maintainers guide the project's technical direction, review and merge contributions, and help triage issues and feature requests.

## Code ownership

The repository's [CODEOWNERS file](https://github.com/vllm-project/speculators/blob/main/.github/CODEOWNERS) lists the people GitHub requests to review changes to specific files and directories. It covers model implementations, checkpoint conversion, data generation, training, tests, and CI. Check the entries for the paths your pull request changes to find the relevant reviewers.

The owners listed for `src/speculators/` are:

- [@fynnsu](https://github.com/fynnsu) — training, data generation, checkpoint conversion, and model implementations.
- [@rahul-tuli](https://github.com/rahul-tuli) — data generation, checkpoint conversion, and MTP models.
- [@orestis-z](https://github.com/orestis-z) — P-Eagle, Eagle3, and DFlash models.
- [@shanjiaz](https://github.com/shanjiaz) — training, data generation, checkpoint conversion, and model implementations.

The file also lists reviewers for CI, tests, and repository configuration.

## Getting involved

You can contribute by reviewing pull requests, helping triage issues, improving documentation, or submitting code. For significant changes, open or find a GitHub issue and discuss the approach with maintainers before starting implementation. The [Contributing Guide](contributing.md) explains when to coordinate and how to get started.
