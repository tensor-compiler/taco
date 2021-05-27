# CUDA testing

The script, `test-pr.py`, is a quick and easy way to run CUDA tests on a PR.

This should be done after code review, see [issue #457](https://github.com/tensor-compiler/taco/issues/457) for a discussion of the overall process.

## System requirements

You will need write access to the `tensor-compiler/taco` github repo, and access to run actions.

You will need to have the command line `git` and `gh` tools installed, configured, and talking to github.

`git` needs to be set up with SSH authentication or HTTPS authentication to push to the `tensor-compiler/taco` github repo without a password prompt.  If you don't have that, please see [ssh setup in the github docs](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh) or [seting up an HTTPS token](https://docs.github.com/en/github/authenticating-to-github/keeping-your-account-and-data-secure/creating-a-personal-access-token).

`gh` is a command-line interface to the github REST API.  If you don't have `gh`, please see [their installation guide](https://github.com/cli/cli#installation).

`gh` needs to be set up to hit REST API endpoints without a password prompt.  If you don't have that, the [gh auth login](https://cli.github.com/manual/gh_auth_login) command should help.

The script requires python version 3.x, and a few basic python modules that should be installed by default (like subprocess and tempfile).

## What it does

This script does the following:

* creates a temporary test branch for a PR
* pushes that test branch to the taco github repo
* kicks off the cuda test (with the default parameters) to run on that test branch
* waits for the test to complete
* removes the temporary test branch

The script will give you a link to the test run on the github website, so you can watch and inspect the results.

## Using it

`python3 test-pr.md [--protocol=(ssh|https)] <PRNUMBER> [<TRYNUMBER>]`

`PRNUMBER` is the PR you want to test, without the `#` prefix.

If specified, `TRYNUMBER` becomes a suffix for the temporary test branch, so you can have multiple test branches going at once.  That can be omitted unless it is needed.

## what it looks like

This output comes from my own fork of taco as I was testing the script:

```
% python3 test-pr.py 3

=== looking up ID and params of test action

=== creating test branch test-pr3
remote:
remote: Create a pull request for 'test-pr3' on GitHub by visiting:
remote:      https://github.com/Infinoid/taco/pull/new/test-pr3
remote:

=== triggering test action
✓ Created workflow_dispatch event for cuda-test-manual.yml at test-pr3

To see runs for this workflow, try: gh run list --workflow=cuda-test-manual.yml
Test action is at: https://github.com/Infinoid/taco/actions/runs/882846018?check_suite_focus=true

=== waiting for action to complete

Refreshing run status every 3 seconds. Press Ctrl+C to quit.

X test-pr3 CUDA build and test (manual) · 882846018
Triggered via workflow_dispatch about 11 minutes ago

JOBS
X tests CUDA in 10m54s (ID 2686889157)
  ✓ Set up job
  ✓ Run actions/checkout@v2
  ✓ create_build
  ✓ cmake
  ✓ make
  X test
  ✓ Post Run actions/checkout@v2
  ✓ Complete job

ANNOTATIONS
X Process completed with exit code 2.
tests CUDA: .github#1


X Run CUDA build and test (manual) (882846018) completed with 'failure'
Test results are at: https://github.com/Infinoid/taco/actions/runs/882846018?check_suite_focus=true

=== cleaning up test branch test-pr3

=== cleaning up temp dir

```

# Troubleshooting

## no access to taco repo, no access to run actions in taco repo

If you are a taco developer, ask Fred for access.

## test workflow does not exist

If you see output that looks like this:

```
=== triggering test action
could not create workflow dispatch event: HTTP 422: Workflow does not have 'workflow_dispatch' trigger (https://api.github.com/repos/tensor-compiler/taco/actions/workflows/someIDnumber/dispatches)
```

This means the test branch does not have the cuda test workflow file.  In other words, the file `.github/workflows/cuda-test-manual.yml` does not exist yet in the version of taco that the PR is based on.

To fix this, rebase or merge the PR to the current taco master branch, and then rerun the test.

## other stuff

If you have some other problem with it, at me on github (@infinoid) or email me for help figuring it out.