#!/usr/bin/env python3

# This script runs the TACO CUDA tests on a pull request.
# Usage: python3 test-pr.py [--protocol=(https|ssh)] <PRnumber> [<Trynumber>]

# see test-pr.md for more details.

repo = "tensor-compiler/taco"
action_name = "CUDA build and test (manual)"

import os
import subprocess
import sys
import tempfile
import time

def main():
    args=list(sys.argv[1:])
    repo_url = "https://github.com/" + repo
    try:
        if len(args) > 0 and args[0].startswith('--protocol='):
            protocol = args[0][11:]
            args = args[1:]
            if protocol == 'ssh':
                repo_url = "ssh://git@github.com/" + repo
            elif protocol == 'https':
                repo_url = "https://github.com/" + repo
            else:
                raise Exception("unknown protocol " + protocol)
        pr=args[0]
        pr = int(pr)
        attempt = None
        if len(args) > 1:
            attempt = args[1]
            attempt = int(attempt)
    except:
        print("Usage: {} [--protocol=(ssh|https)] <prnumber> [attemptnumber]".format(sys.argv[0]))
        exit(1)

    print("\n=== looking up ID and params of test action")
    workflowid = find_workflow_id()

    with tempfile.TemporaryDirectory() as tmpdir:
        #print("tmpdir is", tmpdir)
        branchname = "test-pr{}".format(pr)
        if attempt is not None:
            branchname += "-try{}".format(attempt)

        print("\n=== creating test branch", branchname)
        subprocess.run(["git", "clone", "-q", repo_url, "git"], stdout=subprocess.DEVNULL, cwd=tmpdir, check=True)
        gitdir=os.path.join(tmpdir, "git")
        #print("gitdir is", gitdir)

        subprocess.run(["git", "fetch", "-q", "origin", "pull/{}/head:{}".format(pr, branchname)], stdout=subprocess.DEVNULL, cwd=gitdir, check=True)

        subprocess.run(["git", "checkout", "-q", branchname], stdout=subprocess.DEVNULL, cwd=gitdir, check=True)

        subprocess.run(["git", "push", "-q", "--set-upstream", "origin", branchname], stdout=subprocess.DEVNULL, cwd=gitdir, check=True)

        print("\n=== triggering test action")
        old_job_id = find_latest_workflow_run(workflowid)
        subprocess.run(["gh", "workflow", "run", "-R", repo, action_name, "-r", branchname], check=True)
        job_api_id = None
        while(job_api_id is None):
            # it takes a moment for new run requests to show up in the API.
            try:
                job_api_id, human_url = find_workflow_run(workflowid, branchname, later_than=old_job_id)
            except:
                time.sleep(1)

        print("\nTest action is at:", human_url)

        print("\n=== waiting for action to complete")
        time.sleep(10)
        #subprocess.run(["gh", "run", "watch", "-R", repo, str(job_api_id)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["gh", "run", "watch", "-R", repo, str(job_api_id)])

        print("\nTest results are at:", human_url)

        print("\n=== cleaning up test branch", branchname)
        subprocess.run(["git", "push", "-q", "origin", "--delete", branchname], stdout=subprocess.DEVNULL, cwd=gitdir, check=True)

        print("\n=== cleaning up temp dir")
        return


def find_workflow_id():
    result = subprocess.run(["gh", "workflow", "view", "-R", repo, action_name], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
    output = str(result.stdout, "utf-8")
    lines = output.split("\n")
    if lines[1].startswith("ID: "):
        workflowid = lines[1][4:]
        return int(workflowid)
    raise Exception("cannot find test workflow with name {}".format(action_name))

def find_latest_workflow_run(workflowid):
    output = subprocess.run(["gh", "run", "list", "-R", repo], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
    output = str(output.stdout, "utf-8")
    lines = output.split("\n")
    line = lines[0]
    try:
        status, result, title, workflow, ref, origin, elapsed, runid = line.split("\t")
        return runid
    except:
        pass
    return None

def find_workflow_run(workflowid, branch, later_than=None):
    output = subprocess.run(["gh", "run", "list", "-R", repo, "-w", str(workflowid)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
    output = str(output.stdout, "utf-8")
    lines = output.split("\n")
    for line in lines:
        try:
            status, result, title, workflow, ref, origin, elapsed, runid = line.split("\t")
            if later_than is not None and later_than >= runid:
                continue
            if ref == branch:
                return runid, "https://github.com/{}/actions/runs/{}?check_suite_focus=true".format(repo, runid)
        except:
            pass
    raise Exception("could not find workflow run for branch {}".format(branch))

main()