# Contributor guidelines

Thank you for considering contributing to `elicit`. It's people like you that allow us to improve our software.

`elicit` is an open source project and we love to receive contributions from our community!
There are many ways to contribute, from writing tutorials or guides, improving the documentation, submitting bug reports and feature requests or writing code.

If you want to contribute we suggest the following workflow:

1. Create your own [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) of the code 
2. Do the changes in your fork 
3. If you like the change and think the project could use it submit a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)

## Documentation related changes

If you are looking to improve or fix documentation, you need to be able to build the documentation locally to check that your changes work properly.

* Create an environment 

```
# Create an empty Python environment (here with conda example)
conda create -n elicitation-env python=3.11

# activate environment
conda activate elicitation-env
```

* Clone the <repository>
* Create a new branch

`git checkout -b <my-new-feature>`

* Make your changes
* build the documentation:

```
# clean the build folder
rm -r docs/build/
sphinx-build -M clean docs build

# build the documentation
sphinx-apidoc -o docs/source/api elicit --separate
sphinx-build docs/source docs/build/html
```

* Open ``build/html/index.html`` in a browser and review your changes. Make sure links works and the formatting looks correct.

## Code related changes

Will follow.

## Review process

We will assign reviewers once the Pull request is made, and they will review it as soon as time allows.

Check that all tests pass in the CI, if not a failing test may indicate something isn't right with your changes. 
You can look at the log output for each job. If you need help, ask a reviewer.

Once reviewers are satisified with your changes, they will only approve your submission if there are no conflicts with the main branch and all checks have passed in our CI workflow.

If you want to report a bug, suggest an improvement, or a feature request: Submit an issue using the issue templates
provided (will be shown to you when you open a new issue in GitHub).