#!/bin/bash

get_branch() {
    git symbolic-ref --short HEAD 2>/dev/null || echo "main"
}

git add .
if ! git diff-index --quiet HEAD; then
    MAIN_BRANCH=$(get_branch)
    COMMIT_MSG="${1:-chore: sync main repo}"
    git commit -m "$COMMIT_MSG"

    if git pull --rebase origin "$MAIN_BRANCH"; then
        git push origin "$MAIN_BRANCH"
    else
        git rebase --abort 2>/dev/null
        echo "Failed to sync main repo. Manual intervention may be required."
        exit 1
    fi
else
    echo "No changes in main repo"
fi

echo "All synced."
