#!/bin/bash
current_time=$(date +"%Y-%m-%d_%H_%M_%S")
git add _posts/
git commit -m $current_time
git push