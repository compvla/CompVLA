# Convert raw episodes into LeRobot v3.0 datasets, one task at a time:
#   raw -> LeRobot v2.1 (datasets_convert/raw_to_lerobot.py)
#       -> LeRobot v3.0 (datasets_convert/v21_to_v30.py, converts in place and keeps the
#                        v2.1 copy as datasets/lerobot/{task}_old)
# Run from the repository root.
set -e

for TASK in Erase Pivot PnP Dispense
do
    python -m datasets_convert.raw_to_lerobot --task $TASK
    python datasets_convert/v21_to_v30.py \
        --repo-id=$TASK \
        --root=datasets/lerobot/$TASK \
        --push-to-hub=false
done
