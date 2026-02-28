echo "Deconding Starts."

uv run python ./cs336_basics/solutions/_decode_model.py \
    --run_name "default__lr_0.0005__bsz_128" \
    --input_text "Once upon a time"

echo "Completed!"