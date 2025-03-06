for i in \
"cells.csv       CELL_TYPE           summary-cell.json" \
"methods.csv     LIBRARY_METHODOLOGY summary-method.json" \
"phenotypes.csv  PHENOTYPE_NOTES     summary-phenotype.json" \
"genes_human.csv OFFICIAL_SYMBOL     summary-gene-human.json" \
"genes_mouse.csv OFFICIAL_SYMBOL     summary-gene-mouse.json" \
; do
    set -- $i
    echo $1
    echo $2
    echo $3
    echo
    echo
    python generate_summaries.py \
    --input_file "data/terms/$1" \
    --input_col "$2" \
    --prompt_file "prompts/$3" \
    --output_file "data/summaries/$1" \
    --api_url "https://api.openai.com/v1" \
    --api_key "$OPENAI_API_KEY" \
    --model "gpt-4o-2024-11-20" \
    --num_workers 8
done

