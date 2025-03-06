for i in \
"cells.csv       CELL_TYPE                      cells.npy                  CELL_TYPE" \
"cells.csv       CELL_TYPE_summarized           summarized_cells.npy       CELL_TYPE" \
"methods.csv     LIBRARY_METHODOLOGY            methods.npy                LIBRARY_METHODOLOGY" \
"methods.csv     LIBRARY_METHODOLOGY_summarized summarized_methods.npy     LIBRARY_METHODOLOGY" \
"phenotypes.csv  PHENOTYPE_NOTES                phenotypes.npy             PHENOTYPE_NOTES" \
"phenotypes.csv  PHENOTYPE_NOTES_summarized     summarized_phenotypes.npy  PHENOTYPE_NOTES" \
"genes_human.csv OFFICIAL_SYMBOL                genes_human.npy            IDENTIFIER_ID" \
"genes_human.csv OFFICIAL_SYMBOL_summarized     summarized_genes_human.npy IDENTIFIER_ID" \
"genes_mouse.csv OFFICIAL_SYMBOL                genes_mouse.npy            IDENTIFIER_ID" \
"genes_mouse.csv OFFICIAL_SYMBOL_summarized     summarized_genes_mouse.npy IDENTIFIER_ID" \
; do
    set -- $i
    echo $1 # in-file
    echo $2 # emb col
    echo $3 # out-file
    echo $4 # key col
    echo
    echo
    python generate_embeddings.py \
    --input_file "data/summaries/$1" \
    --input_col "$2" \
    --output_file "data/embeddings/$3" \
    --output_col "$4" \
    --api_url "https://api.openai.com/v1" \
    --api_key "$OPENAI_API_KEY" \
    --model "text-embedding-3-large" \
    --num_workers 0 \
    --batch_size 512 # need smaller batch size to not hit token limit for gene summaries
done
