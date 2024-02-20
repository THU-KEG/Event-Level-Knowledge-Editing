MODELS=("gpt-j-6b")
FILES=("tendency_gen_predictions.json" "tendency_gen_serac_predictions.json" "tendency_gen_ft_predictions.json" "tendency_gen_bm25_predictions.json" "tendency_gen_e5_predictions.json")
for model in "${MODELS[@]}"
do
    for file in "${FILES[@]}"
    do
        test_file=../open-source/output/"$model"/examiner/"$file"
        echo "$test_file"
        python gpt-4.py --test_file "$test_file"
    done
done