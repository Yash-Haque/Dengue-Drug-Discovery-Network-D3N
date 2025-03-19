export SAVE_DIR=D:/CSE498R_Resources/D3N/Dengue-Drug-Discovery-Network-D3N/Models/ner
export LOG_DIR=D:/CSE498R_Resources/D3N/Dengue-Drug-Discovery-Network-D3N/Logs/Pipeline/ner/train

export MAX_LENGTH=192
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SAVE_STEPS=1000
export ENTITY=jnlpba
export FORMAT=hf
export SPLIT=False
export SEED=1

python D:/CSE498R_Resources/D3N/Dengue-Drug-Discovery-Network-D3N/named-entity-recognition/run_ner.py \
    --data_dir ${ENTITY} \
    --model_name_or_path dmis-lab/biobert-base-cased-v1.1 \
    --output_dir ${SAVE_DIR}/${ENTITY} \
    --overwrite_output_dir True \
    --max_seq_length ${MAX_LENGTH} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --logging_dir ${LOG_DIR} \
    --save_strategy epoch \
    --logging_strategy steps \
    --eval_strategy epoch \
    --seed ${SEED} \
    --do_train \
    --do_eval \
    --do_predict \
    --load_best_model_at_end

