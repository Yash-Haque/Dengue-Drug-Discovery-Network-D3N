export SAVE_DIR=D:/CSE498R_Resources/D3N/Dengue-Drug-Discovery-Network-D3N/Data/output/ner
export LOG_DIR=D:/CSE498R_Resources/D3N/Dengue-Drug-Discovery-Network-D3N/Logs/Pipeline/ner/predict

export MAX_LENGTH=192
export BATCH_SIZE=1
export NUM_EPOCHS=30
export SAVE_STEPS=1000
export ENTITY=jnlpba
export FORMAT=hf
export SPIT=True
export SEED=1

python run_ner.py \
    --data_dir D:/CSE498R_Resources/D3N/Dengue-Drug-Discovery-Network-D3N/Data/output/unfiltered/segmented_unfiltered.csv \
    --model_name_or_path D:/CSE498R_Resources/D3N/Dengue-Drug-Discovery-Network-D3N/Models/ner/jnlpba/JNLPBA_ner_model_v1.1.0/ \
    --output_dir ${SAVE_DIR}/${ENTITY} \
    --max_seq_length ${MAX_LENGTH} \
    --overwrite_cache True \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --save_steps ${SAVE_STEPS} \
    --seed ${SEED} \
    --do_predict \
    --overwrite_output_dir

