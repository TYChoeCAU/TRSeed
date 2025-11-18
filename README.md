python generate_n2v_embeddings.py --dataset aminer --p_list "1" --q_list "1"

python main.py --dataset aminer --gnn TRSeed_APPNP --pool sum --Ks "[20]" --step 1 --runs 1 --gpu_id 0 --lamb 1 --n2v_path ./n2v/aminer_p1_q1_d512.kv --batch_size 2048  > .\logs\aminer\aminer_n2v_gpu_lamb1p1q0.5.log 2>&1
