To run TRSeed on the AMiner dataset, execute the following commands in this order.

1. Generate Node2Vec embeddings

```bash
python generate_n2v_embeddings.py --dataset aminer --p_list "1" --q_list "0.5"
python main.py --dataset aminer --gnn TRSeed_APPNP --pool sum --Ks "[20]" --step 1 --runs 1 --gpu_id 0 --lamb 2 --n2v_path ./n2v/aminer_p1_q0.5_d512.kv --batch_size 2048 > .\logs\aminer\aminer_n2v_gpu_lamb2_p1_q0.5.log 2>&1

