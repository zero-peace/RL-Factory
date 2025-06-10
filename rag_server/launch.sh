#tmux new -s rag_server
#conda activate searchr1
#bash retrieval_launch.sh
#tmux detach
#bash train_ppo.sh

nvcc --version

file_path=/mnt/dolphinfs/hdd_pool/docker/share/xuzekun02/searchr1/dataset
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever=/mnt/dolphinfs/hdd_pool/docker/share/xuzekun02/huggingface.co/intfloat/e5-base-v2

python3 rag_server/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_model $retriever &
sleep 1200
