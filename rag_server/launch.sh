#tmux new -s rag_server
#conda activate searchr1
#bash retrieval_launch.sh
#tmux detach
#bash train_ppo.sh

nvcc --version

file_path=/your/path/to/PeterGriffinJin/Search-R1/data/rag_data
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever=/your/path/to/PeterGriffinJin/Search-R1/huggingface.co/intfloat/e5-base-v2

python rag_server/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_model $retriever &
sleep 1200

# 执行后续请求
curl -X POST http://127.0.0.1:5003/retrieve \
  -H "Content-Type: application/json" \
  -d '{
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
      }'